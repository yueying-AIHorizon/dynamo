// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Job-size-aware effective work (JSEW) worker selection.
//!
//! Dynamo's built-in selector charges decode load as *footprint*: the KV blocks a worker's active
//! requests currently occupy. On long-context agentic traces footprint is a poor proxy for the
//! *work* those requests still owe. A 300k-token prompt that emits 24 tokens occupies an enormous
//! footprint while costing almost no decode work; a 45k prompt that emits 60k tokens pins a worker
//! for minutes on a modest footprint. Routing on footprint therefore steers new requests away from
//! workers that are nearly done and toward workers that are about to generate for a long time,
//! which shows up as tail end-to-end latency.
//!
//! This policy reproduces the built-in cost exactly and rescales only its decode term:
//!
//! ```text
//! unshared_blocks = decode_blocks * active_requests ^ decode_share_exponent
//! charged_blocks  = unshared_blocks * (unshared_blocks / mean_unshared_blocks) ^ (decode_load_exponent - 1)
//! decode_term     = decode_footprint_weight * charged_blocks
//!                 + decode_work_scale * (predicted_decode_blocks / request_blocks) ^ decode_work_exponent
//!                   * charged_blocks
//! ```
//!
//! `decode_footprint_weight` retains the stock footprint belief and `decode_work_scale` adds the
//! predicted-work belief, normalized by the incoming request's own size so the term is a
//! dimensionless ratio. `decode_footprint_weight: 1.0` with `decode_work_scale: 0.0` reproduces the
//! built-in selector's decision exactly.
//!
//! Two exponents shape the charge; both are neutral at `1.0`, where the expression above collapses
//! to the linear form and the policy is bit-for-bit what it was without them.
//!
//! - `decode_work_exponent` bends the size ratio. The ratio's denominator is the incoming request's
//!   own block count, so the term charges a short prompt far more per block of a worker's backlog
//!   than a long one. The exponent sets how sharply that sensitivity falls off with prompt size,
//!   which is a different question from how loud the term is overall (`decode_work_scale`).
//! - `decode_load_exponent` bends the charge in the worker's own backlog. Decode step time does not
//!   grow linearly in resident KV: past the batch size where the step goes memory-bound, another
//!   long session costs the whole batch far more than the one before it. A convex charge (`> 1.0`)
//!   makes the last increment of backlog on an already-loaded worker dominate, which is exactly the
//!   "past the point where decode step time explodes" boundary this policy exists to respect.
//!   Dividing by the candidate mean keeps the term's magnitude comparable to the prefill term it is
//!   summed against, so bending the curve does not silently re-weight the whole cost.
//! - `decode_share_exponent` (neutral at `0.0`) corrects a units mismatch upstream of both. The
//!   reported decode footprint counts physically resident blocks, which deduplicates the prefix
//!   that sibling sessions share; decode step time instead scales with the sum of their logical
//!   context lengths. A worker holding several sharing sessions therefore under-reports its load by
//!   roughly that count, and cache-affinity routing sends it more of exactly those sessions. This
//!   factor charges the un-shared sum back.
//!
//! Two predictors supply `predicted_decode_blocks`:
//!
//! - `ema_uncached_tokens` (default) tracks an exponential moving average of the *uncached* prompt
//!   tokens the router has already placed. It uses only what the router legitimately knows at
//!   admission time, so it is deployable.
//! - `ground_truth_future_decode` reads the caller-declared `expected_output_tokens`, falling back
//!   to the EMA when absent. In replay that is the recorded output length, making it an oracle
//!   upper bound on what any predictor can reach.
//!
//! The belief is resolved once per request rather than once per candidate: the EMA is stateful, so
//! folding it per candidate would make the prediction depend on how many workers happen to be
//! eligible.

use std::sync::Arc;

use dynamo_kv_router::services::selection::{
    WorkerSelectionPolicyFactory, WorkerSelectionPolicyParameters,
    WorkerSelectionPolicyProviderError, WorkerSelectionPolicyRegistry,
    WorkerSelectionPolicyRegistryError,
};
use dynamo_kv_router::{
    KvRouterConfig, ScoredWorkerCandidate, WorkerCacheInput, WorkerInputView, WorkerInputs,
    WorkerLoadInput, WorkerPicker, WorkerSelectionContext, WorkerSelectionPolicy,
    WorkerSelectionPolicyError,
};

/// Policy type selected by `worker_selection.instances[].type`.
pub const POLICY_TYPE: &str = "dynamo-jsew";

/// Defaults tuned on the `cc_agent_1200` agentic trace.
///
/// The predicted-work term wants to be loud (`17.6`) and the belief that feeds it wants to be
/// slow (`ema_alpha: 0.017`, a horizon of ~60 requests). That pairing is not a coincidence: on this
/// trace the uncached-prompt signal is extremely heavy-tailed, so a responsive EMA tracks
/// individual outliers and makes the decode charge lurch between requests. Averaging over many
/// requests instead turns the term into a stable estimate of how much decode work a typical
/// arrival owes, which is what the routing decision actually needs. The horizon is ~4x shorter
/// than it was before `decode_share_exponent` existed: once the pile-up the share term corrects is
/// priced directly, the predicted-work term no longer has to stand in for it, so the belief is free
/// to track the recent arrival mix rather than smoothing over the whole session.
///
/// `decode_load_exponent: 1.2` is the knob that moves the tail. Because the belief is nearly
/// constant, the linear form of the term reduces to a fixed multiple of each worker's decode
/// backlog, so it can only rank workers the stock cost already ranked the same way. Bending the
/// charge in the backlog itself is a genuinely new ordering: a worker twice as deep as the pool
/// mean is charged ~2.3x rather than 2x, so long sessions are pushed off the queues that are
/// already accumulating them. `decode_footprint_weight: 1.526` is then the linear counterweight
/// that keeps the convexity from over-draining lightly loaded workers.
///
/// Treat these as a set, not as four independently adjustable dials. The objective surface around
/// them is not a smooth bowl: what moves the tail is whether a handful of very long sessions land
/// on distinct workers, which is a discrete outcome, so the metric steps rather than slopes. The
/// exponent is the sharpest example -- at the previous `(17.0, 1.7, 0.004)` triple, a 1.2 exponent
/// was *bimodal*, mixing two distinct outcomes across repeats of the same seed, and the neighbours
/// at 1.21 and 1.22 were stable but worse. Re-tune the whole set jointly against a target trace
/// rather than hand-adjusting one value, and confirm any candidate across several seeds. Both
/// exponents were re-probed after `decode_share_exponent` landed and both held their previous
/// values, so the share correction is close to orthogonal to the two curvature knobs.
///
/// `decode_share_exponent: 0.13` is deliberately far below the `1.0` that the units argument alone
/// would suggest. Un-sharing the footprint is a correction to a *reported* quantity, and the
/// reported quantity is not uniformly wrong: `active_requests` counts every decoding session on the
/// worker, but only the ones that actually share a prefix are under-counted, and only for the
/// shared part of their context. Charging the full `active_requests` factor therefore overshoots
/// badly -- it turns the decode term into a request-count penalty that fights cache affinity
/// everywhere instead of only where sharing hides load. Measured on the target trace at 8 seeds, a
/// fractional exponent is both better and safer than a full one:
///
/// ```text
/// share exp   objective   p99 TTFT (limit 4281.9)   output tok/s
///      0.00     0.89475                    3741.7         2265.4
///      0.13     0.67857                    3903.1         2468.4
///      0.18     0.67414                    4128.1         2445.1
///      1.00     0.74563                    5330.4         2129.0
/// ```
///
/// The objective is flat from 0.12 to 0.18 and the binding constraint is prefill latency: pushing
/// decode work off the cache-warm worker forces recomputation, so p99 TTFT climbs monotonically
/// with the exponent and breaches above ~0.18. `0.13` takes nearly all of the available gain while
/// leaving ~9% TTFT headroom, which is the margin that survives a seed change -- 0.17 measured as a
/// breach at four seeds even though its neighbours at 0.16 and 0.18 did not.
const DEFAULT_DECODE_WORK_SCALE: f64 = 17.601;
const DEFAULT_DECODE_FOOTPRINT_WEIGHT: f64 = 1.526;
const DEFAULT_EMA_ALPHA: f64 = 0.01737;
const DEFAULT_DECODE_WORK_EXPONENT: f64 = 1.0;
const DEFAULT_DECODE_LOAD_EXPONENT: f64 = 1.2;
const DEFAULT_DECODE_SHARE_EXPONENT: f64 = 0.13;

/// Which belief supplies the predicted remaining decode work.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
enum EffectiveWorkPredictor {
    /// Exponential moving average over the uncached prompt tokens already routed.
    #[default]
    EmaUncachedTokens,
    /// The caller-declared `expected_output_tokens`, falling back to the EMA when absent.
    GroundTruthFutureDecode,
}

/// Tunables for [`POLICY_TYPE`].
///
/// Every field is optional and keeps the tuned default when omitted. Unknown keys are rejected at
/// startup rather than ignored, so a misremembered name fails loudly.
#[derive(Debug, Clone, Copy, serde::Deserialize)]
#[serde(deny_unknown_fields, default)]
struct Parameters {
    /// Which belief supplies the predicted remaining decode work.
    effective_work_predictor: EffectiveWorkPredictor,
    /// Weight on the predicted-work term, relative to the request's own block count. `0.0`
    /// disables the hypothesis and leaves the stock footprint belief in charge.
    decode_work_scale: f64,
    /// Weight retained on the stock decode-footprint term. `1.0` keeps it whole.
    decode_footprint_weight: f64,
    /// Smoothing factor for the EMA, in `(0, 1]`. `1.0` degenerates to "last routed request".
    ema_alpha: f64,
    /// Exponent on the predicted-work-to-request-size ratio. `1.0` leaves the ratio linear;
    /// below `1.0` flattens the term's sensitivity to prompt size, above `1.0` sharpens it.
    decode_work_exponent: f64,
    /// Exponent on the worker's decode backlog, normalized by the candidate mean. `1.0` charges
    /// backlog linearly; above `1.0` charges an already-loaded worker superlinearly.
    decode_load_exponent: f64,
    /// Exponent on the worker's active-request count, which un-shares the deduplicated decode
    /// footprint. `0.0` leaves the footprint as reported.
    decode_share_exponent: f64,
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
            effective_work_predictor: EffectiveWorkPredictor::default(),
            decode_work_scale: DEFAULT_DECODE_WORK_SCALE,
            decode_footprint_weight: DEFAULT_DECODE_FOOTPRINT_WEIGHT,
            ema_alpha: DEFAULT_EMA_ALPHA,
            decode_work_exponent: DEFAULT_DECODE_WORK_EXPONENT,
            decode_load_exponent: DEFAULT_DECODE_LOAD_EXPONENT,
            decode_share_exponent: DEFAULT_DECODE_SHARE_EXPONENT,
        }
    }
}

impl Parameters {
    fn validate(&self) -> Result<(), WorkerSelectionPolicyProviderError> {
        if !self.decode_work_scale.is_finite() || !(0.0..=20.0).contains(&self.decode_work_scale) {
            return Err(WorkerSelectionPolicyProviderError::new(
                "decode_work_scale must be a finite number between 0.0 and 20.0",
            ));
        }
        if !self.decode_footprint_weight.is_finite()
            || !(0.0..=2.0).contains(&self.decode_footprint_weight)
        {
            return Err(WorkerSelectionPolicyProviderError::new(
                "decode_footprint_weight must be a finite number between 0.0 and 2.0",
            ));
        }
        if !self.ema_alpha.is_finite() || self.ema_alpha <= 0.0 || self.ema_alpha > 1.0 {
            return Err(WorkerSelectionPolicyProviderError::new(
                "ema_alpha must be a finite number in (0.0, 1.0]",
            ));
        }
        // Both exponents are bounded away from zero: an exponent of zero collapses its factor to a
        // constant, which silently deletes the term rather than reshaping it.
        if !self.decode_work_exponent.is_finite()
            || !(0.1..=4.0).contains(&self.decode_work_exponent)
        {
            return Err(WorkerSelectionPolicyProviderError::new(
                "decode_work_exponent must be a finite number between 0.1 and 4.0",
            ));
        }
        if !self.decode_load_exponent.is_finite()
            || !(0.1..=4.0).contains(&self.decode_load_exponent)
        {
            return Err(WorkerSelectionPolicyProviderError::new(
                "decode_load_exponent must be a finite number between 0.1 and 4.0",
            ));
        }
        // Zero is the neutral value here, not a degenerate one: it leaves the reported footprint
        // untouched rather than deleting a factor.
        if !self.decode_share_exponent.is_finite()
            || !(0.0..=2.0).contains(&self.decode_share_exponent)
        {
            return Err(WorkerSelectionPolicyProviderError::new(
                "decode_share_exponent must be a finite number between 0.0 and 2.0",
            ));
        }
        Ok(())
    }
}

/// Cross-request belief for the EMA predictor.
///
/// Seeded from the first observation rather than from zero, so the very first requests are not
/// routed against a belief that no traffic has informed.
#[derive(Debug, Default)]
struct EmaBelief {
    uncached_tokens: Option<f64>,
}

impl EmaBelief {
    fn observe(&mut self, uncached_tokens: f64, alpha: f64) -> f64 {
        let updated = match self.uncached_tokens {
            Some(previous) => alpha * uncached_tokens + (1.0 - alpha) * previous,
            None => uncached_tokens,
        };
        self.uncached_tokens = Some(updated);
        updated
    }
}

/// The decode-cost rescaling this policy applies, resolved once per request.
#[derive(Debug, Clone, Copy)]
struct DecodeWorkBelief {
    predicted_decode_blocks: f64,
    mean_decode_blocks: f64,
    decode_work_scale: f64,
    decode_footprint_weight: f64,
    decode_work_exponent: f64,
    decode_load_exponent: f64,
    decode_share_exponent: f64,
}

impl DecodeWorkBelief {
    /// Bend the worker's decode backlog around the candidate mean.
    ///
    /// Normalizing by the mean keeps the charge's magnitude comparable to the linear one at any
    /// exponent, so `decode_load_exponent` reshapes which worker looks cheapest without also
    /// re-weighting the decode term against the prefill term it is summed with.
    fn charged_blocks(self, decode_cost_blocks: f64) -> f64 {
        // An empty decode queue costs nothing at any exponent, and short-circuiting it keeps the
        // `0 ^ negative` pole out of the normalization below.
        if self.decode_load_exponent == 1.0
            || decode_cost_blocks <= 0.0
            || self.mean_decode_blocks <= 0.0
        {
            return decode_cost_blocks;
        }
        decode_cost_blocks
            * (decode_cost_blocks / self.mean_decode_blocks).powf(self.decode_load_exponent - 1.0)
    }

    /// Undo the deduplication in the reported decode footprint.
    ///
    /// `decode_cost_blocks` counts *physically resident* blocks, so sessions sharing a prefix are
    /// counted once no matter how many of them are decoding. Decode step time is driven by the sum
    /// of their *logical* context lengths instead, which is the shared prefix counted once per
    /// session. A worker holding `n` sharing sessions therefore reports roughly `1/n` of the load
    /// it actually imposes, and the stock cost routes yet more of them onto it -- the pile-up this
    /// policy exists to prevent. Scaling by `active_requests ^ decode_share_exponent` recovers that
    /// factor: `0.0` trusts the reported footprint, `1.0` charges the fully un-shared sum.
    fn unshared_blocks(self, decode_cost_blocks: f64, active_requests: usize) -> f64 {
        if self.decode_share_exponent == 0.0 || decode_cost_blocks <= 0.0 {
            return decode_cost_blocks;
        }
        decode_cost_blocks * (active_requests.max(1) as f64).powf(self.decode_share_exponent)
    }

    fn decode_cost_blocks(
        self,
        decode_cost_blocks: f64,
        request_blocks: u64,
        active_requests: usize,
    ) -> f64 {
        let decode_cost_blocks = self.unshared_blocks(decode_cost_blocks, active_requests);
        let charged_blocks = self.charged_blocks(decode_cost_blocks);
        let ratio = self.predicted_decode_blocks / request_blocks.max(1) as f64;
        let shaped_ratio = if self.decode_work_exponent == 1.0 {
            ratio
        } else {
            ratio.powf(self.decode_work_exponent)
        };
        self.decode_footprint_weight * charged_blocks
            + self.decode_work_scale * shaped_ratio * charged_blocks
    }
}

/// Reproduce the built-in selector's per-worker logit, rescaling only its decode term.
///
/// Every input here is read from the host so the two costs stay in lockstep: any drift would make
/// a probe measure this re-implementation rather than the decode-work hypothesis.
#[allow(clippy::too_many_arguments)]
fn worker_logit(
    config: &KvRouterConfig,
    context: &WorkerSelectionContext<'_>,
    worker_label: &str,
    belief: DecodeWorkBelief,
    cache: &WorkerCacheInput,
    load: &WorkerLoadInput,
    min_active_prefill_tokens: usize,
    reports_tier_overlap: bool,
) -> f64 {
    let overlap_score_credit = context.overlap_score_credit();
    let overlap_score_credit_decay = context.overlap_score_credit_decay();

    // Without a per-tier breakdown the host reports zero device blocks and the built-in selector
    // credits the tier-weighted total instead.
    let device_overlap_blocks = if reports_tier_overlap {
        cache.device_overlap_blocks()
    } else {
        cache.effective_overlap_blocks()
    };

    // Normalize backlog above the least-loaded eligible worker by this request's size. The
    // rational decay softly trades cache locality for prefill balance, while leaving workers at
    // the load floor with their full device credit.
    let overlap_credit_decay =
        if context.tracks_prefill_tokens() && overlap_score_credit_decay > 0.0 {
            let excess_active_prefill_blocks = load
                .active_prefill_tokens()
                .saturating_sub(min_active_prefill_tokens)
                as f64
                / context.block_size() as f64;
            let normalized_prefill_load =
                excess_active_prefill_blocks / context.request_blocks() as f64;
            1.0 / (1.0 + overlap_score_credit_decay * normalized_prefill_load)
        } else {
            1.0
        };

    let overlap_credit_blocks = overlap_score_credit * overlap_credit_decay * device_overlap_blocks
        + config.host_cache_hit_weight * cache.host_overlap_blocks()
        + config.disk_cache_hit_weight * cache.disk_overlap_blocks()
        + context.shared_cache_multiplier() * cache.shared_beyond_device_blocks() as f64;

    // The only departure from the built-in cost.
    let decode_cost_blocks = belief.decode_cost_blocks(
        load.decode_cost_blocks(),
        context.request_blocks(),
        load.active_requests(),
    );
    let active_request_cost_blocks =
        config.decode_active_request_weight * load.active_requests() as f64;

    // Decode routers normally force `overlap_score_credit=0`, which preserves load-only disagg
    // routing. When conditional disagg leaves a positive credit in place, the built-in selector
    // prefers cache-hot decode workers while still charging decode backlog.
    if worker_label == "decode" && !context.tracks_prefill_tokens() && overlap_score_credit > 0.0 {
        // Clamped at zero for the same reason the built-in selector clamps: downstream taint
        // multipliers assume non-negative scores.
        return (decode_cost_blocks - overlap_credit_blocks).max(0.0) + active_request_cost_blocks;
    }

    let adjusted_prefill_blocks = (load.raw_prefill_blocks() - overlap_credit_blocks).max(0.0);
    context.prefill_load_scale() * adjusted_prefill_blocks
        + decode_cost_blocks
        + active_request_cost_blocks
}

/// Sample a row from the cost distribution the built-in selector would sample at `temperature`.
fn softmax_sample_row(costs: &[f64], temperature: f64, sample: f64) -> usize {
    let (min_cost, max_cost) = costs
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), cost| {
            (lo.min(*cost), hi.max(*cost))
        });

    let mut probabilities = Vec::with_capacity(costs.len());
    if min_cost == max_cost {
        probabilities.resize(costs.len(), 1.0 / costs.len() as f64);
    } else {
        let range = max_cost - min_cost;
        let magnitude = if range.is_finite() {
            1.0
        } else {
            min_cost.abs().max(max_cost.abs())
        };
        let min_normalized = min_cost / magnitude;
        let scale = -1.0 / ((max_cost / magnitude - min_normalized) * temperature);
        let max_scaled = min_normalized * scale;
        probabilities.extend(
            costs
                .iter()
                .map(|cost| (cost / magnitude * scale - max_scaled).exp()),
        );
    }

    let sum: f64 = probabilities.iter().sum();
    let mut cumulative = 0.0;
    for (row, probability) in probabilities.iter().enumerate() {
        cumulative += probability / sum;
        if sample <= cumulative {
            return row;
        }
    }
    costs.len() - 1
}

struct JsewPicker {
    config: KvRouterConfig,
    worker_label: &'static str,
    parameters: Parameters,
    belief: EmaBelief,
    costs: Vec<f64>,
}

impl JsewPicker {
    /// Resolve the decode-work belief for one request.
    ///
    /// `best_cached_tokens` is the longest prompt prefix any eligible worker can skip, so
    /// `isl_tokens` minus it is the uncached prompt this request actually pays for.
    fn resolve_belief(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        best_cached_tokens: usize,
        mean_decode_blocks: f64,
    ) -> DecodeWorkBelief {
        let uncached_tokens = context.isl_tokens().saturating_sub(best_cached_tokens) as f64;
        // Folded unconditionally so the EMA stays current even when the oracle supplies the
        // prediction, keeping the fallback usable for requests that declare no output length.
        let ema_tokens = self
            .belief
            .observe(uncached_tokens, self.parameters.ema_alpha);
        let predicted_decode_tokens = match self.parameters.effective_work_predictor {
            EffectiveWorkPredictor::GroundTruthFutureDecode => context
                .expected_output_tokens()
                .map(f64::from)
                .unwrap_or(ema_tokens),
            EffectiveWorkPredictor::EmaUncachedTokens => ema_tokens,
        };
        DecodeWorkBelief {
            predicted_decode_blocks: predicted_decode_tokens
                / f64::from(context.block_size().max(1)),
            mean_decode_blocks,
            decode_work_scale: self.parameters.decode_work_scale,
            decode_footprint_weight: self.parameters.decode_footprint_weight,
            decode_work_exponent: self.parameters.decode_work_exponent,
            decode_load_exponent: self.parameters.decode_load_exponent,
            decode_share_exponent: self.parameters.decode_share_exponent,
        }
    }
}

impl WorkerPicker for JsewPicker {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::CACHE | WorkerInputs::LOAD | WorkerInputs::PREFERRED_TAINT
    }

    fn pick(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        let candidates = input.candidates();
        let cache = input
            .cache()
            .ok_or_else(|| WorkerSelectionPolicyError::failed("cache input unavailable"))?;
        let load = input
            .load()
            .ok_or_else(|| WorkerSelectionPolicyError::failed("load input unavailable"))?;
        if candidates.is_empty() || cache.len() != candidates.len() || load.len() != candidates.len()
        {
            return Err(WorkerSelectionPolicyError::failed("no eligible worker"));
        }

        let reports_tier_overlap = context.reports_tier_overlap_blocks();
        let block_size = f64::from(context.block_size());
        // The host derives cached tokens from the same rounding, so this recovers the value the
        // built-in selector's predictor observes.
        let best_cached_tokens = cache
            .iter()
            .map(|item| {
                let effective = if reports_tier_overlap {
                    item.device_overlap_blocks()
                        + self.config.host_cache_hit_weight * item.host_overlap_blocks()
                        + self.config.disk_cache_hit_weight * item.disk_overlap_blocks()
                } else {
                    item.effective_overlap_blocks()
                };
                (effective * block_size).round().max(0.0) as usize
            })
            .max()
            .unwrap_or(0);
        // The reference point `decode_load_exponent` bends around. Taken over the eligible set for
        // this request so the charge tracks how loaded the pool is right now, rather than against a
        // fixed scale that would make the exponent's effect drift with overall traffic.
        //
        // Taken over the un-shared blocks, since that is the quantity `charged_blocks` bends: a
        // mean on the reported scale would make the normalization drift with how much prefix
        // sharing the pool happens to hold.
        let share_exponent = self.parameters.decode_share_exponent;
        let mean_decode_blocks = if self.parameters.decode_load_exponent == 1.0 {
            0.0
        } else {
            load.iter()
                .map(|load| {
                    let blocks = load.decode_cost_blocks();
                    if share_exponent == 0.0 || blocks <= 0.0 {
                        blocks
                    } else {
                        blocks * (load.active_requests().max(1) as f64).powf(share_exponent)
                    }
                })
                .sum::<f64>()
                / load.len() as f64
        };
        let belief = self.resolve_belief(context, best_cached_tokens, mean_decode_blocks);

        let min_active_prefill_tokens =
            if context.tracks_prefill_tokens() && context.overlap_score_credit_decay() > 0.0 {
                load.iter()
                    .map(WorkerLoadInput::active_prefill_tokens)
                    .min()
                    .unwrap_or(0)
            } else {
                0
            };

        self.costs.clear();
        self.costs
            .extend(candidates.iter().zip(cache).zip(load).map(
                |((candidate, cache), load): ((&ScoredWorkerCandidate, _), _)| {
                    let logit = worker_logit(
                        &self.config,
                        context,
                        self.worker_label,
                        belief,
                        cache,
                        load,
                        min_active_prefill_tokens,
                        reports_tier_overlap,
                    );
                    match candidate.preferred_taint_multiplier() {
                        // Matches the built-in selector's multiplicative bias, which assumes a
                        // non-negative score.
                        Some(multiplier) => logit * multiplier,
                        None => logit,
                    }
                },
            ));

        let temperature = context
            .router_temperature_override()
            .unwrap_or(self.config.router_temperature);
        if temperature != 0.0 {
            return Ok(softmax_sample_row(&self.costs, temperature, fastrand::f64()));
        }

        // Break ties uniformly, as the built-in selector does, so equal-cost workers do not all
        // collapse onto whichever row the host happened to emit first.
        let mut best_row = 0;
        let mut best_cost = f64::INFINITY;
        let mut tie_count = 0;
        for (row, &cost) in self.costs.iter().enumerate() {
            if cost < best_cost {
                best_row = row;
                best_cost = cost;
                tie_count = 1;
            } else if cost == best_cost {
                tie_count += 1;
                if fastrand::usize(0..tie_count) == 0 {
                    best_row = row;
                }
            }
        }
        Ok(best_row)
    }
}

fn provider(
    parameters: &WorkerSelectionPolicyParameters,
) -> Result<WorkerSelectionPolicyFactory, WorkerSelectionPolicyProviderError> {
    let parameters: Parameters = parameters.deserialize()?;
    parameters.validate()?;

    Ok(Arc::new(
        move |config: &KvRouterConfig, worker_type, _partition| {
            // The built-in selector distinguishes only prefill pools from all others, and this
            // policy reproduces its cost, so it must label pools the same way.
            let worker_label = dynamo_kv_router::WorkerType::default_selector_label(&worker_type);
            WorkerSelectionPolicy::new(
                config.clone(),
                worker_label,
                Vec::new(),
                Box::new(JsewPicker {
                    config: config.clone(),
                    worker_label,
                    parameters,
                    belief: EmaBelief::default(),
                    costs: Vec::new(),
                }),
            )
        },
    ))
}

pub fn register(
    registry: &mut WorkerSelectionPolicyRegistry,
) -> Result<(), WorkerSelectionPolicyRegistryError> {
    registry.register(POLICY_TYPE, Arc::new(provider))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use dynamo_kv_router::protocols::{RoutingConstraints, WorkerConfigLike, WorkerWithDpRank};
    use dynamo_kv_router::scheduling::{OverlapSignals, ScheduleMode};
    use dynamo_kv_router::{
        DefaultWorkerSelector, SchedulingRequest, WorkerLoadProjection, WorkerSelectionInput,
        WorkerSelector,
    };

    use super::*;

    const BLOCK_SIZE: u32 = 16;
    /// Ten blocks of prompt.
    const TEN_BLOCKS: usize = 160;
    const A: u64 = 29;
    const B: u64 = 41;

    struct TestWorker;

    impl WorkerConfigLike for TestWorker {
        fn data_parallel_start_rank(&self) -> u32 {
            0
        }
        fn data_parallel_size(&self) -> u32 {
            1
        }
        fn max_num_batched_tokens(&self) -> Option<u64> {
            None
        }
        fn total_kv_blocks(&self) -> Option<u64> {
            Some(1_000_000)
        }
    }

    fn worker(id: u64) -> WorkerWithDpRank {
        WorkerWithDpRank::from_worker_id(id)
    }

    /// One worker's signals: `(worker_id, overlap_blocks, active_decode_blocks)`.
    type WorkerSpec = (u64, usize, usize);
    /// As [`WorkerSpec`], plus the worker's active request count.
    type SharedWorkerSpec = (u64, usize, usize, usize);

    /// Widen a [`WorkerSpec`] to one active request per worker, which is what the stock cost
    /// assumes when it charges a deduplicated footprint.
    fn unshared(workers: [WorkerSpec; 2]) -> [SharedWorkerSpec; 2] {
        workers.map(|(id, overlap, decode)| (id, overlap, decode, 1))
    }

    fn request(
        workers: [SharedWorkerSpec; 2],
        expected_output_tokens: Option<u32>,
    ) -> SchedulingRequest {
        let mut request = SchedulingRequest {
            mode: ScheduleMode::QueryOnly { request_id: None },
            token_seq: None,
            isl_tokens: TEN_BLOCKS,
            lora_name: None,
            expected_output_tokens,
            affinity_target: None,
            pinned_worker: None,
            allowed_worker_ids: None,
            routing_constraints: RoutingConstraints::default(),
            router_config_override: None,
            track_prefill_tokens: true,
            priority_jump: 0.0,
            strict_priority: 0,
            policy_class: None,
            session_context: None,
            overlap: OverlapSignals::default(),
            kv_transfer_candidates: None,
            retain_kv_transfer_chain: false,
            shared_cache_hits: None,
            worker_loads: Default::default(),
            resp_tx: None,
        };
        for (id, overlap_blocks, active_decode_blocks, active_requests) in workers {
            // Offline replay reports no per-tier breakdown, only the weighted total.
            request
                .overlap
                .effective_overlap_blocks
                .insert(worker(id), overlap_blocks as f64);
            request
                .overlap
                .effective_cached_tokens
                .insert(worker(id), overlap_blocks * BLOCK_SIZE as usize);
            request.worker_loads.insert(
                worker(id),
                WorkerLoadProjection {
                    active_decode_blocks,
                    active_requests,
                    ..Default::default()
                },
            );
        }
        request
    }

    fn configs(workers: [SharedWorkerSpec; 2]) -> HashMap<u64, TestWorker> {
        HashMap::from(workers.map(|(id, _, _, _)| (id, TestWorker)))
    }

    /// Route one request through this policy and return the worker it picks.
    fn select_with(parameters: Parameters, workers: [WorkerSpec; 2]) -> WorkerWithDpRank {
        select_seq(parameters, [unshared(workers)], None)[0]
    }

    /// Route a sequence of requests through one policy instance, so the EMA carries across them.
    fn select_seq<const N: usize>(
        parameters: Parameters,
        requests: [[SharedWorkerSpec; 2]; N],
        expected_output_tokens: Option<u32>,
    ) -> Vec<WorkerWithDpRank> {
        let config = KvRouterConfig::default();
        let policy = WorkerSelectionPolicy::new(
            config.clone(),
            "decode",
            Vec::new(),
            Box::new(JsewPicker {
                config,
                worker_label: "decode",
                parameters,
                belief: EmaBelief::default(),
                costs: Vec::new(),
            }),
        );
        requests
            .into_iter()
            .map(|workers| {
                let request = request(workers, expected_output_tokens);
                let configs = configs(workers);
                policy
                    .select_worker(WorkerSelectionInput::configured(
                        &configs,
                        &request,
                        request.eligibility(),
                        BLOCK_SIZE,
                    ))
                    .unwrap()
                    .worker
            })
            .collect()
    }

    /// Route the same request through Dynamo's built-in selector.
    fn select_stock(workers: [WorkerSpec; 2]) -> WorkerWithDpRank {
        let workers = unshared(workers);
        let request = request(workers, None);
        let configs = configs(workers);
        DefaultWorkerSelector::new(Some(KvRouterConfig::default()), "decode")
            .select_worker(WorkerSelectionInput::configured(
                &configs,
                &request,
                request.eligibility(),
                BLOCK_SIZE,
            ))
            .unwrap()
            .worker
    }

    /// Neutral parameters: keep the stock footprint term whole, add nothing on top.
    ///
    /// The exponents are pinned to 1.0 rather than inherited, since the tuned default bends the
    /// charge and neutrality here means "identical to the stock cost", not "the shipped setting".
    fn neutral() -> Parameters {
        Parameters {
            decode_work_scale: 0.0,
            decode_footprint_weight: 1.0,
            decode_work_exponent: 1.0,
            decode_load_exponent: 1.0,
            ..Parameters::default()
        }
    }

    #[test]
    fn registers_its_policy_type() {
        let mut registry = WorkerSelectionPolicyRegistry::default();
        register(&mut registry).unwrap();
        assert_eq!(POLICY_TYPE, "dynamo-jsew");
        // Re-registering the same type is what a duplicated catalog entry would do.
        assert!(register(&mut registry).is_err());
    }

    #[test]
    fn neutral_parameters_reproduce_the_built_in_choice() {
        // Cases where cache overlap and decode footprint disagree, so any drift in the ported cost
        // would show up as a different worker.
        for workers in [
            [(A, 0, 0), (B, 8, 40)],
            [(A, 6, 30), (B, 0, 0)],
            [(A, 10, 100), (B, 2, 5)],
            [(A, 3, 12), (B, 4, 20)],
            [(A, 0, 7), (B, 5, 9)],
        ] {
            assert_eq!(
                select_with(neutral(), workers),
                select_stock(workers),
                "diverged from the built-in selector on {workers:?}"
            );
        }
    }

    #[test]
    fn predicted_work_outweighs_cache_affinity() {
        // B holds five of the request's ten blocks but carries more decode backlog than A. Stock
        // cache credit outweighs those four extra decode blocks, and the neutral policy agrees.
        let workers = [(A, 0, 30), (B, 5, 34)];
        assert_eq!(select_with(neutral(), workers), worker(B));

        // Scaling the predicted-work term amplifies the decode gap until it swamps the five blocks
        // of cache credit, so the request goes to the worker that owes less work.
        let tuned = Parameters {
            decode_work_scale: 20.0,
            ..neutral()
        };
        assert_eq!(select_with(tuned, workers), worker(A));
    }

    #[test]
    fn dropping_the_footprint_weight_restores_cache_affinity() {
        // With the footprint term whole, A's empty decode queue outweighs B's full cache hit.
        let workers = [(A, 0, 0), (B, 10, 12)];
        assert_eq!(select_with(neutral(), workers), worker(A));

        // Discarding the footprint belief leaves only prefill load, so the cache hit decides.
        let tuned = Parameters {
            decode_footprint_weight: 0.0,
            decode_work_scale: 0.0,
            ..Parameters::default()
        };
        assert_eq!(select_with(tuned, workers), worker(B));
    }

    #[test]
    fn the_ema_belief_carries_across_requests() {
        // One request, routed twice against identical worker state but different prior traffic.
        let request = [(A, 0, 30), (B, 5, 34)];
        // Fully cached: this warm-up owes almost no prefill, so it teaches the EMA that traffic is
        // cheap.
        let warm_cache = [(A, 10, 0), (B, 10, 0)];
        // Fully uncached: the same warm-up shape, but it teaches the EMA that traffic is expensive.
        let cold_cache = [(A, 0, 0), (B, 0, 0)];

        // The tuned EMA is slow enough that one request cannot move it, so the warm-up decides
        // what the policy believes the next request's neighbours owe. Built on the neutral
        // footprint weight so this exercises the predicted-work term alone.
        let parameters = Parameters {
            decode_work_scale: 20.0,
            ..neutral()
        };

        // After cheap traffic the predicted-work term stays small and cache credit decides.
        assert_eq!(
            select_seq(parameters, [unshared(warm_cache), unshared(request)], None)[1],
            worker(B)
        );
        // After expensive traffic the same term dominates, and the request avoids B's backlog.
        assert_eq!(
            select_seq(parameters, [unshared(cold_cache), unshared(request)], None)[1],
            worker(A)
        );
    }

    #[test]
    fn ground_truth_predictor_reads_the_declared_output_length() {
        let workers = [(A, 0, 30), (B, 5, 34)];
        let ground_truth = Parameters {
            effective_work_predictor: EffectiveWorkPredictor::GroundTruthFutureDecode,
            decode_work_scale: 20.0,
            ..neutral()
        };

        // A short declared completion leaves the predicted-work term small, so cache credit wins.
        assert_eq!(select_seq(ground_truth, [unshared(workers)], Some(1))[0], worker(B));
        // A long one charges decode backlog hard enough to flip the choice.
        assert_eq!(
            select_seq(ground_truth, [unshared(workers)], Some(100_000))[0],
            worker(A)
        );
    }

    #[test]
    fn a_convex_load_exponent_avoids_the_deeper_backlog() {
        // A holds no cache and a small backlog; B holds half the prompt and three more blocks of
        // backlog. Charged linearly, B's five blocks of cache credit cover those three blocks.
        let workers = [(A, 0, 8), (B, 5, 11)];
        assert_eq!(select_with(neutral(), workers), worker(B));

        // Bending the charge in the worker's own backlog costs B's deeper queue disproportionately
        // more than A's, which is enough to outweigh the cache hit.
        let convex = Parameters {
            decode_load_exponent: 3.0,
            ..neutral()
        };
        assert_eq!(select_with(convex, workers), worker(A));
    }

    #[test]
    fn the_work_exponent_reshapes_the_size_ratio() {
        // B holds five of the request's ten blocks but owes four more blocks of decode. A half-
        // cached warm-up leaves the EMA believing an arrival owes five blocks of work against this
        // request's ten, so the size ratio sits at 0.5 -- the regime the exponent acts on.
        let workers = [(A, 0, 30), (B, 5, 34)];
        let warm_cache = [(A, 5, 0), (B, 5, 0)];

        // Charged linearly, that half-strength work term still swamps B's cache credit.
        let linear = Parameters {
            decode_work_scale: 3.0,
            ..neutral()
        };
        assert_eq!(select_seq(linear, [unshared(warm_cache), unshared(workers)], None)[1], worker(A));

        // Sharpening the exponent drives a sub-unit ratio toward zero, quieting the work term for
        // requests larger than the believed arrival and handing the choice back to cache credit.
        // The scale is untouched, so this is a change in shape, not in loudness.
        let sharpened = Parameters {
            decode_work_exponent: 4.0,
            ..linear
        };
        assert_eq!(
            select_seq(sharpened, [unshared(warm_cache), unshared(workers)], None)[1],
            worker(B)
        );
    }

    #[test]
    fn neutral_exponents_leave_the_charge_untouched() {
        // The exponents are new knobs on a tuned policy, so their neutral value has to be exactly
        // inert: every previously reachable decision must still be reachable at 1.0.
        for workers in [
            [(A, 0, 0), (B, 8, 40)],
            [(A, 6, 30), (B, 0, 0)],
            [(A, 10, 100), (B, 2, 5)],
            [(A, 3, 12), (B, 4, 20)],
            [(A, 0, 7), (B, 5, 9)],
        ] {
            let belief = DecodeWorkBelief {
                predicted_decode_blocks: 37.0,
                mean_decode_blocks: 22.0,
                decode_work_scale: DEFAULT_DECODE_WORK_SCALE,
                decode_footprint_weight: DEFAULT_DECODE_FOOTPRINT_WEIGHT,
                decode_work_exponent: 1.0,
                decode_load_exponent: 1.0,
                decode_share_exponent: 0.0,
            };
            for (_, _, decode_blocks) in workers {
                let blocks = decode_blocks as f64;
                let expected = DEFAULT_DECODE_FOOTPRINT_WEIGHT * blocks
                    + DEFAULT_DECODE_WORK_SCALE * (37.0 / 10.0) * blocks;
                // Any active-request count, since the share exponent is inert at 0.0.
                for active_requests in [0, 1, 7] {
                    assert_eq!(
                        belief.decode_cost_blocks(blocks, 10, active_requests),
                        expected
                    );
                }
            }
        }
    }

    #[test]
    fn the_share_exponent_un_shares_a_deduplicated_footprint() {
        // Both workers report the same decode footprint, but B's is the deduplicated total of six
        // sessions sharing a prefix while A's belongs to a single session. B also holds half this
        // request's prompt, so cache credit sends the request to B on the reported numbers.
        let workers: [SharedWorkerSpec; 2] = [(A, 0, 20, 1), (B, 5, 20, 6)];
        let neutral = Parameters {
            decode_work_scale: 0.0,
            decode_footprint_weight: 1.0,
            decode_work_exponent: 1.0,
            decode_load_exponent: 1.0,
            decode_share_exponent: 0.0,
            ..Parameters::default()
        };
        assert_eq!(select_seq(neutral, [workers], None)[0], worker(B));

        // Charging the un-shared sum reveals that B's six sessions each carry that prefix through
        // every decode step, so the request avoids piling a seventh onto it.
        let unshare = Parameters {
            decode_share_exponent: 1.0,
            ..neutral
        };
        assert_eq!(select_seq(unshare, [workers], None)[0], worker(A));
    }

    #[test]
    fn rejects_out_of_range_parameters() {
        let scale = |v| {
            Parameters {
                decode_work_scale: v,
                ..Default::default()
            }
            .validate()
        };
        let footprint = |v| {
            Parameters {
                decode_footprint_weight: v,
                ..Default::default()
            }
            .validate()
        };
        let alpha = |v| {
            Parameters {
                ema_alpha: v,
                ..Default::default()
            }
            .validate()
        };

        let work_exponent = |v| {
            Parameters {
                decode_work_exponent: v,
                ..Default::default()
            }
            .validate()
        };
        let load_exponent = |v| {
            Parameters {
                decode_load_exponent: v,
                ..Default::default()
            }
            .validate()
        };

        assert!(scale(-0.1).is_err() && scale(20.1).is_err() && scale(f64::NAN).is_err());
        assert!(footprint(-0.1).is_err() && footprint(2.1).is_err());
        assert!(alpha(0.0).is_err() && alpha(1.1).is_err() && alpha(f64::NAN).is_err());
        let share_exponent = |v| {
            Parameters {
                decode_share_exponent: v,
                ..Default::default()
            }
            .validate()
        };

        assert!(work_exponent(0.0).is_err() && work_exponent(4.1).is_err());
        assert!(load_exponent(0.0).is_err() && load_exponent(f64::NAN).is_err());
        // Zero is the share exponent's neutral value, so it must be accepted.
        assert!(share_exponent(0.0).is_ok());
        assert!(share_exponent(-0.1).is_err() && share_exponent(2.1).is_err());
        assert!(Parameters::default().validate().is_ok());
    }
}
