// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Join-shortest-expected-work (JSEW) routing.
//!
//! # Why
//!
//! The stock worker logit charges decode load as `decode_cost_blocks`, the KV
//! *footprint* a worker currently holds. Footprint is a good proxy for memory
//! pressure but a poor proxy for how long a worker stays busy. On long-context
//! agentic traces the two diverge by orders of magnitude:
//!
//! - a 300k-token prompt with a 24-token completion holds a huge footprint but
//!   releases it almost immediately;
//! - a 45k-token prompt with 60k output tokens holds a small footprint but pins
//!   the worker for minutes.
//!
//! Routing on footprint alone therefore steers short requests away from workers
//! that are about to go idle, and piles long-decode requests onto workers that
//! merely *look* cheap. The result is a heavy time-to-last-token (TTLT) tail.
//!
//! # What JSEW changes
//!
//! JSEW adds one interaction term to the logit, scaled by a *prediction* of how
//! much decode work the incoming request itself will generate:
//!
//! ```text
//! interaction_blocks = jsew_decode_work_scale
//!                    * (predicted_decode_blocks / request_blocks)
//!                    * decode_cost_blocks
//! ```
//!
//! `predicted_decode_blocks / request_blocks` is this request's decode-to-prefill
//! ratio: dimensionless, near zero for prefill-dominated requests and large for
//! decode-dominated ones. Multiplying it into the worker's decode backlog makes
//! the cache-affinity/load-balance trade *per request* rather than global:
//!
//! - prefill-dominated request -> interaction collapses to zero -> the logit is
//!   dominated by the cache-credit term, so the request chases its cache hit;
//! - decode-dominated request -> interaction dominates -> the request is steered
//!   to whichever worker has the least decode backlog, because that is what will
//!   actually determine its TTLT.
//!
//! `jsew_decode_footprint_weight` independently scales the residual stock
//! footprint term, so the operator can keep it (1.0, the default), drop it
//! entirely (0.0), or blend.
//!
//! Everything here is inert unless [`KvRouterConfig::jsew_enabled`] is true.
//! With the gate off `jsew_decode_adjustment` returns `None` and the caller must
//! compute the stock logit unchanged.

use parking_lot::Mutex;

use crate::scheduling::config::{EffectiveWorkPredictor, KvRouterConfig};
use crate::scheduling::types::SchedulingRequest;

/// Cross-request state the [`EffectiveWorkPredictor::EmaUncachedTokens`]
/// predictor needs.
///
/// Held behind a mutex on the selector because an exponential moving average is
/// inherently stateful. Selection itself stays free of *routing* side effects:
/// this touches no queue, capacity reservation, or prompt registry, and the
/// update is a single f64 blend. The state is shared across clones of a
/// selector so a cloned router keeps one consistent belief.
///
/// Untouched, and never locked, unless `jsew_enabled` is true and the EMA
/// predictor is selected.
#[derive(Debug, Default)]
pub(super) struct JsewEmaState {
    /// Smoothed uncached prompt tokens across routed requests. `None` until the
    /// first request is routed, at which point `jsew_ema_seed_tokens` seeds it.
    uncached_tokens: Mutex<Option<f64>>,
}

impl JsewEmaState {
    /// Fold one request's uncached prompt-token count into the moving average
    /// and return the updated belief, in tokens.
    fn observe(&self, uncached_tokens: f64, alpha: f64, seed_tokens: f64) -> f64 {
        let mut slot = self.uncached_tokens.lock();
        let previous = slot.unwrap_or(seed_tokens);
        let updated = alpha * uncached_tokens + (1.0 - alpha) * previous;
        *slot = Some(updated);
        updated
    }
}

/// The per-request prediction JSEW scores against, resolved once per selection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct JsewPrediction {
    /// Predicted decode work this request will generate, in blocks.
    predicted_decode_blocks: f64,
    /// Weight on the predicted-work interaction term.
    decode_work_scale: f64,
    /// Residual weight on the stock KV-footprint term.
    decode_footprint_weight: f64,
}

/// Resolve the JSEW prediction for one request, or `None` when JSEW is off.
///
/// `best_cached_tokens` is the largest effective cached-token count across the
/// workers this request may land on; `isl_tokens - best_cached_tokens` is the
/// uncached prompt the EMA predictor observes.
pub(super) fn resolve_prediction(
    kv_router_config: &KvRouterConfig,
    request: &SchedulingRequest,
    ema_state: &JsewEmaState,
    best_cached_tokens: usize,
    block_size: u32,
) -> Option<JsewPrediction> {
    if !kv_router_config.jsew_enabled {
        return None;
    }

    let predicted_decode_tokens = match kv_router_config.effective_work_predictor {
        EffectiveWorkPredictor::GroundTruthFutureDecode => {
            // Absent declaration means "no belief", which must not be confused
            // with "predicted to decode nothing": fall back to the EMA so the
            // interaction term stays populated for undeclared requests.
            match request.expected_output_tokens {
                Some(tokens) => f64::from(tokens),
                None => ema_observation(kv_router_config, request, ema_state, best_cached_tokens),
            }
        }
        EffectiveWorkPredictor::EmaUncachedTokens => {
            ema_observation(kv_router_config, request, ema_state, best_cached_tokens)
        }
    };

    Some(JsewPrediction {
        predicted_decode_blocks: predicted_decode_tokens / f64::from(block_size.max(1)),
        decode_work_scale: kv_router_config.jsew_decode_work_scale,
        decode_footprint_weight: kv_router_config.jsew_decode_footprint_weight,
    })
}

fn ema_observation(
    kv_router_config: &KvRouterConfig,
    request: &SchedulingRequest,
    ema_state: &JsewEmaState,
    best_cached_tokens: usize,
) -> f64 {
    let uncached_tokens = request.isl_tokens.saturating_sub(best_cached_tokens) as f64;
    ema_state.observe(
        uncached_tokens,
        kv_router_config.jsew_ema_alpha,
        kv_router_config.jsew_ema_seed_tokens,
    )
}

impl JsewPrediction {
    /// Replace the stock `decode_cost_blocks` contribution with the JSEW one.
    ///
    /// `request_blocks` is the incoming request's prompt size in blocks and is
    /// always at least 1 for a real request (`select_worker_with_policy`
    /// asserts `isl_tokens > 0`), but it is clamped here so a synthetic
    /// zero-length request cannot divide by zero.
    pub(super) fn decode_cost_blocks(self, decode_cost_blocks: f64, request_blocks: u64) -> f64 {
        let decode_to_prefill_ratio =
            self.predicted_decode_blocks / (request_blocks.max(1) as f64);
        self.decode_footprint_weight * decode_cost_blocks
            + self.decode_work_scale * decode_to_prefill_ratio * decode_cost_blocks
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_support::*;
    use super::*;

    fn config(jsew_enabled: bool) -> KvRouterConfig {
        KvRouterConfig {
            jsew_enabled,
            ..Default::default()
        }
    }

    fn request(isl_tokens: usize, expected_output_tokens: Option<u32>) -> SchedulingRequest {
        let mut request = base_request(isl_tokens);
        request.expected_output_tokens = expected_output_tokens;
        request
    }

    #[test]
    fn disabled_gate_yields_no_prediction() {
        let state = JsewEmaState::default();
        assert!(
            resolve_prediction(&config(false), &request(1024, Some(512)), &state, 0, 64).is_none()
        );
        // The disabled gate must not even touch EMA state, so a later enabled
        // call sees a virgin average.
        assert!(state.uncached_tokens.lock().is_none());
    }

    #[test]
    fn neutral_knobs_reproduce_the_stock_decode_term() {
        // scale=0, footprint=1 is the identity configuration: turning the gate
        // on with neutral knobs must leave the decode term byte-identical.
        let mut kv_router_config = config(true);
        kv_router_config.jsew_decode_work_scale = 0.0;
        kv_router_config.jsew_decode_footprint_weight = 1.0;
        let state = JsewEmaState::default();
        let prediction =
            resolve_prediction(&kv_router_config, &request(1024, Some(4096)), &state, 0, 64)
                .expect("gate is on");
        for decode_cost_blocks in [0.0, 1.0, 37.5, 4096.0] {
            assert_eq!(
                prediction.decode_cost_blocks(decode_cost_blocks, 16),
                decode_cost_blocks
            );
        }
    }

    #[test]
    fn ground_truth_scales_with_decode_to_prefill_ratio() {
        let mut kv_router_config = config(true);
        kv_router_config.effective_work_predictor = EffectiveWorkPredictor::GroundTruthFutureDecode;
        kv_router_config.jsew_decode_work_scale = 1.0;
        kv_router_config.jsew_decode_footprint_weight = 0.0;
        let state = JsewEmaState::default();

        // 16 prompt blocks, 16 predicted decode blocks -> ratio 1.0.
        let balanced = resolve_prediction(
            &kv_router_config,
            &request(16 * 64, Some(16 * 64)),
            &state,
            0,
            64,
        )
        .expect("gate is on");
        assert_eq!(balanced.decode_cost_blocks(100.0, 16), 100.0);

        // Same prompt, a hundredth of the output -> the decode term nearly
        // vanishes, so cache credit dominates the logit for this request.
        let prefill_heavy = resolve_prediction(
            &kv_router_config,
            &request(16 * 64, Some(16 * 64 / 100)),
            &state,
            0,
            64,
        )
        .expect("gate is on");
        assert!(prefill_heavy.decode_cost_blocks(100.0, 16) < 1.1);
    }

    #[test]
    fn ema_smooths_uncached_tokens_across_requests() {
        let mut kv_router_config = config(true);
        kv_router_config.effective_work_predictor = EffectiveWorkPredictor::EmaUncachedTokens;
        kv_router_config.jsew_ema_alpha = 0.5;
        kv_router_config.jsew_ema_seed_tokens = 0.0;
        let state = JsewEmaState::default();

        // First observation: 0.5 * 1000 + 0.5 * seed(0) = 500 tokens.
        let first = resolve_prediction(&kv_router_config, &request(1000, None), &state, 0, 1)
            .expect("gate is on");
        assert_eq!(first.predicted_decode_blocks, 500.0);

        // Second identical observation pulls the average toward 1000.
        let second = resolve_prediction(&kv_router_config, &request(1000, None), &state, 0, 1)
            .expect("gate is on");
        assert_eq!(second.predicted_decode_blocks, 750.0);

        // Cache overlap removes tokens from the observation.
        let cached = resolve_prediction(&kv_router_config, &request(1000, None), &state, 1000, 1)
            .expect("gate is on");
        assert_eq!(cached.predicted_decode_blocks, 375.0);
    }

    #[test]
    fn shipped_defaults_are_the_tuned_operating_point() {
        // The claim surface only exposes `jsew_enabled` and
        // `effective_work_predictor`, so the remaining knobs must ship at the
        // tuned values or enabling the gate reproduces a different policy.
        let defaults = KvRouterConfig::default();
        assert!(!defaults.jsew_enabled, "the gate must ship off");
        assert_eq!(
            defaults.effective_work_predictor,
            EffectiveWorkPredictor::EmaUncachedTokens
        );
        assert_eq!(defaults.jsew_decode_work_scale, 8.55);
        assert_eq!(defaults.jsew_ema_alpha, 0.02);
        assert_eq!(defaults.jsew_decode_footprint_weight, 1.0);
        assert_eq!(defaults.jsew_ema_seed_tokens, 0.0);
    }

    #[test]
    fn ground_truth_falls_back_to_ema_when_output_is_undeclared() {
        let mut kv_router_config = config(true);
        kv_router_config.effective_work_predictor = EffectiveWorkPredictor::GroundTruthFutureDecode;
        kv_router_config.jsew_ema_alpha = 1.0;
        let state = JsewEmaState::default();
        let prediction = resolve_prediction(&kv_router_config, &request(640, None), &state, 0, 64)
            .expect("gate is on");
        assert_eq!(prediction.predicted_decode_blocks, 10.0);
    }
}
