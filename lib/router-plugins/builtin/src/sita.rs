// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! KV-aware size-interval task assignment (SITA) worker selection.
//!
//! Classic SITA partitions servers into size bands and routes each job to the band owning its
//! size interval. Under heavy-tailed size distributions this keeps short interactive requests
//! from queueing behind long prefills (head-of-line blocking), which typically beats
//! least-work-left on mean latency. The KV-aware twist: "size" is the *effective* cost, the
//! prefill this request must actually compute (prompt tokens minus the best cached prefix any
//! eligible worker offers) plus `osl_weight` times its expected output tokens.
//!
//! The policy keeps Dynamo's default cost function unchanged and only narrows the candidate set
//! before the pick:
//!
//! 1. Eligible worker ids are sorted and cut into contiguous bands. Band 0 (short requests)
//!    receives `ceil(small_band_share * N)` workers; the remaining workers are split between the
//!    longer bands (`boundary_2 == 0` selects a two-band split).
//! 2. The request's size selects its band. Within the band, the default scorer's costs decide
//!    exactly as they do without SITA (minimum cost with uniform tie-breaking, or temperature
//!    sampling).
//! 3. Relief valves keep bands from becoming saturation sinks: a band holding more than
//!    `spill_threshold` of the pool's per-worker load may widen into a quieter neighbor; a
//!    saturated long band may borrow band 0 while band 0 is measurably idle; and a long request
//!    with no cached prefix anywhere gets the whole non-short pool, because confinement only
//!    pays for itself through cache affinity.
//!
//! `enabled: false` (or a pool of fewer than two workers, or a pinned request) makes the policy
//! byte-identical to `dynamo-default-cost-fn`, which is how an A/B mechanism probe measures it.
//!
//! Select it through `router_policy_config`:
//!
//! ```yaml
//! worker_selection:
//!   aggregated: sita
//!   instances:
//!     - name: sita
//!       type: dynamo-sita-cost-fn
//!       parameters:
//!         boundary_1: 1024
//!         boundary_2: 8192
//!         small_band_share: 0.5
//!         spill_threshold: 0.85
//!         osl_weight: 0.0
//!         overlap_score_credit: 1.0   # default-scorer overrides are accepted too
//! ```

use std::sync::Arc;

use dynamo_kv_router::plugins::worker_selection::{
    WorkerInputView, WorkerInputs, WorkerPicker, WorkerSelectionContext, WorkerSelectionPolicy,
    WorkerSelectionPolicyError, WorkerSelectionPolicyFactory,
};
use dynamo_kv_router::plugins::{
    RouterPluginRegistry, WorkerSelectionPolicyParameters, WorkerSelectionPolicyProviderError,
    WorkerSelectionPolicyRegistryError,
};
use dynamo_kv_router::{KvRouterConfig, WorkerType};

use crate::default;

/// Policy type selected by `worker_selection.instances[].type`.
pub const POLICY_TYPE: &str = "dynamo-sita-cost-fn";

const DEFAULT_BOUNDARY_1: usize = 1024;
const DEFAULT_BOUNDARY_2: usize = 8192;
const DEFAULT_OSL_WEIGHT: f64 = 0.0;
const DEFAULT_SMALL_BAND_SHARE: f64 = 0.5;
const DEFAULT_SPILL_THRESHOLD: f64 = 0.85;

/// Tunables for [`POLICY_TYPE`]. Every field is optional; unknown keys fail startup.
#[derive(Debug, Clone, Copy, serde::Deserialize)]
#[serde(deny_unknown_fields, default)]
pub(crate) struct SitaParameters {
    /// Master switch. When false, selection is exactly the default cost function.
    pub(crate) enabled: bool,
    /// Upper size bound, in tokens, of band 0 (short requests). Must be greater than 0.
    pub(crate) boundary_1: usize,
    /// Upper size bound, in tokens, of band 1. `0` disables the third band (two-band split);
    /// otherwise must exceed `boundary_1`.
    pub(crate) boundary_2: usize,
    /// Weight of expected output tokens in the size estimate. Requests without an output
    /// estimate contribute prefill only.
    pub(crate) osl_weight: f64,
    /// Fraction of the pool reserved for band 0; band 0 gets `ceil(share * N)` workers.
    /// Strictly between 0 and 1.
    pub(crate) small_band_share: f64,
    /// Relative band occupancy above which a request may spill into an adjacent band, in
    /// `[0.5, 1.0]`. `1.0` disables every relief valve (plain confinement).
    pub(crate) spill_threshold: f64,
}

impl Default for SitaParameters {
    fn default() -> Self {
        Self {
            enabled: true,
            boundary_1: DEFAULT_BOUNDARY_1,
            boundary_2: DEFAULT_BOUNDARY_2,
            osl_weight: DEFAULT_OSL_WEIGHT,
            small_band_share: DEFAULT_SMALL_BAND_SHARE,
            spill_threshold: DEFAULT_SPILL_THRESHOLD,
        }
    }
}

impl SitaParameters {
    pub(crate) fn validate(&self) -> Result<(), WorkerSelectionPolicyProviderError> {
        if self.boundary_1 == 0 {
            return Err(WorkerSelectionPolicyProviderError::new(
                "boundary_1 must be greater than 0",
            ));
        }
        if self.boundary_2 != 0 && self.boundary_2 <= self.boundary_1 {
            return Err(WorkerSelectionPolicyProviderError::new(
                "boundary_2 must be 0 (two-band split) or greater than boundary_1",
            ));
        }
        if !(self.small_band_share > 0.0 && self.small_band_share < 1.0) {
            return Err(WorkerSelectionPolicyProviderError::new(
                "small_band_share must be strictly between 0 and 1",
            ));
        }
        if !self.spill_threshold.is_finite() || !(0.5..=1.0).contains(&self.spill_threshold) {
            return Err(WorkerSelectionPolicyProviderError::new(
                "spill_threshold must be a finite number between 0.5 and 1.0",
            ));
        }
        if !self.osl_weight.is_finite() || self.osl_weight < 0.0 {
            return Err(WorkerSelectionPolicyProviderError::new(
                "osl_weight must be finite and non-negative",
            ));
        }
        Ok(())
    }

    fn band_count(&self) -> usize {
        if self.boundary_2 == 0 { 2 } else { 3 }
    }
}

/// Startup parameters: SITA's own knobs plus optional default-scorer overrides, flat in one map.
#[derive(Default, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
struct Parameters {
    enabled: Option<bool>,
    boundary_1: Option<usize>,
    boundary_2: Option<usize>,
    osl_weight: Option<f64>,
    small_band_share: Option<f64>,
    spill_threshold: Option<f64>,
    overlap_score_credit: Option<f64>,
    overlap_score_credit_decay: Option<f64>,
    prefill_load_scale: Option<f64>,
    decode_active_request_weight: Option<f64>,
    host_cache_hit_weight: Option<f64>,
    disk_cache_hit_weight: Option<f64>,
    shared_cache_multiplier: Option<f64>,
    router_temperature: Option<f64>,
}

impl Parameters {
    fn split(&self) -> (SitaParameters, default::parameters::Parameters) {
        let defaults = SitaParameters::default();
        let sita = SitaParameters {
            enabled: self.enabled.unwrap_or(defaults.enabled),
            boundary_1: self.boundary_1.unwrap_or(defaults.boundary_1),
            boundary_2: self.boundary_2.unwrap_or(defaults.boundary_2),
            osl_weight: self.osl_weight.unwrap_or(defaults.osl_weight),
            small_band_share: self.small_band_share.unwrap_or(defaults.small_band_share),
            spill_threshold: self.spill_threshold.unwrap_or(defaults.spill_threshold),
        };
        let scoring = default::parameters::Parameters {
            overlap_score_credit: self.overlap_score_credit,
            overlap_score_credit_decay: self.overlap_score_credit_decay,
            prefill_load_scale: self.prefill_load_scale,
            decode_active_request_weight: self.decode_active_request_weight,
            host_cache_hit_weight: self.host_cache_hit_weight,
            disk_cache_hit_weight: self.disk_cache_hit_weight,
            shared_cache_multiplier: self.shared_cache_multiplier,
            router_temperature: self.router_temperature,
        };
        (sita, scoring)
    }
}

// ---- band geometry -------------------------------------------------------------------------

/// Half-open index range into the sorted eligible worker-id list.
type Slice = (usize, usize);

/// Map an estimated request size in tokens to a band index.
pub(crate) fn band_for_size(size: usize, boundary_1: usize, boundary_2: usize) -> usize {
    if size <= boundary_1 {
        return 0;
    }
    if boundary_2 == 0 || size <= boundary_2 {
        return 1;
    }
    2
}

/// Partition `worker_count` workers into contiguous per-band slices.
///
/// Band 0 receives `ceil(small_band_share * worker_count)` workers. The remaining workers are
/// split evenly across the longer bands. Every returned slice is non-empty, so a band always has
/// somewhere to route.
pub(crate) fn band_slices(
    worker_count: usize,
    small_band_share: f64,
    band_count: usize,
) -> [Slice; 3] {
    debug_assert!(worker_count >= 2);
    let band_0 =
        ((small_band_share * worker_count as f64).ceil() as usize).clamp(1, worker_count - 1);
    let remaining = worker_count - band_0;
    if band_count < 3 || remaining == 1 {
        let tail = (band_0, worker_count);
        return [(0, band_0), tail, tail];
    }
    // Keep at least one worker in the largest band so huge requests never share the whole tail
    // with medium ones.
    let band_1 = remaining.div_ceil(2).clamp(1, remaining - 1);
    [
        (0, band_0),
        (band_0, band_0 + band_1),
        (band_0 + band_1, worker_count),
    ]
}

fn slice_mean(loads: &[f64], slice: Slice) -> f64 {
    let (start, end) = slice;
    if end <= start {
        return 0.0;
    }
    loads[start..end].iter().sum::<f64>() / (end - start) as f64
}

/// Share of the pool's per-worker load that sits in `slice`, in `[0, 1]`.
///
/// An evenly loaded pool scores 0.5, and the score rises toward 1.0 as the band's workers get
/// busier than everyone else's. Expressing occupancy relatively is what makes `spill_threshold`'s
/// `[0.5, 1.0]` range meaningful: an absolute KV-capacity fraction is a few percent under any
/// realistic serving load, so no threshold in that range would ever trip.
fn band_occupancy(loads: &[f64], slice: Slice) -> f64 {
    let (start, end) = slice;
    let inside = slice_mean(loads, slice);
    let outside_count = loads.len() - (end - start);
    if outside_count == 0 {
        return 0.5;
    }
    let outside_total = loads.iter().sum::<f64>() - loads[start..end].iter().sum::<f64>();
    let outside = outside_total / outside_count as f64;
    let denominator = inside + outside;
    if denominator <= 0.0 {
        return 0.5;
    }
    inside / denominator
}

/// Widen `band` into an adjacent band when the target holds more than `spill_threshold` of the
/// pool's load and that neighbor is genuinely quieter.
///
/// Which neighbor is allowed is asymmetric. Sending a request *up* into a longer band costs
/// roughly its own service time. Sending a long request *down* parks a multi-thousand-token
/// prefill in front of everything queued behind it, the head-of-line blocking SITA exists to
/// prevent. So band 0 is never a spill target merely because it is the nearest neighbor; see
/// [`borrow_idle_short_band`] for the one exception. The top band may widen downward into band 1,
/// which holds medium requests, so it never becomes a saturation sink.
fn apply_spill(
    band: usize,
    band_count: usize,
    slices: &[Slice; 3],
    loads: &[f64],
    spill_threshold: f64,
) -> Slice {
    let target = slices[band];
    if spill_threshold >= 1.0 {
        return target;
    }
    let widened = if band_occupancy(loads, target) > spill_threshold {
        let neighbor_band = if band + 1 < band_count {
            Some(band + 1)
        } else if band >= 2 {
            Some(band - 1)
        } else {
            None
        };
        match neighbor_band.map(|next| slices[next]) {
            Some(neighbor)
                if neighbor != target
                    && slice_mean(loads, neighbor) < slice_mean(loads, target) =>
            {
                (target.0.min(neighbor.0), target.1.max(neighbor.1))
            }
            _ => target,
        }
    } else {
        target
    };
    borrow_idle_short_band(widened, slices, loads, spill_threshold)
}

/// Last-resort valve: let a long band borrow band 0's workers, but only while band 0 is
/// measurably *idle*. Reserving workers for short requests is what makes SITA work, but the
/// reservation is only free when the short band is actually using them; a statically fenced-off
/// band 0 starves the long bands of KV capacity and wrecks their end-to-end tail.
fn borrow_idle_short_band(
    target: Slice,
    slices: &[Slice; 3],
    loads: &[f64],
    spill_threshold: f64,
) -> Slice {
    let short = slices[0];
    if target.0 <= short.0 {
        return target;
    }
    if band_occupancy(loads, short) >= 1.0 - spill_threshold {
        return target;
    }
    (short.0, target.1)
}

/// Widen a long request's band to every non-short worker when it has no cached prefix to come
/// back to (band confinement pays for itself through cache affinity, which such a request lacks),
/// and additionally to band 0 while band 0 is idle by a proportionally larger allowance.
fn widen_uncached_long_band(
    band: usize,
    slices: &[Slice; 3],
    worker_count: usize,
    best_cached_tokens: usize,
    loads: &[f64],
    spill_threshold: f64,
) -> Slice {
    if band == 0 || best_cached_tokens > 0 {
        return slices[band];
    }
    let idle_allowance = (1.0 - spill_threshold) * 2.0;
    if band_occupancy(loads, slices[0]) < idle_allowance {
        return (0, worker_count);
    }
    (slices[0].1, worker_count)
}

// ---- picker --------------------------------------------------------------------------------

struct SitaPicker {
    parameters: SitaParameters,
    temperature: f64,
    worker_ids: Vec<u64>,
    loads: Vec<f64>,
    rows: Vec<usize>,
    probabilities: Vec<f64>,
}

impl SitaPicker {
    fn new(parameters: SitaParameters, temperature: f64) -> Self {
        Self {
            parameters,
            temperature,
            worker_ids: Vec::new(),
            loads: Vec::new(),
            rows: Vec::new(),
            probabilities: Vec::new(),
        }
    }

    /// Restrict `self.rows` to the request's band. Leaves every row in place whenever SITA must
    /// not change routing (disabled, pool too small to partition).
    fn restrict_rows(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<(), WorkerSelectionPolicyError> {
        let candidates = input.candidates();
        self.rows.clear();
        self.rows.extend(0..candidates.len());
        if !self.parameters.enabled {
            return Ok(());
        }

        // Distinct worker ids, sorted: bands are contiguous id ranges over the pool.
        self.worker_ids.clear();
        self.worker_ids.extend(
            candidates
                .iter()
                .map(|candidate| candidate.worker().worker_id),
        );
        self.worker_ids.sort_unstable();
        self.worker_ids.dedup();
        let worker_count = self.worker_ids.len();
        if worker_count < 2 {
            return Ok(());
        }

        let load = input
            .load()
            .ok_or_else(|| WorkerSelectionPolicyError::failed("load input unavailable"))?;
        let cache = input
            .cache()
            .ok_or_else(|| WorkerSelectionPolicyError::failed("cache input unavailable"))?;
        if load.len() != candidates.len() || cache.len() != candidates.len() {
            return Err(WorkerSelectionPolicyError::failed(
                "candidate inputs are not index-aligned",
            ));
        }

        // Per-worker queued work in tokens, averaged over the worker's data-parallel ranks.
        // Queued prefill dominates TTFT so it counts directly; resident decode blocks are
        // converted to tokens so both contribute in the same unit.
        let block_size = context.block_size() as f64;
        self.loads.clear();
        self.loads.resize(worker_count, 0.0);
        let mut ranks = vec![0usize; worker_count];
        for (row, candidate) in candidates.iter().enumerate() {
            let index = self.worker_index(candidate.worker().worker_id);
            self.loads[index] += load[row].active_prefill_tokens() as f64
                + load[row].decode_cost_blocks() * block_size;
            ranks[index] += 1;
        }
        for (total, count) in self.loads.iter_mut().zip(ranks) {
            *total /= count.max(1) as f64;
        }

        // The best cached prefix any eligible worker offers is the prefill this request can skip.
        let best_cached_tokens = cache
            .iter()
            .map(|item| item.accounting_cache_estimate().1)
            .max()
            .unwrap_or(0);
        let effective_prefill_tokens = context.prompt_tokens().saturating_sub(best_cached_tokens);
        let output_tokens = context
            .expected_output_tokens()
            .map_or(0.0, |tokens| self.parameters.osl_weight * tokens as f64);
        let size = effective_prefill_tokens.saturating_add(output_tokens as usize);

        let band_count = self.parameters.band_count();
        let band = band_for_size(size, self.parameters.boundary_1, self.parameters.boundary_2);
        let slices = band_slices(worker_count, self.parameters.small_band_share, band_count);
        let spilled = apply_spill(
            band,
            band_count,
            &slices,
            &self.loads,
            self.parameters.spill_threshold,
        );
        let widened = widen_uncached_long_band(
            band,
            &slices,
            worker_count,
            best_cached_tokens,
            &self.loads,
            self.parameters.spill_threshold,
        );
        // A request with no cache to return to gains nothing from confinement: take the wider slice.
        let (start, end) = if widened.1 - widened.0 > spilled.1 - spilled.0 {
            widened
        } else {
            spilled
        };
        let (first, last) = (self.worker_ids[start], self.worker_ids[end - 1]);
        tracing::debug!(
            size,
            band,
            "SITA band restricted routing to worker ids [{first}, {last}]"
        );
        self.rows.retain(|&row| {
            let id = candidates[row].worker().worker_id;
            id >= first && id <= last
        });
        debug_assert!(
            !self.rows.is_empty(),
            "a band always holds at least one worker"
        );
        Ok(())
    }

    fn worker_index(&self, worker_id: u64) -> usize {
        self.worker_ids
            .binary_search(&worker_id)
            .expect("every candidate worker id was collected")
    }
}

impl WorkerPicker for SitaPicker {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::CACHE | WorkerInputs::LOAD
    }

    fn pick(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        let candidates = input.candidates();
        if candidates.is_empty() {
            return Err(WorkerSelectionPolicyError::failed("no eligible worker"));
        }
        if context.pinned_worker().is_some() {
            return Ok(0);
        }
        self.restrict_rows(context, input)?;
        if self.rows.is_empty() {
            return Err(WorkerSelectionPolicyError::failed(
                "no eligible worker in band",
            ));
        }
        if self.temperature == 0.0 {
            // Minimum cost with uniform tie-breaking, as the default picker does.
            let mut best_row = self.rows[0];
            let mut best_cost = f64::INFINITY;
            let mut ties = 0;
            for &row in &self.rows {
                let cost = candidates[row].cost();
                if cost < best_cost {
                    best_row = row;
                    best_cost = cost;
                    ties = 1;
                } else if cost == best_cost {
                    ties += 1;
                    if fastrand::usize(0..ties) == 0 {
                        best_row = row;
                    }
                }
            }
            return Ok(best_row);
        }
        let selected = default::picker::softmax_sample_index(
            &self.rows,
            |&row| candidates[row].cost(),
            self.temperature,
            fastrand::f64(),
            &mut self.probabilities,
        );
        Ok(self.rows[selected])
    }
}

// ---- provider --------------------------------------------------------------------------------

fn policy(
    config: &KvRouterConfig,
    role: WorkerType,
    sita: SitaParameters,
    scoring: default::parameters::Parameters,
) -> WorkerSelectionPolicy {
    let resolved = scoring.resolve(config);
    let label = role.default_selector_label();
    let scorer = default::scorer::build(&resolved, label, default::is_plain_decode(config, role));
    let picker = SitaPicker::new(sita, resolved.router_temperature);
    WorkerSelectionPolicy::new(config.clone(), label, vec![scorer], Box::new(picker))
        .with_exclusive_affinity(true)
}

fn provider(
    parameters: &WorkerSelectionPolicyParameters,
) -> Result<WorkerSelectionPolicyFactory, WorkerSelectionPolicyProviderError> {
    let parameters: Parameters = parameters.deserialize()?;
    let (sita, scoring) = parameters.split();
    sita.validate()?;
    scoring.validate()?;
    Ok(Arc::new(
        move |config: &KvRouterConfig, role, _partition| policy(config, role, sita, scoring),
    ))
}

pub fn register(
    registry: &mut RouterPluginRegistry,
) -> Result<(), WorkerSelectionPolicyRegistryError> {
    registry.register_worker_selection(POLICY_TYPE, Arc::new(provider))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use dynamo_kv_router::protocols::{RoutingConstraints, WorkerConfigLike, WorkerWithDpRank};
    use dynamo_kv_router::scheduling::{OverlapSignals, ScheduleMode};
    use dynamo_kv_router::{
        SchedulingRequest, WorkerLoadProjection, WorkerSelectionInput, WorkerSelector,
    };

    use super::*;

    const BLOCK_SIZE: u32 = 64;

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
            Some(1024)
        }
    }

    fn worker(id: u64) -> WorkerWithDpRank {
        WorkerWithDpRank::from_worker_id(id)
    }

    fn request(isl_tokens: usize) -> SchedulingRequest {
        SchedulingRequest {
            mode: ScheduleMode::QueryOnly { request_id: None },
            token_seq: None,
            isl_tokens,
            lora_name: None,
            expected_output_tokens: None,
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
        }
    }

    /// A request that already has a (tiny) cached prefix on worker 0, so it keeps the cache
    /// affinity that band confinement protects; an uncached request is deliberately let out of
    /// its band (see `widen_uncached_long_band`), which would mask placement under test.
    fn cached_request(isl_tokens: usize, worker_count: u64) -> SchedulingRequest {
        let mut request = request(isl_tokens);
        request
            .overlap
            .effective_cached_tokens
            .insert(worker(0), 64);
        for id in 0..worker_count {
            request
                .worker_loads
                .insert(worker(id), WorkerLoadProjection::default());
        }
        request
    }

    fn confined() -> SitaParameters {
        SitaParameters {
            // Disable every relief valve so band placement is observable on its own.
            spill_threshold: 1.0,
            ..SitaParameters::default()
        }
    }

    fn select(
        sita: SitaParameters,
        workers: &HashMap<u64, TestWorker>,
        request: &SchedulingRequest,
    ) -> WorkerWithDpRank {
        policy(
            &KvRouterConfig::default(),
            WorkerType::Aggregated,
            sita,
            default::parameters::Parameters::default(),
        )
        .select_worker(WorkerSelectionInput::configured(
            workers,
            request,
            request.eligibility(),
            BLOCK_SIZE,
        ))
        .unwrap()
        .worker
    }

    fn pool(count: u64) -> HashMap<u64, TestWorker> {
        (0..count).map(|id| (id, TestWorker)).collect()
    }

    #[test]
    fn band_mapping_uses_boundaries() {
        assert_eq!(band_for_size(0, 1024, 8192), 0);
        assert_eq!(band_for_size(1024, 1024, 8192), 0);
        assert_eq!(band_for_size(1025, 1024, 8192), 1);
        assert_eq!(band_for_size(8192, 1024, 8192), 1);
        assert_eq!(band_for_size(8193, 1024, 8192), 2);
        // boundary_2 == 0 collapses to a two-band split.
        assert_eq!(band_for_size(1025, 1024, 0), 1);
        assert_eq!(band_for_size(usize::MAX, 1024, 0), 1);
    }

    #[test]
    fn band_slices_are_contiguous_and_non_empty() {
        for (worker_count, share, band_count) in [
            (16, 0.5, 3),
            (16, 0.5, 2),
            (16, 0.125, 3),
            (16, 0.75, 3),
            (2, 0.5, 3),
            (3, 0.5, 3),
            (5, 0.125, 3),
            (7, 0.75, 3),
        ] {
            let slices = band_slices(worker_count, share, band_count);
            let used = &slices[..band_count];
            assert_eq!(used[0].0, 0);
            assert_eq!(used[band_count - 1].1, worker_count);
            for &(start, end) in used {
                assert!(start < end, "{slices:?}");
            }
            for pair in used.windows(2) {
                assert!(pair[0].1 == pair[1].0 || pair[0] == pair[1], "{slices:?}");
            }
        }
        assert_eq!(band_slices(16, 0.5, 3)[0], (0, 8));
        assert_eq!(band_slices(16, 0.125, 3)[0], (0, 2));
        assert_eq!(band_slices(10, 0.25, 3)[0], (0, 3));
    }

    #[test]
    fn short_and_long_requests_land_in_disjoint_bands() {
        // share=0.5 over 8 workers: band 0 = ids 0..4, band 1 = 4..6, band 2 = 6..8.
        let workers = pool(8);
        let short = select(confined(), &workers, &cached_request(256, 8));
        let long = select(confined(), &workers, &cached_request(16_384, 8));
        assert!(short.worker_id < 4, "short request left band 0: {short:?}");
        assert!(long.worker_id >= 6, "long request left band 2: {long:?}");
    }

    #[test]
    fn output_weight_moves_a_request_across_bands() {
        let workers = pool(8);
        let mut request = cached_request(512, 8);
        request.expected_output_tokens = Some(4096);
        // Prefill alone is band 0; with osl_weight 1.0 the size is 4608 -> band 1.
        assert!(select(confined(), &workers, &request).worker_id < 4);
        let weighted = SitaParameters {
            osl_weight: 1.0,
            ..confined()
        };
        let chosen = select(weighted, &workers, &request);
        assert!((4..6).contains(&chosen.worker_id), "{chosen:?}");
    }

    #[test]
    fn uncached_long_request_widens_beyond_its_band() {
        let slices = band_slices(8, 0.5, 3);
        assert_eq!(slices, [(0, 4), (4, 6), (6, 8)]);
        let even = [1.0; 8];
        // A cached request stays inside its own band.
        assert_eq!(
            widen_uncached_long_band(2, &slices, 8, 64, &even, 0.85),
            (6, 8)
        );
        // With no cached prefix, the long bands open up to every non-short worker...
        assert_eq!(
            widen_uncached_long_band(2, &slices, 8, 0, &even, 0.85),
            (4, 8)
        );
        assert_eq!(
            widen_uncached_long_band(1, &slices, 8, 0, &even, 0.85),
            (4, 8)
        );
        // ...and to band 0 too while band 0 is idle.
        let idle_short = [0.0, 0.0, 0.0, 0.0, 4.0, 4.0, 4.0, 4.0];
        assert_eq!(
            widen_uncached_long_band(2, &slices, 8, 0, &idle_short, 0.85),
            (0, 8)
        );
        // Band 0 itself never widens.
        assert_eq!(
            widen_uncached_long_band(0, &slices, 8, 0, &even, 0.85),
            (0, 4)
        );
    }

    #[test]
    fn spill_prefers_the_band_above_and_never_the_short_band() {
        let slices = band_slices(8, 0.5, 3);
        // Band 1 saturated (occupancy 0.9), band 2 quiet: widen upward into band 2. Band 0's
        // occupancy is 1/6, above the 0.15 idle bar at this threshold, so it is not lent.
        let loads = [1.0, 1.0, 1.0, 1.0, 9.0, 9.0, 1.0, 1.0];
        assert_eq!(apply_spill(1, 3, &slices, &loads, 0.85), (4, 8));
        // Band 2 saturated: it falls back to band 1, never to band 0.
        let loads = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 9.0, 9.0];
        assert_eq!(apply_spill(2, 3, &slices, &loads, 0.85), (4, 8));
        // Threshold 1.0 disables spilling entirely.
        assert_eq!(apply_spill(2, 3, &slices, &loads, 1.0), (6, 8));
        // Band 0 truly idle (occupancy 0): a long band borrows it even without spilling.
        let loads = [0.0, 0.0, 0.0, 0.0, 9.0, 9.0, 9.0, 9.0];
        assert_eq!(apply_spill(2, 3, &slices, &loads, 0.85), (0, 8));
        // A lower threshold loosens the idle bar too: 1/6 < 0.4 now lends band 0 out.
        let loads = [1.0, 1.0, 1.0, 1.0, 9.0, 9.0, 1.0, 1.0];
        assert_eq!(apply_spill(1, 3, &slices, &loads, 0.6), (0, 8));
    }

    #[test]
    fn disabled_is_identical_to_the_default_policy() {
        let workers = pool(8);
        let disabled = SitaParameters {
            enabled: false,
            ..confined()
        };
        for isl in [256usize, 2048, 16_384] {
            let mut request = cached_request(isl, 8);
            // Make worker 5 the unique lowest-cost worker: it holds most of the prompt.
            request
                .overlap
                .tier_overlap_blocks
                .device
                .insert(worker(5), isl / BLOCK_SIZE as usize);
            request
                .overlap
                .effective_cached_tokens
                .insert(worker(5), isl);
            let expected = default::default_policy(KvRouterConfig::default(), "test")
                .select_worker(WorkerSelectionInput::configured(
                    &workers,
                    &request,
                    request.eligibility(),
                    BLOCK_SIZE,
                ))
                .unwrap()
                .worker;
            assert_eq!(select(disabled, &workers, &request), expected);
            assert_eq!(expected, worker(5));
        }
    }

    #[test]
    fn pool_too_small_to_partition_keeps_every_worker() {
        let workers = pool(1);
        let chosen = select(confined(), &workers, &cached_request(16_384, 1));
        assert_eq!(chosen, worker(0));
    }

    /// Resolve router-policy YAML the way the Python bindings do at startup, so the documented
    /// instance shape (and every parameter name) is checked against the real configuration path.
    #[test]
    fn resolves_documented_yaml_and_rejects_unknown_parameters() {
        let resolve = |yaml: &str| {
            let policy_file = tempfile::NamedTempFile::new().unwrap();
            std::fs::write(policy_file.path(), yaml).unwrap();
            let config = KvRouterConfig {
                router_policy_config: Some(policy_file.path().display().to_string()),
                ..Default::default()
            };
            let mut registry = crate::default_registry();
            crate::register(&mut registry).unwrap();
            let resolved = registry.resolve(&config);
            (config, resolved)
        };
        let (config, resolved) = resolve(
            r#"
worker_selection:
  aggregated: sita
  prefill: sita
  decode: sita
  instances:
    - name: sita
      type: dynamo-sita-cost-fn
      parameters:
        boundary_1: 1024
        boundary_2: 0
        small_band_share: 0.375
        spill_threshold: 0.9
        osl_weight: 0.5
        overlap_score_credit: 2.0
"#,
        );
        let factory = resolved
            .unwrap()
            .expect("a configured instance resolves to a factory");
        let partition = dynamo_kv_router::RoutingPartitionRef::new("model", "default");
        for role in [
            WorkerType::Aggregated,
            WorkerType::Prefill,
            WorkerType::Decode,
        ] {
            factory(&config, role, partition);
        }

        let (_, resolved) = resolve(
            r#"
worker_selection:
  aggregated: sita
  instances:
    - name: sita
      type: dynamo-sita-cost-fn
      parameters:
        sita_boundary_1: 1024
"#,
        );
        let error = resolved
            .err()
            .expect("an unknown parameter must fail resolution");
        assert!(error.to_string().contains("sita_boundary_1"), "{error}");

        let (_, resolved) = resolve(
            r#"
worker_selection:
  aggregated: sita
  instances:
    - name: sita
      type: dynamo-sita-cost-fn
      parameters: {boundary_1: 4096, boundary_2: 1024}
"#,
        );
        assert!(
            resolved.is_err(),
            "inverted boundaries must fail resolution"
        );
    }

    #[test]
    fn rejects_out_of_range_parameters() {
        let check = |mutate: fn(&mut SitaParameters)| {
            let mut parameters = SitaParameters::default();
            mutate(&mut parameters);
            parameters.validate()
        };
        assert!(SitaParameters::default().validate().is_ok());
        assert!(check(|p| p.boundary_2 = 0).is_ok());
        assert!(check(|p| p.boundary_1 = 0).is_err());
        assert!(check(|p| p.boundary_2 = 1024).is_err());
        assert!(check(|p| p.boundary_2 = 512).is_err());
        assert!(check(|p| p.small_band_share = 0.0).is_err());
        assert!(check(|p| p.small_band_share = 1.0).is_err());
        assert!(check(|p| p.spill_threshold = 0.49).is_err());
        assert!(check(|p| p.spill_threshold = 1.01).is_err());
        assert!(check(|p| p.osl_weight = -0.1).is_err());
        assert!(check(|p| p.osl_weight = 2.0).is_ok());
    }
}
