// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;

use crate::common::protocols::MockEngineArgs;
use dynamo_kv_router::config::KvRouterConfig;
use dynamo_kv_router::protocols::{
    ActiveSequenceEvent, WorkerConfigLike, WorkerId, WorkerWithDpRank,
};
use dynamo_kv_router::scheduling::queue::DEFAULT_MAX_BATCHED_TOKENS;
use dynamo_kv_router::sequences::SchedulerLoadSnapshot;
use dynamo_kv_router::protocols::WorkerSelectionResult;
use dynamo_kv_router::{
    ActiveSequencesMultiWorker, DefaultWorkerSelector, KvSchedulerError, LocalScheduler,
    SequencePublisher, WorkerInputs, WorkerSelectionInput, WorkerSelectionPolicy, WorkerSelector,
};

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct ReplayNoopPublisher;

impl SequencePublisher for ReplayNoopPublisher {
    fn enqueue_event(&self, _event: ActiveSequenceEvent) -> anyhow::Result<()> {
        Ok(())
    }

    fn publish_scheduler_load(&self, _load: SchedulerLoadSnapshot) {}

    fn observe_load(&self, _: &WorkerWithDpRank, _: &str, _: usize, _: usize) {}
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct ReplayWorkerConfig {
    pub(super) max_num_batched_tokens: u64,
    pub(super) total_kv_blocks: u64,
    pub(super) data_parallel_start_rank: u32,
    pub(super) data_parallel_size: u32,
}

impl WorkerConfigLike for ReplayWorkerConfig {
    fn data_parallel_start_rank(&self) -> u32 {
        self.data_parallel_start_rank
    }

    fn data_parallel_size(&self) -> u32 {
        self.data_parallel_size
    }

    fn max_num_batched_tokens(&self) -> Option<u64> {
        Some(self.max_num_batched_tokens)
    }

    fn total_kv_blocks(&self) -> Option<u64> {
        Some(self.total_kv_blocks)
    }
}

pub(super) type ReplayScheduler =
    LocalScheduler<ReplayNoopPublisher, ReplayWorkerConfig, ReplayWorkerSelector>;

/// The selector replay schedules with: Dynamo's stock selector, or a custom
/// worker-selection policy resolved from the shipped registry when
/// `router_policy_config` selects one (feature `custom-policy`).
pub(crate) enum ReplayWorkerSelector {
    Default(DefaultWorkerSelector),
    Policy(WorkerSelectionPolicy),
}

impl<C: WorkerConfigLike> WorkerSelector<C> for ReplayWorkerSelector {
    fn required_worker_inputs(&self) -> WorkerInputs {
        match self {
            Self::Default(selector) => WorkerSelector::<C>::required_worker_inputs(selector),
            Self::Policy(policy) => WorkerSelector::<C>::required_worker_inputs(policy),
        }
    }

    fn uses_exclusive_affinity_target(&self) -> bool {
        match self {
            Self::Default(selector) => WorkerSelector::<C>::uses_exclusive_affinity_target(selector),
            Self::Policy(policy) => WorkerSelector::<C>::uses_exclusive_affinity_target(policy),
        }
    }

    fn select_worker(
        &self,
        input: WorkerSelectionInput<'_, C>,
    ) -> Result<WorkerSelectionResult, KvSchedulerError> {
        match self {
            Self::Default(selector) => selector.select_worker(input),
            Self::Policy(policy) => policy.select_worker(input),
        }
    }
}

pub(in crate::replay) fn replay_worker_config(args: &MockEngineArgs) -> ReplayWorkerConfig {
    ReplayWorkerConfig {
        max_num_batched_tokens: args
            .max_num_batched_tokens
            .map(|tokens| tokens as u64)
            .unwrap_or(DEFAULT_MAX_BATCHED_TOKENS),
        total_kv_blocks: args.num_gpu_blocks as u64,
        data_parallel_start_rank: 0,
        data_parallel_size: args.dp_size.max(1),
    }
}

pub(super) fn replay_workers_with_configs(
    args: &MockEngineArgs,
    num_workers: usize,
) -> HashMap<WorkerId, ReplayWorkerConfig> {
    let worker_config = replay_worker_config(args);
    (0..num_workers)
        .map(|worker_idx| (worker_idx as WorkerId, worker_config.clone()))
        .collect()
}

pub(super) fn replay_slots(
    args: &MockEngineArgs,
    workers_with_configs: &HashMap<WorkerId, ReplayWorkerConfig>,
) -> Arc<ActiveSequencesMultiWorker<ReplayNoopPublisher>> {
    let dp_range = workers_with_configs
        .iter()
        .map(|(&worker_id, config)| {
            (
                worker_id,
                (config.data_parallel_start_rank, config.data_parallel_size),
            )
        })
        .collect();
    // NOTE: Offline replay must retire requests through explicit lifecycle events. Wall-clock
    // expiry is a live-router cleanup heuristic and must not observe simulator CPU time: a
    // healthy replay may spend minutes of wall time advancing seconds of virtual time. Keep
    // expiry disabled here until replay has a liveness-aware definition of a stale request; do
    // not mask replay dead ends by expiring requests that are still live in virtual time.
    Arc::new(ActiveSequencesMultiWorker::new_without_expiry(
        ReplayNoopPublisher,
        args.block_size,
        dp_range,
        false,
        0,
        "replay",
    ))
}

pub(super) fn replay_selector(config: &KvRouterConfig) -> anyhow::Result<ReplayWorkerSelector> {
    replay_selector_with_seed(config, None)
}

pub(super) fn replay_selector_with_seed(
    config: &KvRouterConfig,
    selector_seed: Option<u64>,
) -> anyhow::Result<ReplayWorkerSelector> {
    if let Some(instance) = config
        .selected_worker_selection_policy_instance()
        .map_err(anyhow::Error::from)?
    {
        return custom_replay_selector(config, &instance);
    }

    Ok(ReplayWorkerSelector::Default(match selector_seed {
        #[cfg(feature = "replay-bench")]
        Some(seed) => DefaultWorkerSelector::new_seeded(Some(config.clone()), "replay", seed),
        #[cfg(not(feature = "replay-bench"))]
        Some(_) => unreachable!("canonical KV Router replay requires the replay-bench feature"),
        None => DefaultWorkerSelector::new(Some(config.clone()), "replay"),
    }))
}

/// Build the selected custom policy from the registry of policies Dynamo ships
/// (the same `lib/router-plugins/builtin` catalog the frontend links). Replay
/// workers are one aggregated pool in one routing partition.
#[cfg(feature = "custom-policy")]
fn custom_replay_selector(
    config: &KvRouterConfig,
    instance: &str,
) -> anyhow::Result<ReplayWorkerSelector> {
    use dynamo_kv_router::services::selection::WorkerSelectionPolicyRegistry;
    use dynamo_kv_router::{DEFAULT_ROUTING_GROUP, RoutingPartitionRef, WorkerType};

    let mut registry = WorkerSelectionPolicyRegistry::default();
    dynamo_custom_policy_builtin::register(&mut registry)?;
    let factory = registry
        .resolve_for_worker_type(config, WorkerType::Aggregated)?
        .ok_or_else(|| {
            anyhow::anyhow!(
                "custom worker-selection policy {instance:?} is selected but resolves to no factory"
            )
        })?;
    let partition = RoutingPartitionRef::new("replay", DEFAULT_ROUTING_GROUP);
    Ok(ReplayWorkerSelector::Policy(factory(
        config,
        WorkerType::Aggregated,
        partition,
    )))
}

#[cfg(not(feature = "custom-policy"))]
fn custom_replay_selector(
    _config: &KvRouterConfig,
    instance: &str,
) -> anyhow::Result<ReplayWorkerSelector> {
    anyhow::bail!(
        "custom worker-selection policy {instance:?} is not supported by this replay build \
         (enable the dynamo-mocker `custom-policy` feature)"
    );
}

pub(crate) fn replay_router_config(
    args: &MockEngineArgs,
    router_config: Option<KvRouterConfig>,
) -> KvRouterConfig {
    let mut config = router_config.unwrap_or_default();
    if let Some(policy) = args.router_queue_policy {
        config.router_queue_policy = policy;
    }
    config
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replay_selector_rejects_custom_worker_selection() {
        let policy_file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(
            policy_file.path(),
            r#"
worker_selection:
  aggregated: custom
  instances:
    - name: custom
      type: test
      parameters: {}
"#,
        )
        .unwrap();
        let config = KvRouterConfig {
            router_policy_config: Some(policy_file.path().display().to_string()),
            ..Default::default()
        };

        let error = match replay_selector(&config) {
            Err(error) => error,
            Ok(_) => panic!("replay must reject a custom worker selector it cannot execute"),
        };
        let message = error.to_string();
        if cfg!(feature = "custom-policy") {
            // The registry is consulted: an unregistered type is the error.
            assert!(message.contains("test"), "{message}");
        } else {
            assert!(
                message.contains("custom worker-selection policy \"custom\" is not supported"),
                "{message}"
            );
        }
    }

    #[cfg(feature = "custom-policy")]
    #[test]
    fn replay_selector_builds_a_shipped_policy() {
        let policy_file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(
            policy_file.path(),
            r#"
worker_selection:
  aggregated: two-tier
  instances:
    - name: two-tier
      type: dynamo-two-tier-cost-fn
      parameters:
        cache_threshold: 0.5
"#,
        )
        .unwrap();
        let config = KvRouterConfig {
            router_policy_config: Some(policy_file.path().display().to_string()),
            ..Default::default()
        };
        assert!(matches!(
            replay_selector(&config).unwrap(),
            ReplayWorkerSelector::Policy(_)
        ));
        assert!(matches!(
            replay_selector(&KvRouterConfig::default()).unwrap(),
            ReplayWorkerSelector::Default(_)
        ));
    }
}
