// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Startup parameters and provider registration for the default policy.

use dynamo_kv_router::KvRouterConfig;
use dynamo_kv_router::plugins::{
    RouterPluginRegistry, WorkerSelectionPolicyProviderError, WorkerSelectionPolicyRegistryError,
};
use std::sync::Arc;

use super::policy_for_role;

/// Optional startup overrides for the default cost function. Other policies in this crate that
/// keep Dynamo's scoring and only change candidate selection reuse these fields verbatim.
#[derive(Default, Clone, Copy, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct Parameters {
    pub(crate) overlap_score_credit: Option<f64>,
    pub(crate) overlap_score_credit_decay: Option<f64>,
    pub(crate) prefill_load_scale: Option<f64>,
    pub(crate) decode_active_request_weight: Option<f64>,
    pub(crate) host_cache_hit_weight: Option<f64>,
    pub(crate) disk_cache_hit_weight: Option<f64>,
    pub(crate) shared_cache_multiplier: Option<f64>,
    pub(crate) router_temperature: Option<f64>,
}

/// Only the startup values consumed by the default scorer and picker.
#[derive(Clone, Copy)]
pub(crate) struct PolicyParameters {
    pub(crate) overlap_score_credit: f64,
    pub(crate) overlap_score_credit_decay: f64,
    pub(crate) prefill_load_scale: f64,
    pub(crate) decode_active_request_weight: f64,
    pub(crate) host_cache_hit_weight: f64,
    pub(crate) disk_cache_hit_weight: f64,
    pub(crate) shared_cache_multiplier: f64,
    pub(crate) router_temperature: f64,
}

impl From<&KvRouterConfig> for PolicyParameters {
    fn from(config: &KvRouterConfig) -> Self {
        Parameters::default().resolve(config)
    }
}

impl Parameters {
    /// Reject non-finite or negative overrides; every default-scorer weight is a magnitude.
    pub(crate) fn validate(&self) -> Result<(), WorkerSelectionPolicyProviderError> {
        for (name, value) in [
            ("overlap_score_credit", self.overlap_score_credit),
            (
                "overlap_score_credit_decay",
                self.overlap_score_credit_decay,
            ),
            ("prefill_load_scale", self.prefill_load_scale),
            (
                "decode_active_request_weight",
                self.decode_active_request_weight,
            ),
            ("host_cache_hit_weight", self.host_cache_hit_weight),
            ("disk_cache_hit_weight", self.disk_cache_hit_weight),
            ("shared_cache_multiplier", self.shared_cache_multiplier),
            ("router_temperature", self.router_temperature),
        ] {
            if value.is_some_and(|value| !value.is_finite() || value < 0.0) {
                return Err(WorkerSelectionPolicyProviderError::new(format!(
                    "{name} must be finite and non-negative"
                )));
            }
        }
        Ok(())
    }

    pub(crate) fn resolve(&self, config: &KvRouterConfig) -> PolicyParameters {
        PolicyParameters {
            overlap_score_credit: self
                .overlap_score_credit
                .unwrap_or(config.overlap_score_credit),
            overlap_score_credit_decay: self
                .overlap_score_credit_decay
                .unwrap_or(config.overlap_score_credit_decay),
            prefill_load_scale: self.prefill_load_scale.unwrap_or(config.prefill_load_scale),
            decode_active_request_weight: self
                .decode_active_request_weight
                .unwrap_or(config.decode_active_request_weight),
            host_cache_hit_weight: self
                .host_cache_hit_weight
                .unwrap_or(config.host_cache_hit_weight),
            disk_cache_hit_weight: self
                .disk_cache_hit_weight
                .unwrap_or(config.disk_cache_hit_weight),
            shared_cache_multiplier: self
                .shared_cache_multiplier
                .unwrap_or(config.shared_cache_multiplier),
            router_temperature: self.router_temperature.unwrap_or(config.router_temperature),
        }
    }
}

pub(crate) fn register(
    registry: &mut RouterPluginRegistry,
) -> Result<(), WorkerSelectionPolicyRegistryError> {
    registry.register_worker_selection(
        "dynamo-default-cost-fn",
        Arc::new(|parameters| {
            let parameters: Parameters = parameters.deserialize()?;
            parameters.validate()?;
            Ok(Arc::new(
                move |config: &KvRouterConfig, role, _partition| {
                    policy_for_role(config.clone(), role, parameters.resolve(config))
                },
            ))
        }),
    )
}
