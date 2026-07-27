use std::collections::BTreeMap;

const DEFAULT_MAX_BUCKETS: usize = 10_000;
const DEFAULT_MAX_AGE_SECONDS: i64 = 180 * 24 * 60 * 60;
const EWMA_ALPHA: f64 = 0.25;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EstimateConfidence {
    Low,
    Medium,
    High,
}

#[derive(Clone, Debug, Default, Eq, Ord, PartialEq, PartialOrd)]
pub struct EstimateKey {
    pub device_class: String,
    /// Resolved semantic model family (for example `flux` or `ltx2`).
    /// Empty only for records written before schema v14 or unresolved
    /// third-party models, where the full opaque model ID remains the
    /// collision-free fallback.
    pub model_family: String,
    pub model_fingerprint: String,
    pub work_kind: String,
    pub shape_bucket: String,
    pub execution_fingerprint: String,
}

impl EstimateKey {
    /// Collision-free persistence key. Length-prefixing avoids delimiter
    /// ambiguity for opaque catalog IDs and execution fingerprints.
    pub fn persistence_key(&self) -> String {
        [
            &self.device_class,
            &self.model_family,
            &self.model_fingerprint,
            &self.work_kind,
            &self.shape_bucket,
            &self.execution_fingerprint,
        ]
        .into_iter()
        .map(|part| format!("{}:{part}", part.len()))
        .collect::<Vec<_>>()
        .join("|")
    }

    /// A broader fallback bucket that preserves model family, device class,
    /// work kind, and shape while dropping precision/placement specifics.
    pub fn normalized(&self) -> Self {
        let model_family = if self.model_family.is_empty() {
            self.model_fingerprint.clone()
        } else {
            self.model_family.clone()
        };
        Self {
            device_class: self.device_class.clone(),
            model_family: self.model_family.clone(),
            model_fingerprint: model_family,
            work_kind: self.work_kind.clone(),
            shape_bucket: self.shape_bucket.clone(),
            execution_fingerprint: "*".to_string(),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StaticEstimate {
    pub total_ms: u64,
    pub vram_bytes: u64,
    pub host_bytes: u64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum EstimateOutcome {
    #[default]
    Success,
    Failure,
    Invalidated,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EstimatePhaseTimings {
    pub cold_load_ms: Option<u64>,
    pub warm_reload_ms: Option<u64>,
    pub prompt_encode_ms: Option<u64>,
    pub denoise_ms: Option<u64>,
    pub vae_ms: Option<u64>,
    pub upscale_ms: Option<u64>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct EstimateObservation {
    /// Present for successful observations. Failures and invalidations are
    /// persisted for diagnosis but never poison the ETA EWMA.
    pub total_ms: Option<u64>,
    pub phases: EstimatePhaseTimings,
    /// Peak samples measured repeatedly while the lease was active.
    pub vram_high_water_bytes: Option<u64>,
    pub host_incremental_high_water_bytes: Option<u64>,
    pub outcome: EstimateOutcome,
    pub fallback_reason: Option<String>,
    pub invalidated_plan_reason: Option<String>,
    pub observed_at_unix_s: i64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EstimateBucket {
    pub key: EstimateKey,
    pub sample_count: u64,
    pub ewma_total_ms: f64,
    pub ewma_load_ms: Option<f64>,
    pub ewma_warm_reload_ms: Option<f64>,
    pub ewma_prompt_encode_ms: Option<f64>,
    pub ewma_denoise_ms: Option<f64>,
    pub ewma_vae_ms: Option<f64>,
    pub ewma_upscale_ms: Option<f64>,
    /// Decaying conservative envelope of measured execution peaks.
    pub vram_conservative_bytes: Option<u64>,
    pub host_conservative_bytes: Option<u64>,
    pub failure_count: u64,
    pub invalidated_count: u64,
    pub last_outcome: EstimateOutcome,
    pub last_fallback_reason: Option<String>,
    pub last_invalidated_plan_reason: Option<String>,
    pub last_observed_at_unix_s: i64,
}

impl EstimateBucket {
    pub fn confidence(&self) -> EstimateConfidence {
        match self.sample_count {
            0..=2 => EstimateConfidence::Low,
            3..=9 => EstimateConfidence::Medium,
            _ => EstimateConfidence::High,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResolvedEstimate {
    pub total_ms: u64,
    pub vram_bytes: u64,
    pub host_bytes: u64,
    pub confidence: EstimateConfidence,
    pub learned: bool,
}

#[derive(Clone, Debug)]
pub struct EstimateStore {
    buckets: BTreeMap<EstimateKey, EstimateBucket>,
    max_buckets: usize,
    max_age_seconds: i64,
}

impl Default for EstimateStore {
    fn default() -> Self {
        Self::with_limits(DEFAULT_MAX_BUCKETS, DEFAULT_MAX_AGE_SECONDS)
    }
}

impl EstimateStore {
    pub fn with_limits(max_buckets: usize, max_age_seconds: i64) -> Self {
        Self {
            buckets: BTreeMap::new(),
            max_buckets,
            max_age_seconds,
        }
    }

    pub fn from_buckets(buckets: impl IntoIterator<Item = EstimateBucket>) -> Self {
        let mut store = Self::default();
        for bucket in buckets {
            store.buckets.insert(bucket.key.clone(), bucket);
        }
        store.enforce_capacity();
        store
    }

    pub fn buckets(&self) -> impl Iterator<Item = &EstimateBucket> {
        self.buckets.values()
    }

    pub fn exact(&self, key: &EstimateKey) -> Option<&EstimateBucket> {
        self.buckets.get(key)
    }

    pub fn len(&self) -> usize {
        self.buckets.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buckets.is_empty()
    }

    pub fn observe(&mut self, key: EstimateKey, observation: EstimateObservation) {
        match self.buckets.get_mut(&key) {
            Some(bucket) => {
                if observation.outcome == EstimateOutcome::Success {
                    if let Some(total_ms) = observation.total_ms {
                        bucket.ewma_total_ms = update_ewma(bucket.ewma_total_ms, total_ms as f64);
                        bucket.sample_count = bucket.sample_count.saturating_add(1);
                    }
                    bucket.ewma_load_ms = update_optional_ewma(
                        bucket.ewma_load_ms,
                        observation.phases.cold_load_ms.map(|value| value as f64),
                    );
                    bucket.ewma_warm_reload_ms = update_optional_ewma(
                        bucket.ewma_warm_reload_ms,
                        observation.phases.warm_reload_ms.map(|value| value as f64),
                    );
                    bucket.ewma_prompt_encode_ms = update_optional_ewma(
                        bucket.ewma_prompt_encode_ms,
                        observation
                            .phases
                            .prompt_encode_ms
                            .map(|value| value as f64),
                    );
                    bucket.ewma_denoise_ms = update_optional_ewma(
                        bucket.ewma_denoise_ms,
                        observation.phases.denoise_ms.map(|value| value as f64),
                    );
                    bucket.ewma_vae_ms = update_optional_ewma(
                        bucket.ewma_vae_ms,
                        observation.phases.vae_ms.map(|value| value as f64),
                    );
                    bucket.ewma_upscale_ms = update_optional_ewma(
                        bucket.ewma_upscale_ms,
                        observation.phases.upscale_ms.map(|value| value as f64),
                    );
                } else if observation.outcome == EstimateOutcome::Failure {
                    bucket.failure_count = bucket.failure_count.saturating_add(1);
                } else {
                    bucket.invalidated_count = bucket.invalidated_count.saturating_add(1);
                }
                bucket.vram_conservative_bytes = update_conservative_envelope(
                    bucket.vram_conservative_bytes,
                    observation.vram_high_water_bytes,
                );
                bucket.host_conservative_bytes = update_conservative_envelope(
                    bucket.host_conservative_bytes,
                    observation.host_incremental_high_water_bytes,
                );
                bucket.last_outcome = observation.outcome;
                bucket.last_fallback_reason = observation.fallback_reason;
                bucket.last_invalidated_plan_reason = observation.invalidated_plan_reason;
                bucket.last_observed_at_unix_s = observation.observed_at_unix_s;
            }
            None => {
                let successful = observation.outcome == EstimateOutcome::Success;
                self.buckets.insert(
                    key.clone(),
                    EstimateBucket {
                        key,
                        sample_count: u64::from(successful && observation.total_ms.is_some()),
                        ewma_total_ms: observation.total_ms.unwrap_or_default() as f64,
                        ewma_load_ms: observation
                            .phases
                            .cold_load_ms
                            .filter(|_| successful)
                            .map(|value| value as f64),
                        ewma_warm_reload_ms: observation
                            .phases
                            .warm_reload_ms
                            .filter(|_| successful)
                            .map(|value| value as f64),
                        ewma_prompt_encode_ms: observation
                            .phases
                            .prompt_encode_ms
                            .filter(|_| successful)
                            .map(|value| value as f64),
                        ewma_denoise_ms: observation
                            .phases
                            .denoise_ms
                            .filter(|_| successful)
                            .map(|value| value as f64),
                        ewma_vae_ms: observation
                            .phases
                            .vae_ms
                            .filter(|_| successful)
                            .map(|value| value as f64),
                        ewma_upscale_ms: observation
                            .phases
                            .upscale_ms
                            .filter(|_| successful)
                            .map(|value| value as f64),
                        vram_conservative_bytes: observation.vram_high_water_bytes,
                        host_conservative_bytes: observation.host_incremental_high_water_bytes,
                        failure_count: u64::from(observation.outcome == EstimateOutcome::Failure),
                        invalidated_count: u64::from(
                            observation.outcome == EstimateOutcome::Invalidated,
                        ),
                        last_outcome: observation.outcome,
                        last_fallback_reason: observation.fallback_reason,
                        last_invalidated_plan_reason: observation.invalidated_plan_reason,
                        last_observed_at_unix_s: observation.observed_at_unix_s,
                    },
                );
            }
        }
        self.enforce_capacity();
    }

    pub fn estimate(&self, key: &EstimateKey, static_estimate: StaticEstimate) -> ResolvedEstimate {
        let bucket = self
            .exact(key)
            .filter(|bucket| bucket.sample_count > 0)
            .or_else(|| {
                self.exact(&key.normalized())
                    .filter(|bucket| bucket.sample_count > 0)
            });
        let Some(bucket) = bucket else {
            return ResolvedEstimate {
                total_ms: static_estimate.total_ms,
                vram_bytes: static_estimate.vram_bytes,
                host_bytes: static_estimate.host_bytes,
                confidence: EstimateConfidence::Low,
                learned: false,
            };
        };
        ResolvedEstimate {
            total_ms: bucket.ewma_total_ms.max(0.0).round() as u64,
            // The decaying completion-sample envelope is advisory evidence,
            // never authority to weaken static admission safety.
            vram_bytes: static_estimate
                .vram_bytes
                .max(bucket.vram_conservative_bytes.unwrap_or_default()),
            host_bytes: static_estimate
                .host_bytes
                .max(bucket.host_conservative_bytes.unwrap_or_default()),
            confidence: bucket.confidence(),
            learned: true,
        }
    }

    pub fn prune(&mut self, now_unix_s: i64) {
        let cutoff = now_unix_s.saturating_sub(self.max_age_seconds);
        self.buckets
            .retain(|_, bucket| bucket.last_observed_at_unix_s >= cutoff);
        self.enforce_capacity();
    }

    fn enforce_capacity(&mut self) {
        while self.buckets.len() > self.max_buckets {
            let oldest = self
                .buckets
                .iter()
                .min_by_key(|(key, bucket)| (bucket.last_observed_at_unix_s, *key))
                .map(|(key, _)| key.clone());
            if let Some(key) = oldest {
                self.buckets.remove(&key);
            } else {
                break;
            }
        }
    }
}

fn update_ewma(prior: f64, sample: f64) -> f64 {
    // A zero-duration first observation (coarse clock/test double) must not
    // collapse the winsorization window to [0, 0] forever.
    let winsor_center = prior.max(1.0);
    let bounded = sample.clamp(winsor_center * 0.25, winsor_center * 4.0);
    prior.mul_add(1.0 - EWMA_ALPHA, bounded * EWMA_ALPHA)
}

fn update_optional_ewma(prior: Option<f64>, sample: Option<f64>) -> Option<f64> {
    match (prior, sample) {
        (Some(prior), Some(sample)) => Some(update_ewma(prior, sample)),
        (None, Some(sample)) => Some(sample),
        (prior, None) => prior,
    }
}

fn update_conservative_envelope(prior: Option<u64>, sample: Option<u64>) -> Option<u64> {
    match (prior, sample) {
        (Some(prior), Some(sample)) => Some((prior.saturating_mul(95) / 100).max(sample)),
        (prior, None) => prior,
        (None, sample) => sample,
    }
}
