//! MiniMax H3 inference primitives.
//!
//! H3 is not a runnable Mold family yet. This module contains deterministic
//! sampler math plus a conditioner lifecycle that accepts only an already-
//! frozen placement and issued component lease.

pub(crate) mod backend;
pub(crate) mod engine;
pub(crate) mod offload;
pub(crate) mod pipeline;
#[cfg(feature = "h3-private-uat")]
pub(crate) mod private_fl2va_runtime;
#[cfg(feature = "h3-private-uat")]
pub(crate) mod private_opened_evidence;
#[cfg(feature = "h3-private-uat")]
pub mod private_qualification;
#[cfg(feature = "h3-private-uat")]
pub(crate) mod private_qwen;
#[cfg(feature = "h3-private-uat")]
pub(crate) mod private_qwen_support;
#[cfg(feature = "h3-private-uat")]
pub(crate) mod private_runtime;
#[cfg(feature = "h3-private-uat")]
pub mod private_runtime_qualification;
#[cfg(feature = "h3-private-uat")]
pub(crate) mod private_server;
#[cfg(feature = "h3-private-uat")]
pub(crate) mod private_vae_adapter;
pub(crate) mod reference_media;
pub(crate) mod sampler;
pub(crate) mod vae_runtime;

use anyhow::{anyhow, Result};
use candle_core::{Device, Tensor};
use mold_candle::minimax_h3::{
    load_prepared_bf16_conditioner, prepare_conditioner_assets_with_progress, ArtifactError,
    ConditionerArtifacts, ConditionerCheckpoint, H3ConditionerInput, H3DTypeProfile,
    H3Layer50Conditioner, H3LoadError, PreparedH3ConditionerAssets,
};
use std::time::Instant;
use tokenizers::Tokenizer;

use crate::progress::{InferenceCancelled, ProgressReporter};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum H3ConditionerExecution {
    CudaResident,
    CpuOffloaded,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct H3MemoryCharge {
    pub resident_parameter_bytes: u64,
    pub activation_workspace_bytes: u64,
    pub charged_host_bytes: u64,
    pub charged_vram_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FrozenH3ConditionerPlacement {
    pub device_id: String,
    pub execution: H3ConditionerExecution,
    pub execution_fingerprint: String,
    pub memory: H3MemoryCharge,
}

impl FrozenH3ConditionerPlacement {
    pub fn new(
        device_id: impl Into<String>,
        execution: H3ConditionerExecution,
        execution_fingerprint: impl Into<String>,
        host_resident_parameter_bytes: u64,
        device_resident_parameter_bytes: u64,
        activation_workspace_bytes: u64,
    ) -> Result<Self> {
        let resident_parameter_bytes = host_resident_parameter_bytes
            .checked_add(device_resident_parameter_bytes)
            .ok_or_else(|| anyhow!("H3 conditioner resident parameter bytes overflow"))?;
        if resident_parameter_bytes == 0 || activation_workspace_bytes == 0 {
            return Err(anyhow!(
                "H3 conditioner placement must charge nonzero parameter and activation bytes"
            ));
        }
        let (charged_host_bytes, charged_vram_bytes) = match execution {
            H3ConditionerExecution::CudaResident if device_resident_parameter_bytes > 0 => (
                host_resident_parameter_bytes,
                device_resident_parameter_bytes
                    .checked_add(activation_workspace_bytes)
                    .ok_or_else(|| anyhow!("H3 conditioner VRAM charge overflow"))?,
            ),
            H3ConditionerExecution::CpuOffloaded
                if host_resident_parameter_bytes > 0 && device_resident_parameter_bytes == 0 =>
            {
                (
                    host_resident_parameter_bytes
                        .checked_add(activation_workspace_bytes)
                        .ok_or_else(|| anyhow!("H3 conditioner host charge overflow"))?,
                    0,
                )
            }
            _ => {
                return Err(anyhow!(
                    "H3 conditioner parameter residency differs from its execution placement"
                ))
            }
        };
        let device_id = device_id.into();
        let execution_fingerprint = execution_fingerprint.into();
        if device_id.is_empty() || execution_fingerprint.is_empty() {
            return Err(anyhow!(
                "H3 conditioner placement requires device and execution identities"
            ));
        }
        Ok(Self {
            device_id,
            execution,
            execution_fingerprint,
            memory: H3MemoryCharge {
                resident_parameter_bytes,
                activation_workspace_bytes,
                charged_host_bytes,
                charged_vram_bytes,
            },
        })
    }

    fn validate_lease(&self, lease: &impl H3ConditionerLease) -> Result<()> {
        if lease.device_id() != self.device_id {
            return Err(anyhow!(
                "H3 conditioner lease is for {}, but the frozen plan requires {}; sibling-device borrowing is forbidden",
                lease.device_id(),
                self.device_id
            ));
        }
        let backend_matches = match self.execution {
            H3ConditionerExecution::CudaResident => lease.device().is_cuda(),
            H3ConditionerExecution::CpuOffloaded => lease.device().is_cpu(),
        };
        if !backend_matches || lease.device().is_metal() {
            return Err(anyhow!(
                "H3 conditioner lease backend does not match frozen {:?} placement",
                self.execution
            ));
        }
        Ok(())
    }
}

/// Scheduler-issued, attempt-scoped authority for exactly one device.
pub trait H3ConditionerLease {
    fn lease_id(&self) -> &str;
    fn device_id(&self) -> &str;
    fn device(&self) -> &Device;
    fn release(&mut self);
}

struct LeaseReleaseGuard<'a, L: H3ConditionerLease> {
    lease: &'a mut L,
}

impl<L: H3ConditionerLease> Drop for LeaseReleaseGuard<'_, L> {
    fn drop(&mut self) {
        self.lease.release();
    }
}

fn validated_lease_guard<'a, L: H3ConditionerLease>(
    placement: &FrozenH3ConditionerPlacement,
    lease: &'a mut L,
) -> Result<(LeaseReleaseGuard<'a, L>, Device, String, String)> {
    let guard = LeaseReleaseGuard { lease };
    placement.validate_lease(&*guard.lease)?;
    let device = guard.lease.device().clone();
    let device_id = guard.lease.device_id().to_string();
    let lease_id = guard.lease.lease_id().to_string();
    Ok((guard, device, device_id, lease_id))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum H3LoadKind {
    Cold,
    WarmReload,
}

#[derive(Clone, Debug, PartialEq)]
pub struct H3LoadTelemetry {
    pub kind: H3LoadKind,
    pub device_id: String,
    pub lease_id: String,
    pub execution_fingerprint: String,
    pub checkpoint_file_bytes: u64,
    pub memory: H3MemoryCharge,
    pub dtype_profile: H3DTypeProfile,
}

#[derive(Clone, Debug, PartialEq)]
pub enum H3LifecycleEvent {
    Loaded(H3LoadTelemetry),
    ReleasedBeforeDenoising {
        device_id: String,
        execution_fingerprint: String,
    },
    Cancelled {
        device_id: String,
        execution_fingerprint: String,
    },
}

pub struct H3ConditionerLifecycle {
    prepared: PreparedH3ConditionerAssets,
    model: Option<H3Layer50Conditioner>,
    loaded_route: Option<(String, String)>,
    ever_loaded: bool,
    telemetry: Vec<H3LifecycleEvent>,
}

impl H3ConditionerLifecycle {
    pub fn prepare(artifacts: &ConditionerArtifacts, progress: &ProgressReporter) -> Result<Self> {
        progress.checkpoint()?;
        let started = Instant::now();
        progress.stage_start("Validating H3 Qwen3-VL tokenizer, processor, and checkpoint");
        let mut last_reported = None;
        let prepared = prepare_conditioner_assets_with_progress(artifacts, &mut |verification| {
            progress
                .checkpoint()
                .map_err(|_| ArtifactError::VerificationCancelled)?;
            let should_report = last_reported.is_none_or(|previous| {
                verification.total_bytes_verified.saturating_sub(previous) >= 64 * 1024 * 1024
            }) || verification.total_bytes_verified == verification.total_bytes;
            if should_report {
                progress.weight_load(
                    "Authenticating H3 Qwen3-VL artifacts",
                    verification.total_bytes_verified,
                    verification.total_bytes,
                );
                last_reported = Some(verification.total_bytes_verified);
            }
            Ok(())
        });
        let prepared = match prepared {
            Err(H3LoadError::Artifact(ArtifactError::VerificationCancelled)) => {
                return Err(InferenceCancelled.into());
            }
            result => result?,
        };
        progress.checkpoint()?;
        progress.stage_done(
            "Validating H3 Qwen3-VL tokenizer, processor, and checkpoint",
            started.elapsed(),
        );
        Ok(Self {
            prepared,
            model: None,
            loaded_route: None,
            ever_loaded: false,
            telemetry: Vec::new(),
        })
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        self.prepared.tokenizer()
    }

    pub fn telemetry(&self) -> &[H3LifecycleEvent] {
        &self.telemetry
    }

    pub fn is_loaded(&self) -> bool {
        self.model.is_some()
    }

    /// Load, encode on the exact leased route, then release all model weights
    /// before returning the conditioning tensor to the denoising pipeline.
    /// The retained tokenizer/processor state makes subsequent calls warm
    /// reloads without retaining 51.5 GB of parameters.
    ///
    /// # Safety
    ///
    /// Frozen checkpoint files must remain immutable for the duration of this
    /// call, matching Candle's mmaped-safetensors requirement.
    pub unsafe fn encode_and_release_weights<L: H3ConditionerLease>(
        &mut self,
        input: &H3ConditionerInput,
        placement: &FrozenH3ConditionerPlacement,
        lease: &mut L,
        progress: &ProgressReporter,
    ) -> Result<Tensor> {
        let (_release, device, device_id, lease_id) = validated_lease_guard(placement, lease)?;
        progress.checkpoint()?;

        let route = (
            placement.execution_fingerprint.clone(),
            placement.device_id.clone(),
        );
        if self.model.is_some() && self.loaded_route.as_ref() != Some(&route) {
            return Err(anyhow!(
                "loaded H3 conditioner route differs from the frozen placement; refusing implicit reroute"
            ));
        }
        if self.model.is_none() {
            let kind = if self.ever_loaded {
                H3LoadKind::WarmReload
            } else {
                H3LoadKind::Cold
            };
            let stage = match kind {
                H3LoadKind::Cold => "Cold loading H3 Qwen3-VL layer-50 conditioner",
                H3LoadKind::WarmReload => "Warm reloading H3 Qwen3-VL layer-50 conditioner",
            };
            let started = Instant::now();
            progress.stage_start(stage);
            progress.weight_load(
                "H3 Qwen3-VL layer-50 conditioner",
                0,
                self.prepared.checkpoint_file_bytes(),
            );
            // SAFETY: forwarded from this method's caller, on the exact leased
            // device from the frozen placement.
            let model = unsafe { load_prepared_bf16_conditioner(&self.prepared, &device)? };
            progress.weight_load(
                "H3 Qwen3-VL layer-50 conditioner",
                self.prepared.checkpoint_file_bytes(),
                self.prepared.checkpoint_file_bytes(),
            );
            progress.stage_done(stage, started.elapsed());
            let dtype_profile = model.dtype_profile();
            self.model = Some(model);
            self.loaded_route = Some(route.clone());
            self.ever_loaded = true;
            self.telemetry
                .push(H3LifecycleEvent::Loaded(H3LoadTelemetry {
                    kind,
                    device_id: device_id.clone(),
                    lease_id,
                    execution_fingerprint: placement.execution_fingerprint.clone(),
                    checkpoint_file_bytes: self.prepared.checkpoint_file_bytes(),
                    memory: placement.memory.clone(),
                    dtype_profile,
                }));
        }

        let vision_started = input.image.is_some() || input.video.is_some();
        let vision_timer = vision_started.then(Instant::now);
        if vision_started {
            progress.stage_start("Encoding H3 Qwen3-VL vision references");
        }
        progress.stage_start("Running H3 Qwen3-VL language layers 0-49");
        let language_timer = Instant::now();
        let mut cancelled = false;
        let encode_result =
            self.model
                .as_ref()
                .expect("loaded above")
                .encode(input, &mut |checkpoint| {
                    if progress.checkpoint().is_err() {
                        cancelled = true;
                        return Err(candle_core::Error::Msg("H3 conditioner cancelled".into()));
                    }
                    match checkpoint {
                        ConditionerCheckpoint::VisionLayer { completed, total }
                            if completed == total || completed.is_multiple_of(5) =>
                        {
                            progress.info(&format!("H3 Qwen vision layer {completed}/{total}"));
                        }
                        ConditionerCheckpoint::LanguageLayer { completed, total }
                            if completed == total || completed.is_multiple_of(5) =>
                        {
                            progress.info(&format!("H3 Qwen language layer {completed}/{total}"));
                        }
                        _ => {}
                    }
                    Ok(())
                });

        // Drop on success, cancellation, or any tensor error. No model or
        // scheduler lease survives into denoising.
        self.model = None;
        self.loaded_route = None;
        if cancelled {
            self.telemetry.push(H3LifecycleEvent::Cancelled {
                device_id,
                execution_fingerprint: placement.execution_fingerprint.clone(),
            });
            return Err(InferenceCancelled.into());
        }
        let output = encode_result?;
        if let Some(started) = vision_timer {
            progress.stage_done("Encoding H3 Qwen3-VL vision references", started.elapsed());
        }
        progress.stage_done(
            "Running H3 Qwen3-VL language layers 0-49",
            language_timer.elapsed(),
        );
        self.telemetry
            .push(H3LifecycleEvent::ReleasedBeforeDenoising {
                device_id,
                execution_fingerprint: placement.execution_fingerprint.clone(),
            });
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    };

    struct TestLease {
        id: String,
        device_id: String,
        device: Device,
        released: Arc<AtomicBool>,
    }

    impl H3ConditionerLease for TestLease {
        fn lease_id(&self) -> &str {
            &self.id
        }

        fn device_id(&self) -> &str {
            &self.device_id
        }

        fn device(&self) -> &Device {
            &self.device
        }

        fn release(&mut self) {
            self.released.store(true, Ordering::Release);
        }
    }

    #[test]
    fn placement_charges_exact_parameters_and_workspace() {
        let placement = FrozenH3ConditionerPlacement::new(
            "cpu",
            H3ConditionerExecution::CpuOffloaded,
            "frozen",
            mold_candle::minimax_h3::H3_BF16_PARAMETER_BYTES,
            0,
            2_000_000_000,
        )
        .unwrap();
        assert_eq!(
            placement.memory.resident_parameter_bytes,
            mold_candle::minimax_h3::H3_BF16_PARAMETER_BYTES
        );
        assert_eq!(placement.memory.charged_vram_bytes, 0);
        assert_eq!(
            placement.memory.charged_host_bytes,
            mold_candle::minimax_h3::H3_BF16_PARAMETER_BYTES + 2_000_000_000
        );
    }

    #[test]
    fn lease_guard_releases_on_error_and_rejects_sibling_route() {
        let released = Arc::new(AtomicBool::new(false));
        let mut lease = TestLease {
            id: "lease".into(),
            device_id: "cpu-b".into(),
            device: Device::Cpu,
            released: Arc::clone(&released),
        };
        let placement = FrozenH3ConditionerPlacement::new(
            "cpu-a",
            H3ConditionerExecution::CpuOffloaded,
            "frozen",
            mold_candle::minimax_h3::H3_BF16_PARAMETER_BYTES,
            0,
            1,
        )
        .unwrap();
        assert!(validated_lease_guard(&placement, &mut lease).is_err());
        assert!(released.load(Ordering::Acquire));
    }

    #[test]
    fn metal_or_backend_mismatch_is_rejected_before_load() {
        let released = Arc::new(AtomicBool::new(false));
        let lease = TestLease {
            id: "lease".into(),
            device_id: "cpu".into(),
            device: Device::Cpu,
            released,
        };
        let placement = FrozenH3ConditionerPlacement::new(
            "cpu",
            H3ConditionerExecution::CudaResident,
            "frozen",
            0,
            mold_candle::minimax_h3::H3_BF16_PARAMETER_BYTES,
            1,
        )
        .unwrap();
        assert!(placement.validate_lease(&lease).is_err());
    }
}
