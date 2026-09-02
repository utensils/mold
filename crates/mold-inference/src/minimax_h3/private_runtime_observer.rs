//! Synchronous runtime-bound evidence from one real H3 attempt.

use std::cell::RefCell;
use std::collections::HashMap;
use std::fs;
#[cfg(all(feature = "cuda", target_os = "linux"))]
use std::fs::File;
#[cfg(all(feature = "cuda", target_os = "linux"))]
use std::io::Read;
#[cfg(all(feature = "cuda", target_os = "linux"))]
use std::os::unix::ffi::OsStrExt;
#[cfg(all(feature = "cuda", target_os = "linux"))]
use std::os::unix::fs::MetadataExt;

use anyhow::{anyhow, bail, Context, Result};
use candle_core::Device;
use mold_candle::minimax_h3::{H3PrivateWorkspaceCapture, H3PrivateWorkspaceObservation};
use serde::{Deserialize, Serialize};
#[cfg(all(feature = "cuda", target_os = "linux"))]
use sha2::{Digest, Sha256};

use crate::device::{PhaseVramProbe, PhaseVramReport};

use super::pipeline::{H3PipelineEvent, H3PipelinePhase};

pub(crate) const H3_PUBLIC_RUNTIME_BOUND_OBSERVATION_SCHEMA: &str =
    "mold.minimax-h3.runtime-bound-observation.v1";

#[cfg(feature = "h3-private-uat")]
pub(crate) const H3_PRIVATE_RUNTIME_BOUND_OBSERVATION_SCHEMA: &str =
    "mold.minimax-h3.private-uat-runtime-bound-observation.v5";

#[cfg(all(feature = "h3", not(feature = "h3-private-uat")))]
pub(crate) const H3_PRIVATE_RUNTIME_BOUND_OBSERVATION_SCHEMA: &str =
    H3_PUBLIC_RUNTIME_BOUND_OBSERVATION_SCHEMA;

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct H3PrivateRuntimeProcessObservation {
    pub(crate) process_id: u32,
    pub(crate) process_start_time_ticks: u64,
    pub(crate) linux_boot_id_sha256: String,
    pub(crate) executable_device: u64,
    pub(crate) executable_inode: u64,
    pub(crate) executable_bytes: u64,
    pub(crate) executable_sha256: String,
    pub(crate) launch_argv_sha256: String,
    pub(crate) launch_environment_sha256: String,
    pub(crate) cuda_driver_version: u32,
    pub(crate) cuda_toolkit_version: u32,
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
pub(crate) fn capture_process_observation() -> Result<H3PrivateRuntimeProcessObservation> {
    use candle_core::cuda_backend::cudarc::driver::sys;

    let stat = fs::read_to_string("/proc/self/stat")
        .context("private H3 process attestation requires Linux process identity")?;
    let comm_end = stat
        .rfind(')')
        .ok_or_else(|| anyhow!("private H3 process stat has no command terminator"))?;
    let process_start_time_ticks = stat[comm_end + 1..]
        .split_ascii_whitespace()
        .nth(19)
        .ok_or_else(|| anyhow!("private H3 process stat omits its start time"))?
        .parse::<u64>()
        .context("private H3 process start time is not an integer")?;
    if process_start_time_ticks == 0 {
        bail!("private H3 process start time is zero")
    }

    let boot_id = fs::read_to_string("/proc/sys/kernel/random/boot_id")
        .context("private H3 process attestation requires a Linux boot identity")?;
    let boot_id = boot_id.trim();
    if boot_id.len() != 36
        || !boot_id.bytes().enumerate().all(|(index, byte)| {
            matches!(index, 8 | 13 | 18 | 23) && byte == b'-'
                || !matches!(index, 8 | 13 | 18 | 23)
                    && (byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        })
    {
        bail!("private H3 process attestation has an invalid Linux boot identity")
    }

    let mut executable = File::open("/proc/self/exe")
        .context("private H3 process attestation could not open its executing image")?;
    let metadata = executable
        .metadata()
        .context("private H3 process attestation could not stat its executing image")?;
    if !metadata.is_file() || metadata.len() == 0 || metadata.dev() == 0 || metadata.ino() == 0 {
        bail!("private H3 process attestation has an invalid executing image")
    }
    let mut executable_digest = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let read = executable
            .read(&mut buffer)
            .context("private H3 process attestation could not hash its executing image")?;
        if read == 0 {
            break;
        }
        executable_digest.update(&buffer[..read]);
    }

    let mut driver_version = 0_i32;
    // SAFETY: CUDA writes one integer to the supplied live pointer.
    unsafe { sys::cuDriverGetVersion(&mut driver_version) }
        .result()
        .context("private H3 process attestation could not query the CUDA driver")?;
    let cuda_driver_version =
        u32::try_from(driver_version).context("private H3 CUDA driver version is negative")?;
    if cuda_driver_version == 0 || sys::CUDA_VERSION == 0 {
        bail!("private H3 process attestation has an invalid CUDA version")
    }

    Ok(H3PrivateRuntimeProcessObservation {
        process_id: std::process::id(),
        process_start_time_ticks,
        linux_boot_id_sha256: format!("{:x}", Sha256::digest(boot_id.as_bytes())),
        executable_device: metadata.dev(),
        executable_inode: metadata.ino(),
        executable_bytes: metadata.len(),
        executable_sha256: format!("{:x}", executable_digest.finalize()),
        launch_argv_sha256: hash_os_values(
            b"mold.minimax-h3.private-launch-argv.v1\0",
            std::env::args_os().map(|value| value.into_encoded_bytes()),
        ),
        launch_environment_sha256: hash_os_values(
            b"mold.minimax-h3.private-launch-environment.v1\0",
            sorted_environment(),
        ),
        cuda_driver_version,
        cuda_toolkit_version: sys::CUDA_VERSION,
    })
}

#[cfg(not(all(feature = "cuda", target_os = "linux")))]
pub(crate) fn capture_process_observation() -> Result<H3PrivateRuntimeProcessObservation> {
    bail!("private H3 process attestation requires the Linux CUDA runtime")
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
fn sorted_environment() -> impl Iterator<Item = Vec<u8>> {
    let mut values = std::env::vars_os()
        .map(|(key, value)| {
            let mut bytes = key.as_os_str().as_bytes().to_vec();
            bytes.push(b'=');
            bytes.extend_from_slice(value.as_os_str().as_bytes());
            bytes
        })
        .collect::<Vec<_>>();
    values.sort();
    values.into_iter()
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
fn hash_os_values(domain: &[u8], values: impl Iterator<Item = Vec<u8>>) -> String {
    let mut digest = Sha256::new();
    digest.update(domain);
    for value in values {
        digest.update((value.len() as u64).to_le_bytes());
        digest.update(value);
    }
    format!("{:x}", digest.finalize())
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct H3PrivateRuntimeAuthorityObservation {
    pub(crate) bootstrap_record_sha256: String,
    pub(crate) runtime_qualification_identity_sha256: String,
    pub(crate) device_id: String,
    pub(crate) device_ordinal: usize,
    pub(crate) compute_capability: [u16; 2],
    pub(crate) attention_runtime_identity_sha256: String,
    pub(crate) attention_kernel_identity: String,
    pub(crate) attention_qualification_sha256: String,
    pub(crate) process: H3PrivateRuntimeProcessObservation,
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct H3PrivateRuntimeEnvelopeObservation {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) frames: u32,
    pub(crate) fps: u32,
    pub(crate) batch_size: u32,
    pub(crate) steps: u32,
    pub(crate) endpoint_count: u32,
    pub(crate) endpoint_anchor: String,
    pub(crate) qwen_output_text_rows: u64,
    pub(crate) qwen_vision_rows: u64,
    pub(crate) condition_visual_rows: u64,
    pub(crate) target_video_rows: u64,
    pub(crate) target_audio_rows: u64,
    pub(crate) total_packed_rows: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct H3PrivateRuntimeBoundObservation {
    pub(crate) schema: String,
    pub(crate) authority: H3PrivateRuntimeAuthorityObservation,
    pub(crate) envelope: H3PrivateRuntimeEnvelopeObservation,
    pub(crate) fixed_runtime_host_bytes: u64,
    pub(crate) fixed_runtime_device_bytes: u64,
    pub(crate) qwen_activation_workspace_bytes: u64,
    pub(crate) vae_construction_device_workspace_bytes: u64,
    pub(crate) condition_vae_workspace_device_bytes: u64,
    pub(crate) attention_workspace_device_bytes: u64,
    pub(crate) ffn_workspace_device_bytes: u64,
    pub(crate) decoder_tile_workspace_device_bytes: u64,
    pub(crate) audio_decode_workspace_device_bytes: u64,
    pub(crate) encoded_video_host_bytes_bound: u64,
    pub(crate) thumbnail_host_bytes_bound: u64,
    pub(crate) mux_output_host_bytes_bound: u64,
    pub(crate) aac_mux_staging_host_bytes: u64,
}

#[derive(Default)]
struct ObservationState {
    device: Option<Device>,
    qwen_on_cpu: bool,
    envelope: Option<H3PrivateRuntimeEnvelopeObservation>,
    authority: Option<H3PrivateRuntimeAuthorityObservation>,
    first_error: Option<String>,
    fixed_runtime_host_bytes: u64,
    overall: Option<PhaseVramProbe>,
    phases: HashMap<H3PipelinePhase, PhaseVramProbe>,
    host_phase_entry: HashMap<H3PipelinePhase, u64>,
    host_phase_peak_growth: HashMap<H3PipelinePhase, u64>,
    reports: HashMap<H3PipelinePhase, PhaseVramReport>,
    /// Set when the attempt served its conditioner output from the in-process
    /// cache. There is then no conditioner run to describe, so the record is
    /// withheld rather than synthesized (see `build_observation`).
    qwen_conditioning_cached: bool,
    encoded_video_host_bytes: u64,
    thumbnail_host_bytes: u64,
    mux_output_host_bytes: u64,
    aac_mux_staging_host_bytes: u64,
}

thread_local! {
    static ACTIVE: RefCell<Option<ObservationState>> = const { RefCell::new(None) };
}

pub(crate) struct H3PrivateRuntimeBoundCapture {
    workspace: Option<H3PrivateWorkspaceCapture>,
    active: bool,
}

impl H3PrivateRuntimeBoundCapture {
    pub(crate) fn begin(
        device: &Device,
        qwen_on_cpu: bool,
        envelope: H3PrivateRuntimeEnvelopeObservation,
        authority: H3PrivateRuntimeAuthorityObservation,
    ) -> Result<Self> {
        device
            .synchronize()
            .context("private H3 runtime-bound capture could not fence its entry")?;
        ACTIVE.with(|active| {
            if active.borrow().is_some() {
                bail!("private H3 runtime-bound capture is already active")
            }
            *active.borrow_mut() = Some(ObservationState {
                device: Some(device.clone()),
                qwen_on_cpu,
                envelope: Some(envelope),
                authority: Some(authority),
                fixed_runtime_host_bytes: process_resident_bytes()?,
                overall: Some(PhaseVramProbe::enter("h3.private.complete-attempt")),
                ..ObservationState::default()
            });
            Ok(())
        })?;
        let workspace = match H3PrivateWorkspaceCapture::begin() {
            Ok(workspace) => workspace,
            Err(error) => {
                ACTIVE.with(|active| {
                    if let Some(mut state) = active.borrow_mut().take() {
                        finish_open_probes(&mut state);
                    }
                });
                return Err(error.into());
            }
        };
        Ok(Self {
            workspace: Some(workspace),
            active: true,
        })
    }

    /// `Ok(None)` means the attempt served its conditioner from the cache and
    /// the observation is deliberately withheld; every probe is still closed
    /// and the device still fenced before returning.
    pub(crate) fn finish(mut self) -> Result<Option<H3PrivateRuntimeBoundObservation>> {
        let workspace = self
            .workspace
            .take()
            .ok_or_else(|| anyhow!("private H3 workspace capture was already consumed"))?
            .finish()?;
        let state = ACTIVE
            .with(|active| active.borrow_mut().take())
            .ok_or_else(|| anyhow!("private H3 runtime-bound capture disappeared"))?;
        self.active = false;
        build_observation(state, workspace)
    }
}

impl Drop for H3PrivateRuntimeBoundCapture {
    fn drop(&mut self) {
        if self.active {
            ACTIVE.with(|active| {
                if let Some(mut state) = active.borrow_mut().take() {
                    finish_open_probes(&mut state);
                }
            });
        }
    }
}

fn finish_open_probes(state: &mut ObservationState) {
    if let Some(device) = &state.device {
        let _ = device.synchronize();
    }
    for (_, probe) in std::mem::take(&mut state.phases) {
        let _ = probe.finish();
    }
    if let Some(probe) = state.overall.take() {
        let _ = probe.finish();
    }
}

fn captured_phase(phase: H3PipelinePhase) -> bool {
    matches!(
        phase,
        H3PipelinePhase::VaeLoad
            | H3PipelinePhase::QwenEncode
            | H3PipelinePhase::VisualConditionEncode
            | H3PipelinePhase::ReferenceVisualEncode
            | H3PipelinePhase::VisualDecode
            | H3PipelinePhase::AudioDecode
            | H3PipelinePhase::Mux
    )
}

/// The two phases that run the visual VAE's ENCODER: FL2VA encodes its
/// boundary endpoint under `VisualConditionEncode`, Ref2VA encodes every
/// ordered visual reference under `ReferenceVisualEncode`, and no render runs
/// both. The condition-VAE workspace observation is whichever ran (the larger
/// if a future task ran both); requiring FL2VA's phase alone refused every
/// completed Ref2VA render at `finish()`, after the print had been muxed
/// ("phase VisualConditionEncode has no attributed workspace peak", #1418 UAT).
const CONDITION_ENCODE_PHASES: [H3PipelinePhase; 2] = [
    H3PipelinePhase::VisualConditionEncode,
    H3PipelinePhase::ReferenceVisualEncode,
];

fn condition_encode_transient_bytes(state: &ObservationState) -> Result<u64> {
    CONDITION_ENCODE_PHASES
        .iter()
        .filter_map(|phase| phase_transient_bytes(state, *phase).ok())
        .max()
        .ok_or_else(|| {
            anyhow!(
                "private H3 visual condition encode has no attributed workspace peak: neither {:?} nor {:?} ran",
                CONDITION_ENCODE_PHASES[0],
                CONDITION_ENCODE_PHASES[1]
            )
        })
}

pub(crate) fn observe_event(event: H3PipelineEvent) {
    ACTIVE.with(|active| {
        let mut active = active.borrow_mut();
        let Some(state) = active.as_mut() else {
            return;
        };
        if event.phase == H3PipelinePhase::QwenConditioningCached {
            state.qwen_conditioning_cached = true;
        }
        if !captured_phase(event.phase) {
            return;
        }
        if state.first_error.is_some() {
            return;
        }
        if event.completed == 0 {
            if !synchronize_boundary(state, event.phase, "entry") {
                return;
            }
            if let Some(previous) = state.phases.remove(&event.phase) {
                retain_report(state, event.phase, previous.finish());
            }
            state.phases.insert(
                event.phase,
                PhaseVramProbe::enter(format!("h3.private.{:?}", event.phase)),
            );
            if let Ok(bytes) = process_resident_bytes() {
                state.host_phase_entry.insert(event.phase, bytes);
            }
        }
        if event.completed == event.total {
            if !synchronize_boundary(state, event.phase, "exit") {
                return;
            }
            if let Some(probe) = state.phases.remove(&event.phase) {
                retain_report(state, event.phase, probe.finish());
            }
            if let (Some(entry), Ok(peak)) = (
                state.host_phase_entry.remove(&event.phase),
                process_peak_resident_bytes(),
            ) {
                let growth = peak.saturating_sub(entry);
                state
                    .host_phase_peak_growth
                    .entry(event.phase)
                    .and_modify(|current| *current = (*current).max(growth))
                    .or_insert(growth);
            }
        }
    });
}

fn synchronize_boundary(
    state: &mut ObservationState,
    phase: H3PipelinePhase,
    boundary: &str,
) -> bool {
    let Some(device) = &state.device else {
        state.first_error = Some("private H3 runtime-bound capture lost its device".into());
        return false;
    };
    if let Err(error) = device.synchronize() {
        state.first_error = Some(format!(
            "private H3 {phase:?} {boundary} synchronization failed: {error}"
        ));
        false
    } else {
        true
    }
}

fn retain_report(state: &mut ObservationState, phase: H3PipelinePhase, report: PhaseVramReport) {
    let replace = state.reports.get(&phase).is_none_or(|current| {
        report.incremental_peak_pool_used().unwrap_or(0)
            > current.incremental_peak_pool_used().unwrap_or(0)
    });
    if replace {
        state.reports.insert(phase, report);
    }
}

pub(crate) fn observe_staged_host_bytes(
    encoded_video_capacity: usize,
    thumbnail_capacity: usize,
) -> Result<()> {
    let encoded_video = u64::try_from(encoded_video_capacity)
        .context("private H3 encoded video capacity exceeds u64")?;
    let thumbnail =
        u64::try_from(thumbnail_capacity).context("private H3 thumbnail capacity exceeds u64")?;
    with_state(|state| {
        state.encoded_video_host_bytes = state.encoded_video_host_bytes.max(encoded_video);
        state.thumbnail_host_bytes = state.thumbnail_host_bytes.max(thumbnail);
    })
}

pub(crate) fn observe_aac_staging_capacity(
    sample_capacity: usize,
    internal_staging_bytes: u64,
) -> Result<()> {
    let samples = sample_capacity
        .checked_mul(std::mem::size_of::<f32>())
        .and_then(|value| u64::try_from(value).ok())
        .context("private H3 AAC staging capacity exceeds u64")?;
    let bytes = samples
        .checked_add(internal_staging_bytes)
        .context("private H3 total AAC staging capacity exceeds u64")?;
    with_state(|state| state.aac_mux_staging_host_bytes = bytes)
}

pub(crate) fn observe_mux_output_capacity(capacity: usize) -> Result<()> {
    let bytes = u64::try_from(capacity).context("private H3 mux output capacity exceeds u64")?;
    with_state(|state| state.mux_output_host_bytes = bytes)
}

fn with_state(update: impl FnOnce(&mut ObservationState)) -> Result<()> {
    ACTIVE.with(|active| {
        let mut active = active.borrow_mut();
        if let Some(state) = active.as_mut() {
            update(state);
        }
        Ok(())
    })
}

fn phase_transient_bytes(state: &ObservationState, phase: H3PipelinePhase) -> Result<u64> {
    state
        .reports
        .get(&phase)
        .and_then(PhaseVramReport::incremental_transient_pool_used)
        .ok_or_else(|| anyhow!("private H3 phase {phase:?} has no attributed workspace peak"))
}

fn process_resident_bytes() -> Result<u64> {
    let statm = fs::read_to_string("/proc/self/statm")
        .context("private H3 runtime capture requires Linux process memory evidence")?;
    let resident_pages = statm
        .split_ascii_whitespace()
        .nth(1)
        .ok_or_else(|| anyhow!("private H3 process memory evidence omits resident pages"))?
        .parse::<u64>()
        .context("private H3 resident page count is not an integer")?;
    // SAFETY: sysconf is read-only and `_SC_PAGESIZE` has no pointer contract.
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    let page_size =
        u64::try_from(page_size).context("private H3 process page size is unavailable")?;
    resident_pages
        .checked_mul(page_size)
        .filter(|bytes| *bytes > 0)
        .ok_or_else(|| anyhow!("private H3 resident bytes overflow or are zero"))
}

fn process_peak_resident_bytes() -> Result<u64> {
    let status = fs::read_to_string("/proc/self/status")
        .context("private H3 runtime capture requires Linux peak-memory evidence")?;
    let kib = status
        .lines()
        .find_map(|line| line.strip_prefix("VmHWM:")?.split_ascii_whitespace().next())
        .ok_or_else(|| anyhow!("private H3 process memory evidence omits VmHWM"))?
        .parse::<u64>()
        .context("private H3 VmHWM is not an integer")?;
    kib.checked_mul(1024)
        .filter(|bytes| *bytes > 0)
        .ok_or_else(|| anyhow!("private H3 VmHWM overflows or is zero"))
}

fn build_observation(
    mut state: ObservationState,
    workspace: H3PrivateWorkspaceObservation,
) -> Result<Option<H3PrivateRuntimeBoundObservation>> {
    // A boundary failure is fatal on EVERY exit, withheld ones included. This
    // is the error `synchronize_boundary` records when a captured phase could
    // not fence the device, and a render that could not fence its own phases
    // is not a render whose print may be published just because its
    // conditioner came from the cache.
    if let Some(error) = state.first_error.take() {
        finish_open_probes(&mut state);
        bail!(error)
    }
    // Withhold, never synthesize. A hit ran no conditioner, so there is no
    // Qwen workspace peak to report; and on Ref2VA the outer `QwenEncode`
    // boundary still brackets `construct_vaes`, so a probe read here would
    // attribute VAE construction bytes to the conditioner. Every exit from
    // this function first closes the probes the state owns, and this one
    // fences the device exactly as the terminal sample does.
    if state.qwen_conditioning_cached {
        finish_open_probes(&mut state);
        if let Some(device) = state.device.as_ref() {
            device
                .synchronize()
                .context("private H3 withheld runtime-bound capture could not fence its exit")?;
        }
        return Ok(None);
    }
    if !state.phases.is_empty() || !state.host_phase_entry.is_empty() {
        finish_open_probes(&mut state);
        bail!("private H3 runtime-bound capture ended with an incomplete phase lifecycle")
    }
    let Some(device) = state.device.as_ref() else {
        finish_open_probes(&mut state);
        bail!("private H3 runtime-bound capture lost its device")
    };
    let terminal_fence = device.synchronize();
    if let Err(error) = terminal_fence {
        finish_open_probes(&mut state);
        return Err(error)
            .context("private H3 runtime-bound capture could not fence its terminal sample");
    }
    let overall = state
        .overall
        .take()
        .ok_or_else(|| anyhow!("private H3 overall device probe disappeared"))?
        .finish();
    let fixed_runtime_device_bytes = overall
        .global_total
        .zip(overall.global_free_entry)
        .map(|(total, free)| total.saturating_sub(free))
        .filter(|bytes| *bytes > 0)
        .ok_or_else(|| anyhow!("private H3 attempt has no nonzero device-entry baseline"))?;
    let host_qwen = state
        .host_phase_peak_growth
        .get(&H3PipelinePhase::QwenEncode)
        .copied();
    let device_qwen = state
        .reports
        .get(&H3PipelinePhase::QwenEncode)
        .and_then(PhaseVramReport::incremental_peak_pool_used);
    let qwen_activation_workspace_bytes =
        routed_qwen_workspace(state.qwen_on_cpu, host_qwen, device_qwen)
            .ok_or_else(|| anyhow!("private H3 Qwen encode has no routed workspace peak"))?;
    let observation = H3PrivateRuntimeBoundObservation {
        schema: H3_PRIVATE_RUNTIME_BOUND_OBSERVATION_SCHEMA.into(),
        authority: state
            .authority
            .clone()
            .ok_or_else(|| anyhow!("private H3 runtime authority observation disappeared"))?,
        envelope: state
            .envelope
            .clone()
            .ok_or_else(|| anyhow!("private H3 runtime envelope observation disappeared"))?,
        fixed_runtime_host_bytes: state.fixed_runtime_host_bytes,
        fixed_runtime_device_bytes,
        qwen_activation_workspace_bytes,
        vae_construction_device_workspace_bytes: phase_transient_bytes(
            &state,
            H3PipelinePhase::VaeLoad,
        )?,
        condition_vae_workspace_device_bytes: condition_encode_transient_bytes(&state)?,
        attention_workspace_device_bytes: workspace.attention_workspace_device_bytes,
        ffn_workspace_device_bytes: workspace.ffn_workspace_device_bytes,
        decoder_tile_workspace_device_bytes: phase_transient_bytes(
            &state,
            H3PipelinePhase::VisualDecode,
        )?,
        audio_decode_workspace_device_bytes: phase_transient_bytes(
            &state,
            H3PipelinePhase::AudioDecode,
        )?,
        encoded_video_host_bytes_bound: state.encoded_video_host_bytes,
        thumbnail_host_bytes_bound: state.thumbnail_host_bytes,
        mux_output_host_bytes_bound: state.mux_output_host_bytes,
        aac_mux_staging_host_bytes: state.aac_mux_staging_host_bytes,
    };
    let encoded = serde_json::to_value(&observation)?;
    let zero_fields = encoded
        .as_object()
        .into_iter()
        .flat_map(|object| object.iter())
        .filter(|(key, _)| {
            !matches!(
                key.as_str(),
                "schema" | "vae_construction_device_workspace_bytes"
            )
        })
        .filter(|(_, value)| value.as_u64() == Some(0))
        .map(|(key, _)| key.as_str())
        .collect::<Vec<_>>();
    if !zero_fields.is_empty() {
        bail!(
            "private H3 runtime-bound observation contains a zero byte count: {}",
            zero_fields.join(", ")
        )
    }
    Ok(Some(observation))
}

fn routed_qwen_workspace(
    qwen_on_cpu: bool,
    host_bytes: Option<u64>,
    device_bytes: Option<u64>,
) -> Option<u64> {
    if qwen_on_cpu {
        host_bytes
    } else {
        device_bytes
    }
    .filter(|bytes| *bytes > 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn captured_phase_set_is_exact() {
        assert!(captured_phase(H3PipelinePhase::QwenEncode));
        assert!(captured_phase(H3PipelinePhase::VisualConditionEncode));
        assert!(captured_phase(H3PipelinePhase::ReferenceVisualEncode));
        assert!(!captured_phase(H3PipelinePhase::ReferenceVisualEncodeChunk));
        assert!(!captured_phase(H3PipelinePhase::Denoise));
        assert!(!captured_phase(H3PipelinePhase::Complete));
        assert!(
            !captured_phase(H3PipelinePhase::QwenConditioningCached),
            "a served conditioner has no workspace to attribute"
        );
    }

    fn test_envelope() -> H3PrivateRuntimeEnvelopeObservation {
        H3PrivateRuntimeEnvelopeObservation {
            width: 768,
            height: 768,
            frames: 124,
            fps: 24,
            batch_size: 1,
            steps: 5,
            endpoint_count: 1,
            endpoint_anchor: "first".into(),
            qwen_output_text_rows: 1_024,
            qwen_vision_rows: 4_096,
            condition_visual_rows: 1,
            target_video_rows: 2,
            target_audio_rows: 3,
            total_packed_rows: 4,
        }
    }

    fn test_authority() -> H3PrivateRuntimeAuthorityObservation {
        H3PrivateRuntimeAuthorityObservation {
            bootstrap_record_sha256: "0".repeat(64),
            runtime_qualification_identity_sha256: "1".repeat(64),
            device_id: "cuda:0".into(),
            device_ordinal: 0,
            compute_capability: [8, 9],
            attention_runtime_identity_sha256: "2".repeat(64),
            attention_kernel_identity: "kernel".into(),
            attention_qualification_sha256: "3".repeat(64),
            process: H3PrivateRuntimeProcessObservation {
                process_id: 1,
                process_start_time_ticks: 1,
                linux_boot_id_sha256: "4".repeat(64),
                executable_device: 1,
                executable_inode: 1,
                executable_bytes: 1,
                executable_sha256: "5".repeat(64),
                launch_argv_sha256: "6".repeat(64),
                launch_environment_sha256: "7".repeat(64),
                cuda_driver_version: 1,
                cuda_toolkit_version: 1,
            },
        }
    }

    /// A hit has no conditioner run to describe, so the record is WITHHELD
    /// rather than synthesized: `build_observation` would otherwise attribute
    /// the Ref2VA outer boundary's VAE construction to "Qwen activation
    /// workspace", or hard-fail a print that is already muxed.
    ///
    /// `cfg(not(cuda))` because `PhaseVramProbe::enter` retains a CUDA context
    /// on a CUDA build, and a unit test must not claim a device.
    #[test]
    #[cfg(not(feature = "cuda"))]
    fn cached_conditioner_withholds_the_observation_and_rearms_cleanly() {
        let depth_before = crate::device::open_probe_depth();
        let capture = H3PrivateRuntimeBoundCapture::begin(
            &Device::Cpu,
            false,
            test_envelope(),
            test_authority(),
        )
        .expect("capture begins on the host device");
        // Leave one phase probe OPEN on purpose: the withheld path must still
        // close it, or the next attempt on this worker thread inherits it.
        observe_event(H3PipelineEvent {
            phase: H3PipelinePhase::QwenEncode,
            completed: 0,
            total: 1,
        });
        observe_event(H3PipelineEvent {
            phase: H3PipelinePhase::QwenConditioningCached,
            completed: 1,
            total: 1,
        });
        assert!(
            capture
                .finish()
                .expect("withholding is not a failure")
                .is_none(),
            "a served conditioner produces no runtime-bound observation"
        );
        // The floor stack is the only durable trace of an unclosed probe, and
        // the second `begin()` below would succeed either way: `finish` takes
        // ACTIVE unconditionally. This is the assertion that actually proves
        // the withheld path ran `finish_open_probes`.
        assert_eq!(
            crate::device::open_probe_depth(),
            depth_before,
            "the withheld path must close the overall probe and the open phase probe"
        );

        let second = H3PrivateRuntimeBoundCapture::begin(
            &Device::Cpu,
            false,
            test_envelope(),
            test_authority(),
        )
        .expect("the withheld attempt released the capture slot");
        observe_event(H3PipelineEvent {
            phase: H3PipelinePhase::QwenEncode,
            completed: 0,
            total: 1,
        });
        observe_event(H3PipelineEvent {
            phase: H3PipelinePhase::QwenEncode,
            completed: 1,
            total: 1,
        });
        let error = second
            .finish()
            .expect_err("a host-only probe has no device baseline to report")
            .to_string();
        assert!(
            !error.contains("incomplete phase lifecycle"),
            "the second attempt must not inherit the first attempt's open probes: {error}"
        );
    }

    /// A withheld observation is not a withheld FAILURE. `synchronize_boundary`
    /// records `first_error` at every captured-phase boundary — VaeLoad, the
    /// Ref2VA outer QwenEncode, the visual encoders, both decoders, Mux — and a
    /// device that could not be fenced there is fatal whether or not the
    /// conditioner came from the cache. Before this ordering the flag swallowed
    /// it and the print was published.
    #[test]
    fn a_withheld_observation_still_reports_a_boundary_failure() {
        let state = ObservationState {
            qwen_conditioning_cached: true,
            first_error: Some("private H3 VaeLoad exit synchronization failed: device lost".into()),
            ..ObservationState::default()
        };
        let error = build_observation(state, H3PrivateWorkspaceObservation::default())
            .expect_err("a boundary failure is fatal on a cache hit too")
            .to_string();
        assert!(
            error.contains("VaeLoad exit synchronization failed"),
            "the withheld path must surface the recorded boundary failure: {error}"
        );
    }

    #[test]
    #[cfg(not(feature = "cuda"))]
    fn a_conditioner_encode_still_produces_an_observation_attempt() {
        let capture = H3PrivateRuntimeBoundCapture::begin(
            &Device::Cpu,
            false,
            test_envelope(),
            test_authority(),
        )
        .expect("capture begins on the host device");
        observe_event(H3PipelineEvent {
            phase: H3PipelinePhase::QwenEncode,
            completed: 0,
            total: 1,
        });
        observe_event(H3PipelineEvent {
            phase: H3PipelinePhase::QwenEncode,
            completed: 1,
            total: 1,
        });
        assert!(
            capture.finish().is_err(),
            "without the cached flag the record is still demanded, not withheld"
        );
    }

    #[test]
    #[cfg(all(feature = "h3", not(feature = "h3-private-uat")))]
    fn public_runtime_emits_only_the_public_observation_schema() {
        assert_eq!(
            H3_PRIVATE_RUNTIME_BOUND_OBSERVATION_SCHEMA,
            H3_PUBLIC_RUNTIME_BOUND_OBSERVATION_SCHEMA
        );
        assert!(!H3_PRIVATE_RUNTIME_BOUND_OBSERVATION_SCHEMA.contains("private-uat"));
    }

    #[test]
    fn qwen_workspace_never_crosses_its_placement_domain() {
        assert_eq!(routed_qwen_workspace(true, Some(19), Some(3)), Some(19));
        assert_eq!(routed_qwen_workspace(false, Some(19), Some(3)), Some(3));
        assert_eq!(routed_qwen_workspace(true, None, Some(3)), None);
        assert_eq!(routed_qwen_workspace(false, Some(19), None), None);
    }

    #[test]
    fn vae_load_can_have_zero_transient_after_retained_allocations_are_removed() {
        let mut state = ObservationState::default();
        state.reports.insert(
            H3PipelinePhase::VaeLoad,
            PhaseVramReport {
                phase: "h3.private.VaeLoad".into(),
                peak_pool_used: Some(12),
                peak_pool_reserved: Some(12),
                entry_pool_used: Some(2),
                delta_pool_used: Some(10),
                global_free_entry: Some(100),
                global_free_exit: Some(90),
                global_total: Some(200),
                predicted_bytes: None,
                elapsed: std::time::Duration::from_millis(1),
                attribution: crate::device::VramAttribution::Pool,
            },
        );
        assert_eq!(
            phase_transient_bytes(&state, H3PipelinePhase::VaeLoad).unwrap(),
            0
        );
        assert!(phase_transient_bytes(&state, H3PipelinePhase::VisualDecode).is_err());
    }

    fn report(phase: &str, peak: u64, entry: u64, exit_used: u64) -> PhaseVramReport {
        PhaseVramReport {
            phase: phase.into(),
            peak_pool_used: Some(peak),
            peak_pool_reserved: Some(peak),
            entry_pool_used: Some(entry),
            delta_pool_used: Some(exit_used as i64 - entry as i64),
            global_free_entry: Some(1_000),
            global_free_exit: Some(900),
            global_total: Some(2_000),
            predicted_bytes: None,
            elapsed: std::time::Duration::from_millis(1),
            attribution: crate::device::VramAttribution::Pool,
        }
    }

    /// A Ref2VA render encodes its references under `ReferenceVisualEncode`
    /// and never runs FL2VA's `VisualConditionEncode`; the condition-VAE
    /// workspace has to come from the phase that ran, and only a render that
    /// ran neither is refused.
    #[test]
    fn condition_encode_workspace_comes_from_whichever_encoder_phase_ran() {
        let mut ref2va = ObservationState::default();
        ref2va.reports.insert(
            H3PipelinePhase::ReferenceVisualEncode,
            report("h3.private.ReferenceVisualEncode", 40, 10, 10),
        );
        assert_eq!(condition_encode_transient_bytes(&ref2va).unwrap(), 30);

        let mut fl2va = ObservationState::default();
        fl2va.reports.insert(
            H3PipelinePhase::VisualConditionEncode,
            report("h3.private.VisualConditionEncode", 25, 5, 5),
        );
        assert_eq!(condition_encode_transient_bytes(&fl2va).unwrap(), 20);

        let error = condition_encode_transient_bytes(&ObservationState::default())
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("neither VisualConditionEncode nor ReferenceVisualEncode ran"),
            "{error}"
        );
    }
}
