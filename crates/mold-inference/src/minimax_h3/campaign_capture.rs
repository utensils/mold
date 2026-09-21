//! Machine-readable Metal campaign capture rows.
//!
//! The row schema and its required columns are owned by
//! `docs/qualification/minimax-h3-metal-next-campaign.md`: one row at every
//! phase entry/exit, chunk boundary, memory-guard sample, and attempt
//! completion. The capture is opt-in through `MOLD_H3_METAL_CAMPAIGN_CAPTURE`
//! and is evidence plumbing, not a second memory authority — the budget
//! columns are exported views of `H3FactoryTargetBudgetInput::phase_budget_rows()`,
//! and the host/swap samples come from the same device helpers the shipped
//! Metal memory guard reads. A campaign run without a case id, an evidence
//! directory, or a native-allocation ceiling refuses to prepare (fail-closed);
//! a normal server never sets these variables and pays one `env` check.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex, OnceLock};
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use candle_core::Device;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::metal_memory_guard::H3MetalMemorySample;
use super::pipeline::{H3PipelineEvent, H3PipelinePhase};

pub(crate) const H3_METAL_CAMPAIGN_CAPTURE_SCHEMA: &str =
    "mold.minimax-h3.metal-campaign-capture.v1";
pub(crate) const H3_METAL_CAMPAIGN_BUDGET_SCHEMA: &str = "mold.minimax-h3.metal-campaign-budget.v1";

const EVENT_PREPARE: &str = "prepare_complete";
const EVENT_ATTACH: &str = "attach";
const EVENT_SAMPLE: &str = "sample";
const EVENT_CACHE_HIT: &str = "cache_hit";
const EVENT_PHASE_ENTER: &str = "phase_enter";
const EVENT_CHUNK: &str = "chunk";
const EVENT_PHASE_EXIT: &str = "phase_exit";
const EVENT_ATTEMPT_COMPLETE: &str = "attempt_complete";
const EVENT_ATTEMPT_FAILED: &str = "attempt_failed";
const CEILING_ENV: &str = "MOLD_H3_METAL_CAMPAIGN_CEILING_MB";

/// One required campaign capture row. The first twenty payload columns are
/// the schema the campaign document names; the trailing budget columns carry
/// the applicable phase-budget prefix row so no reader has to join files to
/// see what the admission authority charged a phase.
#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct H3MetalCampaignRow {
    pub(crate) schema: String,
    pub(crate) case_id: String,
    pub(crate) request_sha256: String,
    pub(crate) executable_sha256: String,
    pub(crate) monotonic_ns: u64,
    pub(crate) phase: String,
    pub(crate) event: String,
    pub(crate) iteration: u64,
    pub(crate) allocated_native_bytes: u64,
    pub(crate) native_peak_bytes: u64,
    /// Individual allocation requests are not observable without allocator
    /// interposition, which this capture deliberately does not ship; the
    /// column stays for instrumented builds and is `0` here.
    pub(crate) requested_allocation_bytes: u64,
    pub(crate) allocator_ceiling_bytes: u64,
    pub(crate) process_resident_bytes: u64,
    pub(crate) process_peak_resident_bytes: u64,
    pub(crate) host_available_bytes: u64,
    pub(crate) swap_used_bytes: u64,
    /// `kern.memorystatus_vm_pressure_level` (1 = normal). `0` marks a sample
    /// that could not be read; the failure is also recorded in `error`.
    pub(crate) kernel_pressure: u32,
    pub(crate) budget_phase: String,
    pub(crate) budget_device_bytes: u64,
    pub(crate) budget_host_bytes: u64,
    pub(crate) budget_combined_bytes: u64,
    pub(crate) budget_applicable: bool,
    pub(crate) budget_binding_maximum: bool,
    pub(crate) phase_complete: bool,
    pub(crate) error: Option<String>,
}

impl H3MetalCampaignRow {
    fn empty(
        core: &CampaignCaptureCore,
        phase: String,
        event: &'static str,
        budget_phase: Option<BudgetPhaseRow>,
    ) -> Self {
        Self {
            schema: H3_METAL_CAMPAIGN_CAPTURE_SCHEMA.to_string(),
            case_id: core.case_id.clone(),
            request_sha256: core.request_sha256.clone(),
            executable_sha256: core.executable_sha256.clone(),
            monotonic_ns: monotonic_ns(),
            phase,
            event: event.to_string(),
            iteration: 0,
            allocated_native_bytes: 0,
            native_peak_bytes: core.native_peak_bytes.load(Ordering::Acquire),
            requested_allocation_bytes: 0,
            allocator_ceiling_bytes: core.allocator_ceiling_bytes,
            process_resident_bytes: 0,
            process_peak_resident_bytes: 0,
            host_available_bytes: 0,
            swap_used_bytes: 0,
            kernel_pressure: 0,
            budget_phase: budget_phase.map_or(String::new(), |row| row.prefix.to_string()),
            budget_device_bytes: budget_phase.map_or(0, |row| row.device_bytes),
            budget_host_bytes: budget_phase.map_or(0, |row| row.host_bytes),
            budget_combined_bytes: budget_phase.map_or(0, |row| row.combined_bytes),
            budget_applicable: budget_phase.is_some(),
            budget_binding_maximum: budget_phase.is_some_and(|row| row.binding_maximum),
            phase_complete: false,
            error: None,
        }
    }
}

/// The capture-side view of one `H3PhaseBudgetRow`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct BudgetPhaseRow {
    pub(crate) prefix: &'static str,
    pub(crate) device_bytes: u64,
    pub(crate) host_bytes: u64,
    pub(crate) combined_bytes: u64,
    pub(crate) binding_maximum: bool,
}

impl BudgetPhaseRow {
    fn from_phase_budget(row: &crate::h3_factory::H3PhaseBudgetRow) -> Self {
        Self {
            prefix: row.phase,
            device_bytes: row.device_bytes,
            host_bytes: row.host_bytes,
            combined_bytes: row.combined_bytes,
            binding_maximum: row.binding_maximum,
        }
    }
}

/// Every budget prefix that a pipeline phase can name. `waveform_transfer`
/// deliberately has no phase: it is a transfer-only budget prefix whose
/// boundaries the campaign document requires to be measured even where the
/// pipeline emits no event for them, and the mux row prices the container
/// write.
pub(crate) fn budget_prefix_for_phase(phase: H3PipelinePhase) -> Option<&'static str> {
    use H3PipelinePhase as Phase;
    match phase {
        Phase::VaeLoad | Phase::VaeLoadChunk => Some("vae_load"),
        Phase::ReferenceDecode | Phase::ReferenceDecodeChunk => Some("reference_decode"),
        Phase::ReferencePreprocess | Phase::ReferencePreprocessChunk => {
            Some("reference_preprocess")
        }
        // Endpoint media preprocessing for FL2VA and reference preprocessing
        // for Ref2VA are the same reference-media budget prefix.
        Phase::EndpointPreprocess => Some("reference_preprocess"),
        // The conditioner's weight load prices into the qwen_encode prefix:
        // the fifteen budget prefixes have no separate conditioner-load row.
        Phase::QwenLoad | Phase::QwenLoadChunk | Phase::QwenEncode | Phase::QwenEncodeChunk => {
            Some("qwen_encode")
        }
        Phase::QwenConditioningCached => None,
        Phase::PromptEncode => Some("qwen_encode"),
        Phase::VisualConditionEncode | Phase::VisualConditionEncodeChunk => {
            Some("condition_encode")
        }
        Phase::ReferenceVisualEncode | Phase::ReferenceVisualEncodeChunk => {
            Some("reference_visual_encode")
        }
        Phase::ReferenceAudioEncode | Phase::ReferenceAudioEncodeChunk => {
            Some("reference_audio_encode")
        }
        Phase::NoiseAllocation => Some("noise_allocation"),
        Phase::TransformerLoad => Some("transformer_load"),
        Phase::Denoise | Phase::TransformerBlock => Some("denoise"),
        Phase::VisualDecode | Phase::VisualDecodeChunk => Some("visual_decode"),
        Phase::AudioDecode | Phase::AudioDecodeChunk => Some("audio_decode"),
        Phase::VideoEncode => None,
        Phase::Mux => Some("mux"),
        Phase::Validate | Phase::Staged | Phase::Complete => None,
    }
}

static MONOTONIC_ANCHOR: LazyLock<Instant> = LazyLock::new(Instant::now);

fn monotonic_ns() -> u64 {
    MONOTONIC_ANCHOR.elapsed().as_nanos().min(u64::MAX as u128) as u64
}

/// The native (malloc-zone) allocation total. Metal device buffers are NOT
/// malloc allocations and are never counted here; the budget columns and the
/// guard's device accounting describe them.
#[cfg(target_os = "macos")]
pub(crate) fn native_allocated_bytes() -> Result<u64> {
    // SAFETY: `stats` is a plain output record the allocator fills in.
    let mut stats: libc::malloc_statistics_t = unsafe { std::mem::zeroed() };
    // SAFETY: `malloc_default_zone` returns the live default zone, which is
    // valid for the life of the process.
    let zone = unsafe { libc::malloc_default_zone() };
    // SAFETY: `zone` is live and `stats` is a valid, correctly-sized output.
    unsafe { libc::malloc_zone_statistics(zone, &mut stats) };
    u64::try_from(stats.size_in_use).context("malloc zone statistics overflowed u64")
}

#[cfg(not(target_os = "macos"))]
pub(crate) fn native_allocated_bytes() -> Result<u64> {
    bail!("campaign capture native allocation sampling requires macOS")
}

/// `kern.memorystatus_vm_pressure_level`: 1 normal .. 4 critical. `Err` on
/// any read failure; callers record the failure rather than inventing a level.
#[cfg(target_os = "macos")]
pub(crate) fn kernel_pressure_level() -> Result<u32> {
    let mut level: libc::c_int = 0;
    let mut size = std::mem::size_of::<libc::c_int>();
    let name = b"kern.memorystatus_vm_pressure_level\0";
    // SAFETY: `name` is NUL-terminated; `level`/`size` are the standard
    // sysctlbyname output pair for an integer sysctl.
    let status = unsafe {
        libc::sysctlbyname(
            name.as_ptr().cast(),
            (&mut level as *mut libc::c_int).cast(),
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    if status != 0 {
        bail!("campaign capture could not read the kernel memory pressure level")
    }
    if !(1..=4).contains(&level) {
        bail!("kernel memory pressure level {level} is out of range")
    }
    u32::try_from(level).context("kernel memory pressure level overflowed u32")
}

#[cfg(not(target_os = "macos"))]
pub(crate) fn kernel_pressure_level() -> Result<u32> {
    bail!("campaign capture kernel pressure sampling requires macOS")
}

/// Injectable campaign configuration. Production resolves this from the
/// environment once, at activation; tests construct it directly so parallel
/// tests never mutate process state.
#[derive(Clone, Debug, Eq, PartialEq)]
struct CampaignConfig {
    path: PathBuf,
    case_id: String,
    allocator_ceiling_bytes: u64,
}

fn campaign_capture_path() -> Option<PathBuf> {
    std::env::var_os("MOLD_H3_METAL_CAMPAIGN_CAPTURE")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

fn required_env(name: &str) -> Result<String> {
    std::env::var(name)
        .map_err(|_| anyhow!("campaign capture requires {name} to be set"))
        .and_then(|value| {
            if value.trim().is_empty() {
                Err(anyhow!("campaign capture requires a nonempty {name}"))
            } else {
                Ok(value)
            }
        })
}

/// The native-allocation ceiling the campaign guard enforces. Required
/// whenever the campaign is active (the document's gate: no launch without a
/// derived, recorded ceiling), read in MiB to keep the operator interface
/// round.
pub(crate) fn campaign_ceiling_bytes() -> Result<u64> {
    parse_ceiling_mib(&std::env::var(CEILING_ENV).ok())
}

fn parse_ceiling_mib(value: &Option<String>) -> Result<u64> {
    let value = value
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| anyhow!("campaign capture requires {CEILING_ENV} to be set"))?;
    let mib: u64 = value
        .parse()
        .map_err(|error| anyhow!("{CEILING_ENV} is not an integer: {error}"))?;
    if mib == 0 {
        bail!("{CEILING_ENV} must be positive")
    }
    mib.checked_mul(1 << 20)
        .ok_or_else(|| anyhow!("{CEILING_ENV} overflows the byte range"))
}

struct CampaignCaptureCore {
    file: Mutex<Option<std::io::BufWriter<std::fs::File>>>,
    case_id: String,
    request_sha256: String,
    executable_sha256: String,
    allocator_ceiling_bytes: u64,
    budget_rows: HashMap<&'static str, BudgetPhaseRow>,
    device: Mutex<Option<Device>>,
    native_peak_bytes: AtomicU64,
    finished: AtomicBool,
}

/// One capture per process, set once, at preparation. A second activation in
/// the same process is a refusal: the campaign runs one case per cold process
/// group, and a shared server that unexpectedly serves a second attempt under
/// campaign variables must fail closed rather than mix evidence.
static ACTIVE_CAPTURE: OnceLock<Arc<CampaignCaptureCore>> = OnceLock::new();

// In a test binary the capture belongs to the thread that activated it.
// Production runs one campaign case per cold process, so process-wide is the
// contract there; under `cargo test` every other pipeline test is a thread of
// the SAME process whose `observe_event` reaches this static, and one of them
// completing a generation closed the capture under the activation test. That
// only ever failed in the coverage job, because nextest gives each test its
// own process.
#[cfg(test)]
thread_local! {
    static CAPTURE_ACTIVATED_HERE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

fn active_capture() -> Option<&'static Arc<CampaignCaptureCore>> {
    #[cfg(test)]
    if !CAPTURE_ACTIVATED_HERE.with(std::cell::Cell::get) {
        return None;
    }
    ACTIVE_CAPTURE.get()
}

fn emit(core: &Arc<CampaignCaptureCore>, mut row: H3MetalCampaignRow) {
    if core.finished.load(Ordering::Acquire) {
        return;
    }
    if let Err(error) = sample_row_into(&mut row) {
        row.error = Some(match row.error.take() {
            Some(existing) => format!("{existing}; {error}"),
            None => error.to_string(),
        });
    }
    match native_allocated_bytes() {
        Ok(native) => {
            core.native_peak_bytes.fetch_max(native, Ordering::AcqRel);
            row.allocated_native_bytes = native;
            row.native_peak_bytes = core.native_peak_bytes.load(Ordering::Acquire);
        }
        Err(error) => {
            row.error = Some(match row.error.take() {
                Some(existing) => format!("{existing}; {error}"),
                None => error.to_string(),
            });
        }
    }
    let mut guard = core
        .file
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let Some(file) = guard.as_mut() else {
        return;
    };
    let line = match serde_json::to_string(&row) {
        Ok(line) => line,
        Err(error) => {
            eprintln!("MiniMax H3 campaign capture could not serialize a row: {error}");
            return;
        }
    };
    if let Err(error) = writeln!(file, "{line}").and_then(|()| file.flush()) {
        eprintln!("MiniMax H3 campaign capture could not write a row: {error}");
    }
}

/// Fill the live sample columns of a row. A failed sample is recorded on the
/// row (evidence), never silently zeroed: the watchdog separately fails
/// closed on missing or stale samples.
fn sample_row_into(row: &mut H3MetalCampaignRow) -> Result<()> {
    row.process_resident_bytes = process_resident()?;
    row.process_peak_resident_bytes = process_peak_resident()?;
    let sample = sample_host_memory()?;
    row.host_available_bytes = sample.available_bytes;
    row.swap_used_bytes = sample.used_swap_bytes;
    row.kernel_pressure = kernel_pressure_level()?;
    Ok(())
}

#[cfg(target_os = "macos")]
fn process_resident() -> Result<u64> {
    super::private_runtime_observer::process_resident_bytes()
}

#[cfg(target_os = "macos")]
fn process_peak_resident() -> Result<u64> {
    super::private_runtime_observer::process_peak_resident_bytes()
}

#[cfg(target_os = "linux")]
fn process_resident() -> Result<u64> {
    super::private_runtime_observer::process_resident_bytes()
}

#[cfg(target_os = "linux")]
fn process_peak_resident() -> Result<u64> {
    super::private_runtime_observer::process_peak_resident_bytes()
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn process_resident() -> Result<u64> {
    bail!("campaign capture process-resident sampling requires macOS")
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn process_peak_resident() -> Result<u64> {
    bail!("campaign capture process peak-resident sampling requires macOS")
}

fn sample_host_memory() -> Result<H3MetalMemorySample> {
    let available_bytes = crate::device::available_system_memory_bytes()
        .filter(|bytes| *bytes > 0)
        .ok_or_else(|| anyhow!("campaign capture could not sample available host memory"))?;
    let used_swap_bytes = crate::device::used_system_swap_bytes()
        .ok_or_else(|| anyhow!("campaign capture could not sample used swap"))?;
    Ok(H3MetalMemorySample {
        available_bytes,
        used_swap_bytes,
    })
}

fn hash_executable() -> Result<String> {
    let path =
        std::env::current_exe().context("campaign capture could not locate its executable")?;
    let mut file = std::fs::File::open(&path)
        .with_context(|| format!("campaign capture could not open {}", path.display()))?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

/// Activate the capture from the environment. Inactive (no capture path
/// variable) is the production default and does nothing. A set path with a
/// missing case id, missing ceiling, unwritable evidence file, or an already
/// active capture REFUSES: the campaign must fail closed rather than mix
/// evidence.
pub(crate) fn prepare_capture(
    request_sha256: &str,
    budget_rows: &[crate::h3_factory::H3PhaseBudgetRow],
) -> Result<()> {
    let Some(path) = campaign_capture_path() else {
        return Ok(());
    };
    let config = CampaignConfig {
        path,
        case_id: required_env("MOLD_H3_METAL_CAMPAIGN_CASE_ID")?,
        allocator_ceiling_bytes: campaign_ceiling_bytes()?,
    };
    prepare_capture_with(&config, request_sha256, budget_rows)
}

fn prepare_capture_with(
    config: &CampaignConfig,
    request_sha256: &str,
    budget_rows: &[crate::h3_factory::H3PhaseBudgetRow],
) -> Result<()> {
    if request_sha256.len() != 64 || !request_sha256.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        bail!("campaign capture requires the prepared request identity sha256");
    }
    if budget_rows.is_empty() {
        bail!("campaign capture requires the exported phase budget rows");
    }
    let executable = hash_executable()?;
    let budget_rows: HashMap<&'static str, BudgetPhaseRow> = budget_rows
        .iter()
        .map(BudgetPhaseRow::from_phase_budget)
        .map(|row| (row.prefix, row))
        .collect();
    let core = Arc::new(CampaignCaptureCore {
        file: Mutex::new(None),
        case_id: config.case_id.clone(),
        request_sha256: request_sha256.to_string(),
        executable_sha256: executable,
        allocator_ceiling_bytes: config.allocator_ceiling_bytes,
        budget_rows,
        device: Mutex::new(None),
        native_peak_bytes: AtomicU64::new(0),
        finished: AtomicBool::new(false),
    });
    if ACTIVE_CAPTURE.set(core.clone()).is_err() {
        bail!(
            "campaign capture is already active for case {:?}; one attempt per process",
            ACTIVE_CAPTURE.get().expect("just set").case_id
        );
    }
    #[cfg(test)]
    CAPTURE_ACTIVATED_HERE.with(|activated| activated.set(true));

    // The budget sidecar is the allocation-free authority export for this
    // exact prepared request; the binding row names the phase the unified
    // peak binds on.
    let sidecar = serde_json::json!({
        "schema": H3_METAL_CAMPAIGN_BUDGET_SCHEMA,
        "case_id": core.case_id,
        "request_sha256": core.request_sha256,
        "allocator_ceiling_bytes": core.allocator_ceiling_bytes,
        "rows": budget_rows_for_export(&core),
    });
    let budget_path = budget_sidecar_path(&config.path);
    let budget_bytes = serde_json::to_vec_pretty(&sidecar)
        .context("campaign budget sidecar serialization failed")?;
    std::fs::write(&budget_path, budget_bytes).with_context(|| {
        format!(
            "campaign capture could not write the budget sidecar {}",
            budget_path.display()
        )
    })?;

    let file = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&config.path)
        .with_context(|| {
            format!(
                "campaign capture could not open its evidence file {} (refusing to append to existing evidence)",
                config.path.display()
            )
        })?;
    *core
        .file
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(std::io::BufWriter::new(file));
    emit(
        &core,
        H3MetalCampaignRow::empty(&core, "prepare".into(), EVENT_PREPARE, None),
    );
    Ok(())
}

fn budget_rows_for_export(core: &CampaignCaptureCore) -> Vec<serde_json::Value> {
    let mut rows: Vec<&BudgetPhaseRow> = core.budget_rows.values().collect();
    rows.sort_by_key(|row| row.prefix);
    rows.iter()
        .map(|row| {
            serde_json::json!({
                "phase": row.prefix,
                "device_bytes": row.device_bytes,
                "host_bytes": row.host_bytes,
                "combined_bytes": row.combined_bytes,
                "binding_maximum": row.binding_maximum,
            })
        })
        .collect()
}

fn budget_sidecar_path(path: &std::path::Path) -> PathBuf {
    let mut os = path.as_os_str().to_os_string();
    os.push(".budget.json");
    PathBuf::from(os)
}

/// Bind the execution device to the active capture. Phase entry/exit rows
/// synchronize this device before sampling so completed-device residency is
/// reported, per the campaign document. Test binaries may bind a CPU device;
/// production refuses anything but Metal because the campaign is Metal-scoped.
///
/// Under `MOLD_H3_METAL_CAMPAIGN_BUDGET_ONLY=1` the capture refuses here,
/// AFTER the prepared budget sidecar exists and BEFORE any model tensor is
/// allocated. This is the allocation-free pre-flight pass: the same prepared
/// request, admission chain and authority export the real attempt would use,
/// retained as a refusal rather than a render.
pub(crate) fn attach(device: &Device) -> Result<()> {
    let Some(core) = active_capture() else {
        return Ok(());
    };
    let allowed = device.is_metal() || (cfg!(test) && device.is_cpu());
    if !allowed {
        bail!("the H3 Metal campaign capture only attaches to the Metal execution device");
    }
    if std::env::var_os("MOLD_H3_METAL_CAMPAIGN_BUDGET_ONLY").is_some_and(|value| value == "1") {
        finish_attempt(
            core,
            EVENT_ATTEMPT_FAILED,
            Some(
                "campaign budget-only pass: the prepared phase budget is exported; refusing to allocate model tensors"
                    .into(),
            ),
        );
        bail!(
            "MiniMax H3 campaign budget-only pass complete for case {:?}: prepared phase budget exported, no model tensors allocated",
            core.case_id
        );
    }
    *core
        .device
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(device.clone());
    emit(
        core,
        H3MetalCampaignRow::empty(core, "attempt".into(), EVENT_ATTACH, None),
    );
    Ok(())
}

fn synchronized_device(core: &CampaignCaptureCore) {
    let guard = core
        .device
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(device) = guard.as_ref() {
        let _ = device.synchronize();
    }
}

/// Record the memory guard's periodic sample as an attempt-level row. The
/// guard keeps its own fail-closed violation logic; this row is evidence and
/// carries the capture's own fresh sample columns.
pub(crate) fn record_memory_sample() {
    let Some(core) = active_capture() else {
        return;
    };
    if core.finished.load(Ordering::Acquire) {
        return;
    }
    emit(
        core,
        H3MetalCampaignRow::empty(core, "attempt".into(), EVENT_SAMPLE, None),
    );
}

/// The event tap for pipeline progress. Driven from the runtime observer's
/// `observe_event` BEFORE the runtime-bound state gate: on Metal the
/// runtime-bound capture is never begun (there is no CUDA authority to
/// attest), so the campaign rows are this capture's own responsibility.
pub(crate) fn observe_pipeline_event(event: H3PipelineEvent) {
    let Some(core) = active_capture() else {
        return;
    };
    if core.finished.load(Ordering::Acquire) {
        return;
    }
    if event.phase == H3PipelinePhase::QwenConditioningCached {
        let mut row =
            H3MetalCampaignRow::empty(core, "QwenConditioningCached".into(), EVENT_CACHE_HIT, None);
        row.error = Some(
            "the conditioner output was served from the in-process cache; this case is not a cold conditioner measurement"
                .into(),
        );
        emit(core, row);
        return;
    }
    let phase = format!("{:?}", event.phase);
    let budget = budget_prefix_for_phase(event.phase)
        .and_then(|prefix| core.budget_rows.get(prefix).copied());
    let entry = event.completed == 0;
    let exit = event.completed == event.total;
    if entry {
        synchronized_device(core);
        let mut row = H3MetalCampaignRow::empty(core, phase.clone(), EVENT_PHASE_ENTER, budget);
        row.error = sample_row_into(&mut row)
            .err()
            .map(|error| error.to_string());
        emit(core, row);
    }
    if exit {
        synchronized_device(core);
        let mut row = H3MetalCampaignRow::empty(core, phase, EVENT_PHASE_EXIT, budget);
        row.phase_complete = true;
        row.iteration = event.completed as u64;
        row.error = sample_row_into(&mut row)
            .err()
            .map(|error| error.to_string());
        let completed = event.phase == H3PipelinePhase::Complete;
        emit(core, row);
        if completed {
            finish_attempt(core, EVENT_ATTEMPT_COMPLETE, None);
        }
        return;
    }
    if !entry {
        // Chunk rows keep the parent phase's budget columns: they are
        // descriptive, not additive, and nested membership is preserved by
        // the event column rather than by re-attributing bytes.
        let row = H3MetalCampaignRow::empty(core, phase, EVENT_CHUNK, budget);
        emit_chunk_row(core, row, event.completed);
    }
}

fn emit_chunk_row(core: &Arc<CampaignCaptureCore>, mut row: H3MetalCampaignRow, completed: usize) {
    row.iteration = completed as u64;
    emit(core, row);
}

fn finish_attempt(core: &Arc<CampaignCaptureCore>, event: &'static str, error: Option<String>) {
    let mut row = H3MetalCampaignRow::empty(core, "attempt".into(), event, None);
    row.error = error;
    row.phase_complete = true;
    emit(core, row);
    core.finished.store(true, Ordering::Release);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::h3_factory::H3PhaseBudgetRow;

    const REQUIRED_COLUMNS: [&str; 20] = [
        "case_id",
        "request_sha256",
        "executable_sha256",
        "monotonic_ns",
        "phase",
        "event",
        "iteration",
        "allocated_native_bytes",
        "native_peak_bytes",
        "requested_allocation_bytes",
        "allocator_ceiling_bytes",
        "process_resident_bytes",
        "process_peak_resident_bytes",
        "host_available_bytes",
        "swap_used_bytes",
        "kernel_pressure",
        "budget_device_bytes",
        "budget_host_bytes",
        "phase_complete",
        "error",
    ];

    fn phase_row(prefix: &'static str, device: u64, host: u64, binding: bool) -> H3PhaseBudgetRow {
        H3PhaseBudgetRow {
            phase: prefix,
            device_bytes: device,
            host_bytes: host,
            combined_bytes: device + host,
            binding_maximum: binding,
        }
    }

    fn fake_budget() -> Vec<H3PhaseBudgetRow> {
        vec![
            phase_row("vae_load", 100, 10, false),
            phase_row("qwen_encode", 200, 20, false),
            phase_row("denoise", 300, 30, true),
        ]
    }

    fn unique_tmp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "h3-campaign-capture-{name}-{}-{}",
            std::process::id(),
            monotonic_ns()
        ));
        std::fs::create_dir_all(&dir).expect("test directory creation");
        dir
    }

    fn rows_has_event(path: &std::path::Path, event: &str) -> bool {
        std::fs::read_to_string(path)
            .map(|lines| {
                lines.lines().any(|line| {
                    serde_json::from_str::<serde_json::Value>(line)
                        .ok()
                        .is_some_and(|row| row["event"] == event)
                })
            })
            .unwrap_or(false)
    }

    #[test]
    fn required_row_columns_round_trip_exactly() {
        let row = H3MetalCampaignRow::empty(
            &Arc::new(CampaignCaptureCore {
                file: Mutex::new(None),
                case_id: "case".into(),
                request_sha256: "a".repeat(64),
                executable_sha256: "b".repeat(64),
                allocator_ceiling_bytes: 8 << 30,
                budget_rows: HashMap::new(),
                device: Mutex::new(None),
                native_peak_bytes: AtomicU64::new(7),
                finished: AtomicBool::new(false),
            }),
            "Denoise".into(),
            EVENT_CHUNK,
            None,
        );
        let value = serde_json::to_value(&row).expect("row serialization");
        let object = value.as_object().expect("row is a JSON object");
        for column in REQUIRED_COLUMNS {
            assert!(object.contains_key(column), "row is missing {column}");
        }
        assert_eq!(object["schema"], H3_METAL_CAMPAIGN_CAPTURE_SCHEMA);
        assert_eq!(object["native_peak_bytes"], 7);
        let back: H3MetalCampaignRow = serde_json::from_value(value).expect("row deserialization");
        assert_eq!(back, row);
    }

    #[test]
    fn budget_prefixes_cover_the_reported_phases() {
        use H3PipelinePhase as Phase;
        let all = [
            Phase::Validate,
            Phase::EndpointPreprocess,
            Phase::VaeLoad,
            Phase::VaeLoadChunk,
            Phase::ReferenceDecode,
            Phase::ReferenceDecodeChunk,
            Phase::ReferencePreprocess,
            Phase::ReferencePreprocessChunk,
            Phase::QwenLoad,
            Phase::QwenLoadChunk,
            Phase::QwenEncode,
            Phase::QwenEncodeChunk,
            Phase::QwenConditioningCached,
            Phase::PromptEncode,
            Phase::VisualConditionEncode,
            Phase::VisualConditionEncodeChunk,
            Phase::ReferenceVisualEncode,
            Phase::ReferenceVisualEncodeChunk,
            Phase::ReferenceAudioEncode,
            Phase::ReferenceAudioEncodeChunk,
            Phase::NoiseAllocation,
            Phase::TransformerLoad,
            Phase::Denoise,
            Phase::TransformerBlock,
            Phase::VisualDecode,
            Phase::VisualDecodeChunk,
            Phase::AudioDecode,
            Phase::AudioDecodeChunk,
            Phase::VideoEncode,
            Phase::Mux,
            Phase::Staged,
            Phase::Complete,
        ];
        let mut covered: Vec<&str> = all
            .iter()
            .filter_map(|phase| budget_prefix_for_phase(*phase))
            .collect();
        covered.sort_unstable();
        covered.dedup();
        let mut expected = [
            "vae_load",
            "reference_decode",
            "reference_preprocess",
            "qwen_encode",
            "condition_encode",
            "reference_visual_encode",
            "reference_audio_encode",
            "noise_allocation",
            "transformer_load",
            "denoise",
            "visual_decode",
            "audio_decode",
            "mux",
        ]
        .to_vec();
        expected.sort_unstable();
        assert_eq!(
            covered, expected,
            "the phase mapping must cover exactly the phases a pipeline can emit"
        );
    }

    #[test]
    fn ceiling_parsing_requires_positive_mib() {
        assert!(parse_ceiling_mib(&None).is_err());
        assert!(parse_ceiling_mib(&Some("  ".into())).is_err());
        assert!(parse_ceiling_mib(&Some("zero".into())).is_err());
        assert!(parse_ceiling_mib(&Some("0".into())).is_err());
        assert_eq!(parse_ceiling_mib(&Some("8192".into())).unwrap(), 8 << 30);
        assert!(parse_ceiling_mib(&Some("99999999999999999999".into())).is_err());
    }

    #[test]
    fn capture_refuses_a_request_identity_that_is_not_sha256() {
        let dir = unique_tmp_dir("refuse-identity");
        let config = CampaignConfig {
            path: dir.join("rows.jsonl"),
            case_id: "uat-row-a".into(),
            allocator_ceiling_bytes: 8 << 30,
        };
        assert!(prepare_capture_with(&config, "short", &fake_budget()).is_err());
        assert!(prepare_capture_with(&config, &"z".repeat(64), &fake_budget()).is_err());
        assert!(
            !dir.join("rows.jsonl").exists(),
            "a refused activation must not create its evidence file"
        );
        std::fs::remove_dir_all(dir).ok();
    }

    /// The one activation-dependent test: the capture is a process-global, so
    /// every activation-path assertion runs in this single serialized test.
    #[test]
    fn capture_activates_once_and_writes_sidecar_and_rows() {
        let dir = unique_tmp_dir("end-to-end");
        let path = dir.join("rows.jsonl");
        let config = CampaignConfig {
            path: path.clone(),
            case_id: "uat-row-a".into(),
            allocator_ceiling_bytes: 8 << 30,
        };
        prepare_capture_with(&config, &"a".repeat(64), &fake_budget())
            .expect("activation succeeds");

        // `cargo test` (the coverage job) runs the crate's tests as threads of
        // ONE process, so every other pipeline test's `observe_event` reaches
        // this process-global. One of them completing a generation used to
        // close the capture before `attach` below and drop every later row;
        // nextest's process-per-test never showed it. Another test is exactly
        // this: a foreign thread reporting completion.
        std::thread::spawn(|| {
            observe_pipeline_event(H3PipelineEvent {
                phase: H3PipelinePhase::Complete,
                completed: 1,
                total: 1,
            });
            record_memory_sample();
        })
        .join()
        .expect("the foreign test thread finishes");
        assert!(
            !rows_has_event(&path, EVENT_ATTEMPT_COMPLETE) && !rows_has_event(&path, EVENT_SAMPLE),
            "another test's thread must not write into, or close, this test's capture"
        );
        // The same call from the activating thread does land, so the negative
        // above is the thread gate and not a sampler that never writes.
        record_memory_sample();
        assert!(rows_has_event(&path, EVENT_SAMPLE));

        // A second activation in the same process refuses and creates nothing.
        let second = CampaignConfig {
            path: dir.join("second.jsonl"),
            case_id: "second-case".into(),
            allocator_ceiling_bytes: 8 << 30,
        };
        let error = prepare_capture_with(&second, &"a".repeat(64), &fake_budget())
            .expect_err("the second activation refuses");
        assert!(error.to_string().contains("already active for case"));
        assert!(
            !second.path.exists(),
            "a refused activation must not create its evidence file"
        );

        let budget_path = budget_sidecar_path(&path);
        let budget: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&budget_path).expect("sidecar exists"))
                .expect("sidecar parses");
        assert_eq!(budget["schema"], H3_METAL_CAMPAIGN_BUDGET_SCHEMA);
        assert_eq!(budget["case_id"], "uat-row-a");
        assert_eq!(budget["allocator_ceiling_bytes"], 8u64 << 30);
        let sidecar_rows = budget["rows"].as_array().expect("sidecar rows");
        assert_eq!(sidecar_rows.len(), 3);
        assert!(sidecar_rows
            .iter()
            .any(|row| row["phase"] == "denoise" && row["binding_maximum"] == true));

        // A CPU device is allowed only under cfg(test); production refuses it.
        attach(&Device::Cpu).expect("test CPU device attaches");
        assert!(rows_has_event(&path, EVENT_ATTACH));

        // Vae load: entry then exit with the vae_load budget columns.
        observe_pipeline_event(H3PipelineEvent {
            phase: H3PipelinePhase::VaeLoad,
            completed: 0,
            total: 1,
        });
        observe_pipeline_event(H3PipelineEvent {
            phase: H3PipelinePhase::VaeLoad,
            completed: 1,
            total: 1,
        });
        // Denoise chunks keep the parent budget columns and carry iterations.
        observe_pipeline_event(H3PipelineEvent {
            phase: H3PipelinePhase::TransformerBlock,
            completed: 3,
            total: 20,
        });
        observe_pipeline_event(H3PipelineEvent {
            phase: H3PipelinePhase::TransformerBlock,
            completed: 4,
            total: 20,
        });
        // A cache hit is recorded as evidence that the case is not cold.
        observe_pipeline_event(H3PipelineEvent {
            phase: H3PipelinePhase::QwenConditioningCached,
            completed: 1,
            total: 1,
        });
        // Completion closes the capture.
        observe_pipeline_event(H3PipelineEvent {
            phase: H3PipelinePhase::Complete,
            completed: 1,
            total: 1,
        });
        // Rows after completion are dropped.
        observe_pipeline_event(H3PipelineEvent {
            phase: H3PipelinePhase::Mux,
            completed: 0,
            total: 1,
        });

        let lines = std::fs::read_to_string(&path).expect("capture file exists");
        let rows: Vec<H3MetalCampaignRow> = lines
            .lines()
            .map(|line| serde_json::from_str(line).expect("every line is a valid row"))
            .collect();
        assert_eq!(
            rows.first().expect("prepare row").event,
            EVENT_PREPARE,
            "the first row is the prepare row"
        );
        let attach_row = rows
            .iter()
            .find(|row| row.event == EVENT_ATTACH)
            .expect("attach row");
        assert_eq!(attach_row.phase, "attempt");

        let vae_enter = rows
            .iter()
            .find(|row| row.event == EVENT_PHASE_ENTER && row.phase == "VaeLoad")
            .expect("vae entry row");
        assert_eq!(vae_enter.budget_phase, "vae_load");
        assert_eq!(vae_enter.budget_device_bytes, 100);
        assert_eq!(vae_enter.budget_host_bytes, 10);
        assert_eq!(vae_enter.budget_combined_bytes, 110);
        assert!(vae_enter.budget_applicable);
        assert!(!vae_enter.budget_binding_maximum);
        assert!(!vae_enter.phase_complete);

        let vae_exit = rows
            .iter()
            .find(|row| row.event == EVENT_PHASE_EXIT && row.phase == "VaeLoad")
            .expect("vae exit row");
        assert!(vae_exit.phase_complete);

        let chunk = rows
            .iter()
            .find(|row| {
                row.event == EVENT_CHUNK && row.phase == "TransformerBlock" && row.iteration == 3
            })
            .expect("chunk row");
        assert_eq!(chunk.budget_phase, "denoise");
        assert_eq!(chunk.budget_device_bytes, 300);
        assert_eq!(chunk.budget_host_bytes, 30);
        assert!(chunk.budget_binding_maximum);

        let cache_hit = rows
            .iter()
            .find(|row| row.event == EVENT_CACHE_HIT)
            .expect("cache hit row");
        assert!(cache_hit
            .error
            .as_deref()
            .is_some_and(|error| error.contains("not a cold conditioner measurement")));

        let complete = rows.last().expect("rows are nonempty");
        assert_eq!(complete.event, EVENT_ATTEMPT_COMPLETE);
        assert!(complete.phase_complete);
        assert_eq!(
            rows.iter().filter(|row| row.phase == "Mux").count(),
            0,
            "no rows are recorded after completion"
        );

        for row in &rows {
            assert_eq!(row.case_id, "uat-row-a");
            assert_eq!(row.request_sha256, "a".repeat(64));
            assert_eq!(row.allocator_ceiling_bytes, 8 << 30);
            assert_eq!(row.executable_sha256.len(), 64);
            assert!(
                row.native_peak_bytes >= row.allocated_native_bytes,
                "the peak never trails the current allocation"
            );
        }
        std::fs::remove_dir_all(dir).ok();
    }
}
