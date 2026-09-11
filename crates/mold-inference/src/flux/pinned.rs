//! Page-locked (pinned) host memory + async H2D prefetch wiring for the
//! FLUX block-streaming offload path.
//!
//! Two independent levers:
//!
//! 1. **Pinning** — every CPU-resident block weight is registered with
//!    `cuMemHostRegister_v2`. cudarc's `cuMemcpyHtoDAsync_v2` then hits the
//!    full PCIe DMA bandwidth instead of the staged-through-pageable
//!    bounce-buffer path. ComfyUI calls this `pin_memory()`
//!    (model_management.py:1152). Total pinned bytes are gated by
//!    [`PinnedMemoryTracker`] against `RAM × 0.5` (overridable via
//!    `MOLD_PINNED_VRAM_MAX_GB`).
//!
//! 2. **Side stream + reusable buffer** — a non-default `CudaStream` plus
//!    a single ~600 MB `CudaSlice<u8>` reused across blocks, so block N+1's
//!    H2D can be issued (and waited on by the compute stream via
//!    `CudaStream::wait`) while block N is still computing on the default
//!    stream. ComfyUI uses `STREAMS=2` for the same overlap
//!    (model_management.py:1379-1425).
//!
//! Both levers are no-ops on Metal/CPU.

#[cfg(feature = "cuda")]
use anyhow::Context;
use anyhow::Result;
use candle_core::Tensor;

#[cfg(feature = "cuda")]
use std::ffi::c_void;
use std::sync::{Arc, Mutex};

// ── Env / cap helpers ────────────────────────────────────────────────────────

/// Bytes-in-a-gigabyte (decimal — matches `MOLD_PINNED_VRAM_MAX_GB` user
/// expectations: 1 GB = 1,000,000,000 bytes).
const GB: u64 = 1_000_000_000;

/// Soft cap for total pinned host memory.
///
/// Resolution order:
/// 1. `MOLD_PINNED_VRAM_MAX_GB` (decimal GB; clamped to non-zero).
/// 2. `total_system_ram_bytes() × 0.5` on Linux, `× 0.4` on macOS (where
///    macOS is effectively a no-op anyway since pinning is CUDA-only).
/// 3. Fallback: 16 GB — sane for a 32 GB machine if RAM probing fails.
pub fn pinned_cap_bytes() -> u64 {
    if let Some(v) = crate::runtime_env::value("MOLD_PINNED_VRAM_MAX_GB") {
        if let Some(gb) = crate::runtime_env::parse_f64(&v) {
            if gb > 0.0 {
                return (gb * GB as f64) as u64;
            }
        }
    }
    let total = total_system_ram_bytes().unwrap_or(32 * GB);
    let frac = if cfg!(target_os = "macos") { 0.4 } else { 0.5 };
    ((total as f64) * frac) as u64
}

/// `MOLD_OFFLOAD_PREFETCH` — `off` disables the side stream, anything else
/// (including unset) leaves it on. Default-on lets the offload path opt-out
/// for debugging without anyone having to flip a flag.
pub fn prefetch_enabled_from_env() -> bool {
    match crate::runtime_env::value("MOLD_OFFLOAD_PREFETCH") {
        Some(v) => !matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "off" | "0" | "false"
        ),
        None => true,
    }
}

/// Parse one `/proc/meminfo` field, in bytes.
///
/// Pure so the platform reader is testable without a `/proc`: every fixture in
/// this module's tests is real `/proc/meminfo` text.
///
/// `field` includes its colon (`"MemTotal:"`), because the colon is what makes
/// the match exact — without it `"Mem"` would match `MemTotal`, `MemFree` and
/// `MemAvailable` alike, and the whole point of this function is which of the
/// three a caller asked for. The kernel reports these in kB meaning KiB.
fn parse_meminfo_field_bytes(text: &str, field: &str) -> Option<u64> {
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix(field) {
            let kib: u64 = rest.split_ascii_whitespace().next()?.parse().ok()?;
            return Some(kib.saturating_mul(1024));
        }
    }
    None
}

#[cfg(target_os = "linux")]
fn meminfo_field_bytes(field: &str) -> Option<u64> {
    parse_meminfo_field_bytes(&std::fs::read_to_string("/proc/meminfo").ok()?, field)
}

/// Headroom left inside this process's memory cgroup, given its limit and
/// current charge.
///
/// Pure, so the container arithmetic is testable on a host that is not in one.
/// `None` means "no limit applies" — either the controller is absent or the
/// limit is `max`/unset — and the caller then trusts the host reading.
fn cgroup_headroom_bytes(limit: Option<u64>, current: Option<u64>) -> Option<u64> {
    let limit = limit?;
    // `memory.max` is `max` on an uncapped cgroup, which the parser already
    // turns into `None`; a limit at or above the machine's own RAM is the
    // same statement written numerically, and clamping to it would be a
    // no-op that costs a syscall.
    let current = current.unwrap_or(0);
    Some(limit.saturating_sub(current))
}

/// Parse a cgroup memory file that holds a single number, or the literal
/// `max` (cgroup v2) / a sentinel at or above `PAGE_COUNTER_MAX` (cgroup v1,
/// which writes `9223372036854771712` for "unlimited").
fn parse_cgroup_bytes(text: &str) -> Option<u64> {
    let trimmed = text.trim();
    if trimmed.is_empty() || trimmed == "max" {
        return None;
    }
    let value: u64 = trimmed.parse().ok()?;
    // cgroup v1 spells "unlimited" as a number near u64::MAX rounded down to a
    // page multiple. Anything in that range is not a real cap.
    const V1_UNLIMITED_FLOOR: u64 = 1 << 62;
    (value < V1_UNLIMITED_FLOOR).then_some(value)
}

/// This process's memory-cgroup headroom, or `None` when it is not capped.
///
/// cgroup v2 first (`/sys/fs/cgroup/memory.{max,current}`), then v1
/// (`memory/memory.{limit_in_bytes,usage_in_bytes}`). Both are read relative
/// to the cgroup ROOT mount rather than resolved through `/proc/self/cgroup`:
/// inside a container the namespace root IS the limited cgroup, which is the
/// case this exists for, and a partial read is `None` rather than a guess.
#[cfg(target_os = "linux")]
fn process_cgroup_headroom_bytes() -> Option<u64> {
    let read = |path: &str| std::fs::read_to_string(path).ok();
    let v2 = cgroup_headroom_bytes(
        read("/sys/fs/cgroup/memory.max").and_then(|t| parse_cgroup_bytes(&t)),
        read("/sys/fs/cgroup/memory.current").and_then(|t| parse_cgroup_bytes(&t)),
    );
    if v2.is_some() {
        return v2;
    }
    cgroup_headroom_bytes(
        read("/sys/fs/cgroup/memory/memory.limit_in_bytes").and_then(|t| parse_cgroup_bytes(&t)),
        read("/sys/fs/cgroup/memory/memory.usage_in_bytes").and_then(|t| parse_cgroup_bytes(&t)),
    )
}

/// Host RAM the kernel believes a new allocation can have, in bytes.
///
/// **`MemAvailable`, not `MemFree`.** `MemFree` is only the untouched pages;
/// on any host that has read a checkpoint it is a small number next to the
/// page cache, and a residency decision reading it would refuse to park on a
/// 1.5 TB machine with 940 GB genuinely available. `MemAvailable` is the
/// kernel's own estimate of what is obtainable without swapping, which is the
/// question `decide_text_encoder_residency` is asking.
///
/// Falls back to `MemFree` on a kernel too old to publish `MemAvailable`
/// (pre-3.14). That is strictly conservative — `MemFree <= MemAvailable`
/// always — so the park simply engages less often rather than engaging against
/// memory that is not there.
///
/// `None` off Linux and on a `/proc` that cannot be read; every caller treats
/// an unmeasurable host as "do not park", which is the pre-campaign behaviour.
#[cfg(target_os = "linux")]
pub fn available_system_ram_bytes() -> Option<u64> {
    let host = meminfo_field_bytes("MemAvailable:").or_else(|| meminfo_field_bytes("MemFree:"))?;
    // CLAMPED BY THE CGROUP, and this is the whole reason the park could not
    // simply be switched on. `/proc/meminfo` describes the MACHINE, not this
    // process's limit: inside a memory-capped container — which is how mold
    // ships (the GHCR matrix, Lambda/RunPod provisioning) — `MemAvailable`
    // reports the host's free RAM, and a probe-driven park would reserve
    // against a ceiling it cannot see and get the process OOM-killed. Reading
    // `memory.max`/`memory.current` is what closes that, so the standing
    // objection recorded on `keep_te_in_ram` no longer applies.
    Some(match process_cgroup_headroom_bytes() {
        Some(headroom) => host.min(headroom),
        None => host,
    })
}

#[cfg(not(target_os = "linux"))]
pub fn available_system_ram_bytes() -> Option<u64> {
    None
}

/// Total system RAM in bytes. Linux: `/proc/meminfo` `MemTotal`. macOS: not
/// implemented (returns `None` so the caller falls back to a default — pinning
/// is CUDA-only and macOS uses Metal). Other unixes: `None`.
#[cfg(target_os = "linux")]
pub fn total_system_ram_bytes() -> Option<u64> {
    meminfo_field_bytes("MemTotal:")
}

#[cfg(not(target_os = "linux"))]
pub fn total_system_ram_bytes() -> Option<u64> {
    None
}

/// Largest element of `sizes`. Used to size the reusable prefetch buffer
/// at `OffloadedFluxTransformer::load` time so we never `cudaMalloc` per
/// block.
pub fn largest_block_size_bytes(sizes: &[usize]) -> usize {
    sizes.iter().copied().max().unwrap_or(0)
}

// ── Pinned memory tracker ────────────────────────────────────────────────────

/// Soft cap on cumulative bytes pinned across all blocks. Cheap clones — the
/// real state lives behind `Arc<Mutex<…>>`. Atomic add+check would race against
/// the cap on the boundary, so we mutex-guard the whole "would this fit?" RMW.
#[derive(Debug, Clone)]
pub struct PinnedMemoryTracker {
    cap_bytes: u64,
    used: Arc<Mutex<u64>>,
    capped_warning_issued: Arc<Mutex<bool>>,
}

impl PinnedMemoryTracker {
    pub fn new(cap_bytes: u64) -> Self {
        Self {
            cap_bytes,
            used: Arc::new(Mutex::new(0)),
            capped_warning_issued: Arc::new(Mutex::new(false)),
        }
    }

    #[allow(dead_code)]
    pub fn cap_bytes(&self) -> u64 {
        self.cap_bytes
    }

    pub fn used_bytes(&self) -> u64 {
        *self.used.lock().unwrap()
    }

    /// Reserve `n` bytes. Returns `true` on success, `false` if `n` would
    /// exceed the cap. On the first rejection the tracker logs a one-shot
    /// INFO so users learn pinning was capped without flooding the journal.
    pub fn try_reserve(&self, n: u64) -> bool {
        let mut used = self.used.lock().unwrap();
        if used.saturating_add(n) > self.cap_bytes {
            let mut warned = self.capped_warning_issued.lock().unwrap();
            if !*warned {
                tracing::info!(
                    "FLUX offload: pinned-memory soft cap reached ({:.2} GB used, cap {:.2} GB) — \
                     remaining blocks will fall back to pageable copies. \
                     Override with MOLD_PINNED_VRAM_MAX_GB.",
                    *used as f64 / GB as f64,
                    self.cap_bytes as f64 / GB as f64,
                );
                *warned = true;
            }
            false
        } else {
            *used += n;
            true
        }
    }

    /// Release `n` bytes (called from `PinnedRegion::drop`).
    pub fn release(&self, n: u64) {
        let mut used = self.used.lock().unwrap();
        *used = used.saturating_sub(n);
    }
}

// ── Pinned region — RAII wrapper around cuMemHostRegister/Unregister ────────

/// A region of host memory that has been page-locked via
/// `cuMemHostRegister_v2`. On drop, calls `cuMemHostUnregister`. Keeps a
/// handle on the tracker to release the byte budget.
///
/// Constructed only on `feature="cuda"` builds; on non-CUDA builds the
/// constructor is unavailable and callers go through `try_pin_to_host`
/// which short-circuits to `Ok(None)`.
pub struct PinnedRegion {
    #[cfg(feature = "cuda")]
    ptr: *mut c_void,
    n_bytes: u64,
    tracker: PinnedMemoryTracker,
}

// SAFETY: the only field `Send`/`Sync`-relevant is `ptr`, which is a stable
// host address belonging to a `Vec<T>` owned by the parent `Tensor`. The
// `PinnedRegion` does not dereference it — it only passes it to
// `cuMemHostUnregister` on drop, which is thread-safe per the CUDA driver
// contract.
#[cfg(feature = "cuda")]
unsafe impl Send for PinnedRegion {}
#[cfg(feature = "cuda")]
unsafe impl Sync for PinnedRegion {}

impl Drop for PinnedRegion {
    fn drop(&mut self) {
        #[cfg(feature = "cuda")]
        unsafe {
            use candle_core::cuda_backend::cudarc::driver::{
                preflight_current_execution_attempt_latch, sys,
            };
            if let Err(error) = preflight_current_execution_attempt_latch() {
                tracing::debug!(
                    "skipping cuMemHostUnregister for {} bytes after CUDA attempt retention: {error}",
                    self.n_bytes
                );
            } else if let Err(error) = sys::cuMemHostUnregister(self.ptr).result() {
                tracing::debug!(
                    "cuMemHostUnregister returned {error} for {} bytes (continuing)",
                    self.n_bytes
                );
            }
        }
        self.tracker.release(self.n_bytes);
    }
}

// ── try_pin_to_host ──────────────────────────────────────────────────────────

/// Page-lock the contiguous CPU buffer backing `tensor` for fast async H2D
/// DMA. Returns `Ok(None)` (a no-op) when:
/// - The build has no CUDA feature.
/// - The tensor is not on `Device::Cpu`.
/// - The tracker's soft cap would be exceeded.
/// - The driver rejects the pin call (e.g. the page is already pinned).
///
/// Returns `Ok(Some(region))` on a successful pin. The region must be held
/// for the lifetime of the underlying `Tensor` storage.
pub fn try_pin_to_host(
    tensor: &Tensor,
    tracker: &PinnedMemoryTracker,
) -> Result<Option<PinnedRegion>> {
    if !tensor.device().is_cpu() {
        return Ok(None);
    }
    let view = match cpu_tensor_byte_view(tensor)? {
        Some(v) => v,
        None => return Ok(None),
    };
    let (ptr, n_bytes) = view;
    if n_bytes == 0 {
        return Ok(None);
    }

    if !tracker.try_reserve(n_bytes as u64) {
        return Ok(None);
    }

    #[cfg(feature = "cuda")]
    {
        use candle_core::cuda_backend::cudarc::driver::{
            preflight_current_execution_attempt_latch, sys,
        };
        // CU_MEMHOSTREGISTER_PORTABLE = 1 (visible to all contexts in the
        // process). We don't request DEVICEMAP since we still hand the
        // tensor to candle's `to_device`, which does its own allocation +
        // memcpy — pinning just makes that copy fast.
        const CU_MEMHOSTREGISTER_PORTABLE: u32 = 1;
        preflight_current_execution_attempt_latch()
            .context("CUDA execution attempt rejected host-memory registration")?;
        let result = unsafe {
            sys::cuMemHostRegister_v2(ptr as *mut c_void, n_bytes, CU_MEMHOSTREGISTER_PORTABLE)
        }
        .result();
        if let Err(error) = result {
            // Already pinned, or platform refuses (WSL2 sometimes does). Roll
            // back the tracker reservation and treat as a no-op.
            tracker.release(n_bytes as u64);
            tracing::debug!(
                "cuMemHostRegister_v2 returned {error} for {} bytes — falling back to pageable",
                n_bytes
            );
            return Ok(None);
        }
        Ok(Some(PinnedRegion {
            ptr: ptr as *mut c_void,
            n_bytes: n_bytes as u64,
            tracker: tracker.clone(),
        }))
    }

    // Non-CUDA build: roll back the reservation and return a no-op.
    #[cfg(not(feature = "cuda"))]
    {
        tracker.release(n_bytes as u64);
        let _ = ptr; // suppress unused-variable warning on non-CUDA builds
        Ok(None)
    }
}

/// Get `(*const u8, num_bytes)` for the contiguous CPU buffer backing a
/// tensor. Returns `Ok(None)` if the tensor isn't a contiguous CPU tensor
/// (sliced view, GPU, etc.) — caller treats it as "skip pinning".
fn cpu_tensor_byte_view(tensor: &Tensor) -> Result<Option<(*const u8, usize)>> {
    use candle_core::{DType, Storage};

    if !tensor.is_contiguous() {
        return Ok(None);
    }

    let (storage, layout) = tensor.storage_and_layout();
    let cpu = match &*storage {
        Storage::Cpu(c) => c,
        _ => return Ok(None),
    };

    // The contiguous slice begins at layout.start_offset() and has
    // `tensor.elem_count()` elements. We pin the **whole** underlying Vec
    // — pinning is per-allocation in the CUDA driver; you cannot pin a sub-
    // page slice without UB risk if the surrounding pages get written to.
    let base_offset_bytes = layout.start_offset() * tensor.dtype().size_in_bytes();
    let elem_bytes = tensor.elem_count() * tensor.dtype().size_in_bytes();

    // For pinning correctness we want a stable pointer + total length the
    // driver will accept. Vec's heap allocation is stable as long as the
    // Tensor (and therefore the Vec) is held — `PinnedRegion`'s lifetime
    // contract delegates that to the caller.
    let (vec_ptr, vec_bytes): (*const u8, usize) = match (cpu, tensor.dtype()) {
        (candle_core::CpuStorage::U8(v), DType::U8) => {
            (v.as_ptr(), std::mem::size_of_val(v.as_slice()))
        }
        (candle_core::CpuStorage::U32(v), DType::U32) => {
            (v.as_ptr() as *const u8, std::mem::size_of_val(v.as_slice()))
        }
        (candle_core::CpuStorage::I16(v), DType::I16) => {
            (v.as_ptr() as *const u8, std::mem::size_of_val(v.as_slice()))
        }
        (candle_core::CpuStorage::I32(v), DType::I32) => {
            (v.as_ptr() as *const u8, std::mem::size_of_val(v.as_slice()))
        }
        (candle_core::CpuStorage::I64(v), DType::I64) => {
            (v.as_ptr() as *const u8, std::mem::size_of_val(v.as_slice()))
        }
        (candle_core::CpuStorage::BF16(v), DType::BF16) => {
            (v.as_ptr() as *const u8, std::mem::size_of_val(v.as_slice()))
        }
        (candle_core::CpuStorage::F16(v), DType::F16) => {
            (v.as_ptr() as *const u8, std::mem::size_of_val(v.as_slice()))
        }
        (candle_core::CpuStorage::F32(v), DType::F32) => {
            (v.as_ptr() as *const u8, std::mem::size_of_val(v.as_slice()))
        }
        (candle_core::CpuStorage::F64(v), DType::F64) => {
            (v.as_ptr() as *const u8, std::mem::size_of_val(v.as_slice()))
        }
        (candle_core::CpuStorage::F8E4M3(v), DType::F8E4M3) => {
            (v.as_ptr() as *const u8, std::mem::size_of_val(v.as_slice()))
        }
        // Dtype/storage mismatch or an exotic dtype the offload path doesn't
        // exercise — leave it to the pageable fallback.
        _ => return Ok(None),
    };

    // Belt-and-suspenders: don't pin past the Vec extent.
    if base_offset_bytes + elem_bytes > vec_bytes {
        return Ok(None);
    }

    Ok(Some((vec_ptr, vec_bytes)))
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    /// Real `/proc/meminfo` text from plato — the 1.5 TB host whose park the
    /// wave-2 UAT found could never fire. `MemFree` and `MemAvailable` differ
    /// by 890 GB here, which is the entire point of the field choice.
    const PLATO_MEMINFO: &str = "\
MemTotal:       1584079540 kB
MemFree:        50758932 kB
MemAvailable:   985694784 kB
Buffers:          123456 kB
Cached:         892340112 kB
SwapCached:            0 kB
Active:         402118400 kB
";

    /// The field choice is the defect. `MemFree` on a host that has read a
    /// checkpoint is a small number beside the page cache, and reading it
    /// would refuse to park on a machine with 985 GB genuinely available.
    #[test]
    fn the_meminfo_reader_takes_mem_available_not_mem_free() {
        let available = parse_meminfo_field_bytes(PLATO_MEMINFO, "MemAvailable:")
            .expect("MemAvailable is present");
        let free =
            parse_meminfo_field_bytes(PLATO_MEMINFO, "MemFree:").expect("MemFree is present");
        let total =
            parse_meminfo_field_bytes(PLATO_MEMINFO, "MemTotal:").expect("MemTotal is present");

        assert_eq!(available, 985_694_784 * 1024, "kB in /proc/meminfo is KiB");
        assert_eq!(free, 50_758_932 * 1024);
        assert_eq!(total, 1_584_079_540 * 1024);
        assert!(
            available > free * 19,
            "the fixture must keep the two far enough apart that confusing them is visible"
        );

        // The colon is what makes the match exact: without it, a prefix test
        // for "Mem" would answer MemTotal for all three.
        assert_ne!(
            parse_meminfo_field_bytes(PLATO_MEMINFO, "MemTotal:"),
            parse_meminfo_field_bytes(PLATO_MEMINFO, "MemFree:")
        );
    }

    /// An absent field is `None`, never a zero — a caller cannot tell a host
    /// with no memory from a host it could not measure, so the two must not
    /// share a value.
    #[test]
    fn an_absent_meminfo_field_is_none_and_a_malformed_one_is_too() {
        assert_eq!(parse_meminfo_field_bytes(PLATO_MEMINFO, "Shmem:"), None);
        assert_eq!(parse_meminfo_field_bytes("", "MemTotal:"), None);
        assert_eq!(
            parse_meminfo_field_bytes("MemTotal:       not-a-number kB\n", "MemTotal:"),
            None
        );
        assert_eq!(parse_meminfo_field_bytes("MemTotal:\n", "MemTotal:"), None);
    }

    /// The container question the park's opt-in default was standing on:
    /// `/proc/meminfo` describes the MACHINE, so a park driven by it alone
    /// reserves against a ceiling it cannot see and gets OOM-killed.
    #[test]
    fn a_capped_cgroup_bounds_the_available_reading() {
        // 16 GiB limit, 4 GiB already charged, on a host advertising 985 GB
        // available: the answer is the cgroup's 12 GiB, not the host's.
        assert_eq!(
            cgroup_headroom_bytes(Some(16 * 1024 * 1024 * 1024), Some(4 * 1024 * 1024 * 1024)),
            Some(12 * 1024 * 1024 * 1024)
        );
        // A cgroup already over its limit has no headroom, not a negative one.
        assert_eq!(
            cgroup_headroom_bytes(Some(4 * 1024 * 1024 * 1024), Some(9 * 1024 * 1024 * 1024)),
            Some(0)
        );
        // No limit means the host reading stands.
        assert_eq!(cgroup_headroom_bytes(None, Some(4)), None);
        // An unreadable current charge is treated as zero rather than
        // discarding a limit we did read.
        assert_eq!(cgroup_headroom_bytes(Some(100), None), Some(100));
    }

    /// Both cgroup spellings of "unlimited", and the shapes a partial read
    /// produces.
    #[test]
    fn cgroup_limit_parsing_knows_both_spellings_of_unlimited() {
        // cgroup v2.
        assert_eq!(parse_cgroup_bytes("max\n"), None);
        assert_eq!(parse_cgroup_bytes("17179869184\n"), Some(17_179_869_184));
        // cgroup v1's sentinel for "no limit".
        assert_eq!(parse_cgroup_bytes("9223372036854771712\n"), None);
        // Junk and emptiness are "unknown", never zero — a zero would read as
        // a cgroup with no memory at all and refuse every park.
        assert_eq!(parse_cgroup_bytes(""), None);
        assert_eq!(parse_cgroup_bytes("  \n"), None);
        assert_eq!(parse_cgroup_bytes("not-a-number"), None);
    }

    /// A pre-3.14 kernel publishes no `MemAvailable`; the reader falls back to
    /// `MemFree`, which is strictly smaller, so the park engages less often
    /// rather than against memory that is not there.
    #[test]
    fn a_kernel_without_mem_available_falls_back_to_the_conservative_field() {
        const ANCIENT: &str = "MemTotal:       16000000 kB\nMemFree:         2000000 kB\n";
        assert_eq!(parse_meminfo_field_bytes(ANCIENT, "MemAvailable:"), None);
        assert_eq!(
            parse_meminfo_field_bytes(ANCIENT, "MemFree:"),
            Some(2_000_000 * 1024)
        );
    }

    /// The platform reader itself, on the platform the defect was found on.
    ///
    /// `residency_matrix_over_host_ram_and_gpu_count` passes on any host
    /// because it feeds the inputs struct directly; nothing exercised the
    /// reader, which is how a `None`-on-Linux arm shipped and made every park
    /// decision unreachable.
    #[cfg(target_os = "linux")]
    #[test]
    fn the_linux_host_reports_a_real_available_figure() {
        let total = total_system_ram_bytes().expect("Linux reads MemTotal from /proc/meminfo");
        let available =
            available_system_ram_bytes().expect("Linux must read MemAvailable, not answer None");

        assert!(total > 0);
        assert!(available > 0, "a running host has memory available");
        assert!(
            available <= total,
            "available {available} exceeds total {total}"
        );

        // And the accessor the residency budgets actually consult answers at
        // all — the seam the defect sat in, where it returned `None`.
        //
        // Compared for PRESENCE and bounds, never for equality: this is a
        // second independent `/proc/meminfo` sample and `MemAvailable` moves
        // between two reads on a busy machine. Asserting the two numbers
        // match made this test fail under the full parallel suite while
        // passing alone.
        let via_device = crate::device::available_host_ram_bytes()
            .expect("the host-RAM accessor must not answer None on Linux");
        assert!(via_device > 0);
        assert!(
            via_device <= total,
            "accessor reported {via_device} against a {total}-byte machine"
        );
    }

    #[test]
    fn try_pin_to_host_no_op_on_cpu_tensor() {
        // A small CPU tensor: even on CUDA-feature builds with no GPU, the
        // pin call is harmless — it either succeeds or fails gracefully.
        // On non-CUDA builds, `try_pin_to_host` short-circuits to Ok(None)
        // before any FFI call. We assert the no-error contract regardless.
        let t = Tensor::zeros((4, 4), DType::F32, &Device::Cpu).unwrap();
        let tracker = PinnedMemoryTracker::new(10 * GB);
        let r = try_pin_to_host(&t, &tracker).expect("pinning a CPU tensor must not error");
        // On non-CUDA: must be None. On CUDA: may be Some if a runtime is
        // available, otherwise None — we don't care which, only that it
        // didn't blow up.
        let _ = r;
    }

    #[test]
    fn pinned_memory_tracker_caps_total_bytes() {
        let t = PinnedMemoryTracker::new(100);
        assert!(
            t.try_reserve(40),
            "first reservation under cap must succeed"
        );
        assert!(
            t.try_reserve(50),
            "second reservation that fits must succeed"
        );
        assert_eq!(t.used_bytes(), 90);
        assert!(
            !t.try_reserve(20),
            "reservation that would exceed the cap must be rejected"
        );
        assert_eq!(
            t.used_bytes(),
            90,
            "rejected reservation must not consume budget"
        );
        assert!(t.try_reserve(10), "exactly-fits reservation must succeed");
        assert_eq!(t.used_bytes(), 100);
        t.release(40);
        assert_eq!(t.used_bytes(), 60);
        assert!(
            t.try_reserve(40),
            "release should let new reservations through"
        );
    }

    #[test]
    fn prefetch_buffer_sized_for_largest_block() {
        assert_eq!(largest_block_size_bytes(&[]), 0);
        assert_eq!(largest_block_size_bytes(&[100]), 100);
        assert_eq!(largest_block_size_bytes(&[100, 200, 50]), 200);
        assert_eq!(largest_block_size_bytes(&[7, 7, 7]), 7);
        assert_eq!(
            largest_block_size_bytes(&[1, 1_000_000_000, 999]),
            1_000_000_000
        );
    }

    /// Single test that exercises both the env-on and env-off branches,
    /// because mutating the process-global env from concurrent `#[test]`
    /// threads is unsound (see device.rs:1714 comment for the same pattern).
    #[test]
    fn prefetch_disabled_via_env() {
        // SAFETY (set_var/remove_var on Rust 1.95+): we wrap both branches
        // in one #[test] so cargo's per-test thread is the only one mutating
        // MOLD_OFFLOAD_PREFETCH.
        unsafe { std::env::remove_var("MOLD_OFFLOAD_PREFETCH") };
        assert!(
            prefetch_enabled_from_env(),
            "missing var must default to enabled"
        );

        for off in ["off", "OFF", "0", "false", "False"] {
            unsafe { std::env::set_var("MOLD_OFFLOAD_PREFETCH", off) };
            assert!(
                !prefetch_enabled_from_env(),
                "value {off:?} must disable prefetch"
            );
        }

        for on in ["on", "1", "true", "yes", "anything-else"] {
            unsafe { std::env::set_var("MOLD_OFFLOAD_PREFETCH", on) };
            assert!(
                prefetch_enabled_from_env(),
                "value {on:?} must keep prefetch enabled"
            );
        }

        unsafe { std::env::remove_var("MOLD_OFFLOAD_PREFETCH") };
    }

    #[test]
    fn try_pin_returns_none_when_tracker_cap_exceeded() {
        // tracker cap=0 means *every* reservation should fail. With cuda off
        // (default-feature build) this exercises the early-return path inside
        // try_pin_to_host that runs *after* cpu_tensor_byte_view succeeds but
        // before any FFI call.
        let t = Tensor::ones((16, 16), DType::F32, &Device::Cpu).unwrap();
        let tracker = PinnedMemoryTracker::new(0);
        let r = try_pin_to_host(&t, &tracker).expect("zero-cap pin must not error");
        assert!(r.is_none(), "zero-cap tracker must yield no pinned region");
        // Once-only warning latch: a second reservation that would also
        // exceed the cap re-enters the warned branch but doesn't spam.
        assert!(!tracker.try_reserve(1));
        assert!(!tracker.try_reserve(1));
    }

    #[test]
    fn try_pin_handles_every_supported_cpu_dtype() {
        // cpu_tensor_byte_view branches on every CpuStorage variant the
        // offload path can hand it. The default-feature build short-circuits
        // through `tracker.release` after byte-view succeeds, but the dtype
        // dispatch still runs — covers the U8 / I16 / I32 / I64 / BF16 / F16
        // / F32 / F64 arms in one test.
        let device = Device::Cpu;
        let tracker = PinnedMemoryTracker::new(10 * GB);

        for dtype in [
            DType::U8,
            DType::U32,
            DType::I64,
            DType::F32,
            DType::F64,
            DType::BF16,
            DType::F16,
        ] {
            let t = Tensor::zeros((8, 8), dtype, &device).unwrap();
            try_pin_to_host(&t, &tracker)
                .unwrap_or_else(|e| panic!("dtype {dtype:?} broke try_pin_to_host: {e}"));
        }
    }

    #[test]
    fn try_pin_skips_non_contiguous_views() {
        // cpu_tensor_byte_view returns Ok(None) for non-contiguous tensors —
        // pinning a sliced view would risk overwriting memory the caller
        // doesn't own. Slice → narrow → not contiguous → pin must no-op.
        let base = Tensor::ones((8, 16), DType::F32, &Device::Cpu).unwrap();
        let view = base.transpose(0, 1).unwrap();
        assert!(
            !view.is_contiguous(),
            "transposed view must be non-contiguous"
        );
        let tracker = PinnedMemoryTracker::new(10 * GB);
        let r = try_pin_to_host(&view, &tracker).expect("non-contiguous must not error");
        assert!(r.is_none(), "non-contiguous tensors must skip pinning");
        assert_eq!(
            tracker.used_bytes(),
            0,
            "no reservation may charge against the cap when pin is skipped"
        );
    }

    #[test]
    fn try_pin_skips_when_byte_count_is_zero() {
        // A zero-element tensor has no allocation worth pinning. The
        // n_bytes==0 short-circuit must run before the tracker reservation.
        let t = Tensor::zeros((0, 8), DType::F32, &Device::Cpu).unwrap();
        let tracker = PinnedMemoryTracker::new(10 * GB);
        let r = try_pin_to_host(&t, &tracker).expect("empty tensor must not error");
        assert!(r.is_none(), "empty tensors must skip pinning");
        assert_eq!(tracker.used_bytes(), 0);
    }

    #[test]
    fn pinned_memory_tracker_cap_bytes_accessor_returns_construction_value() {
        // Round-trip the cap_bytes setter through the accessor — no other
        // existing test reads cap_bytes() since the offload path keeps it
        // private. Exercises the `#[allow(dead_code)]` getter directly.
        let t = PinnedMemoryTracker::new(7 * GB);
        assert_eq!(t.cap_bytes(), 7 * GB);
    }

    #[test]
    fn pinned_cap_respects_env_override() {
        // Same single-test pattern — env is process-global.
        unsafe { std::env::remove_var("MOLD_PINNED_VRAM_MAX_GB") };
        let baseline = pinned_cap_bytes();
        assert!(baseline > 0, "default cap must be positive");

        unsafe { std::env::set_var("MOLD_PINNED_VRAM_MAX_GB", "8") };
        assert_eq!(pinned_cap_bytes(), 8 * GB);

        unsafe { std::env::set_var("MOLD_PINNED_VRAM_MAX_GB", "0.5") };
        assert_eq!(pinned_cap_bytes(), GB / 2);

        // Bogus value — fall back to the RAM-based default.
        unsafe { std::env::set_var("MOLD_PINNED_VRAM_MAX_GB", "garbage") };
        assert_eq!(pinned_cap_bytes(), baseline);

        unsafe { std::env::remove_var("MOLD_PINNED_VRAM_MAX_GB") };
    }
}
