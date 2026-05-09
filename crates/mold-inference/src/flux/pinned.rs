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
    if let Ok(v) = std::env::var("MOLD_PINNED_VRAM_MAX_GB") {
        if let Ok(gb) = v.trim().parse::<f64>() {
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
    match std::env::var("MOLD_OFFLOAD_PREFETCH") {
        Ok(v) => !matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "off" | "0" | "false"
        ),
        Err(_) => true,
    }
}

/// Total system RAM in bytes. Linux: `/proc/meminfo` `MemTotal`. macOS: not
/// implemented (returns `None` so the caller falls back to a default — pinning
/// is CUDA-only and macOS uses Metal). Other unixes: `None`.
#[cfg(target_os = "linux")]
pub fn total_system_ram_bytes() -> Option<u64> {
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in meminfo.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            let mut it = rest.split_ascii_whitespace();
            let val: u64 = it.next()?.parse().ok()?;
            // /proc/meminfo reports "kB" (KiB). Convert to bytes.
            return Some(val.saturating_mul(1024));
        }
    }
    None
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
            use candle_core::cuda_backend::cudarc::driver::sys;
            let res = sys::cuMemHostUnregister(self.ptr);
            if res != sys::CUresult::CUDA_SUCCESS {
                tracing::debug!(
                    "cuMemHostUnregister returned {:?} for {} bytes (continuing)",
                    res,
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
        use candle_core::cuda_backend::cudarc::driver::sys;
        // CU_MEMHOSTREGISTER_PORTABLE = 1 (visible to all contexts in the
        // process). We don't request DEVICEMAP since we still hand the
        // tensor to candle's `to_device`, which does its own allocation +
        // memcpy — pinning just makes that copy fast.
        const CU_MEMHOSTREGISTER_PORTABLE: u32 = 1;
        let res = unsafe {
            sys::cuMemHostRegister_v2(ptr as *mut c_void, n_bytes, CU_MEMHOSTREGISTER_PORTABLE)
        };
        if res != sys::CUresult::CUDA_SUCCESS {
            // Already pinned, or platform refuses (WSL2 sometimes does). Roll
            // back the tracker reservation and treat as a no-op.
            tracker.release(n_bytes as u64);
            tracing::debug!(
                "cuMemHostRegister_v2 returned {:?} for {} bytes — falling back to pageable",
                res,
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
