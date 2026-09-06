//! Convert a calibrated whole-process Wan peak to an incremental allocation.
//! A certificate is issued only by the device owner after releasing its sole
//! Wan engine and synchronizing a retained primary CUDA context.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CertifiedBaseline {
    pub owner_epoch: u64,
    pub bytes: u64,
}

impl CertifiedBaseline {
    pub fn incremental_peak(self, total_peak: u64) -> u64 {
        total_peak.saturating_sub(self.bytes)
    }

    pub fn bounded(
        self,
        owner_epoch: u64,
        attributed: Option<u64>,
        active_credit: u64,
    ) -> Option<Self> {
        if owner_epoch != self.owner_epoch || active_credit != 0 {
            return None;
        }
        let bytes = self.bytes.min(attributed?);
        (bytes > 0).then_some(Self { bytes, ..self })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const GIB: u64 = 1024 * 1024 * 1024;

    #[test]
    fn repeated_wan_peak_counts_the_retained_context_once_after_learning() {
        let baseline = CertifiedBaseline {
            owner_epoch: 7,
            bytes: 2 * GIB,
        };
        let static_peak = 23 * GIB;
        let learned_peak = 24 * GIB;
        let demand = baseline.incremental_peak(static_peak.max(learned_peak));
        assert_eq!(demand, 22 * GIB);
        assert!(
            demand <= 22 * GIB,
            "same render must fit after its cold run"
        );
        assert!(demand > 21 * GIB, "foreign pressure must still refuse it");
    }

    #[test]
    fn certificate_requires_live_attribution_and_no_overlapping_active_credit() {
        let baseline = CertifiedBaseline {
            owner_epoch: 7,
            bytes: 2 * GIB,
        };
        assert_eq!(baseline.bounded(8, Some(3 * GIB), 0), None);
        assert_eq!(baseline.bounded(7, None, 0), None);
        assert_eq!(baseline.bounded(7, Some(0), 0), None);
        assert_eq!(baseline.bounded(7, Some(3 * GIB), GIB), None);
        assert_eq!(baseline.bounded(7, Some(GIB), 0).unwrap().bytes, GIB);
        assert_eq!(
            baseline.bounded(7, Some(3 * GIB), 0).unwrap().bytes,
            2 * GIB
        );
    }
}

#[derive(Default)]
pub(crate) struct OwnerContext {
    state: std::sync::Mutex<OwnerState>,
}

#[derive(Default)]
struct OwnerState {
    disabled: bool,
    baseline: Option<CertifiedBaseline>,
    #[cfg(feature = "cuda")]
    context: Option<std::sync::Arc<cudarc::driver::CudaContext>>,
}

impl OwnerContext {
    pub fn invalidate(&self) {
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        state.disabled = true;
        state.baseline = None;
        // Keep the retained context alive; invalidating accounting must never
        // release a primary context beneath another engine's handles.
    }

    pub fn contain_poisoned(&self) {
        self.invalidate();
        #[cfg(feature = "cuda")]
        if let Some(context) = self
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .context
            .take()
        {
            std::mem::forget(context);
        }
    }

    #[cfg(feature = "cuda")]
    pub fn accepts_certificate(&self) -> bool {
        let state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        !state.disabled
    }

    pub fn snapshot(
        &self,
        epoch: u64,
        attributed: Option<u64>,
        active_credit: u64,
    ) -> Option<CertifiedBaseline> {
        let state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        if state.disabled {
            return None;
        }
        state.baseline?.bounded(epoch, attributed, active_credit)
    }

    #[cfg(feature = "cuda")]
    pub fn certify(
        &self,
        epoch: u64,
        bytes: u64,
        context: std::sync::Arc<cudarc::driver::CudaContext>,
    ) {
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        if install_certificate(&mut state, epoch, bytes) {
            state.context = Some(context);
        }
    }
}

fn install_certificate(state: &mut OwnerState, epoch: u64, bytes: u64) -> bool {
    if state.disabled || bytes == 0 {
        return false;
    }
    let bytes = state
        .baseline
        .filter(|baseline| baseline.owner_epoch == epoch)
        .map_or(bytes, |baseline| baseline.bytes.max(bytes));
    state.baseline = Some(CertifiedBaseline {
        owner_epoch: epoch,
        bytes,
    });
    true
}

impl OwnerContext {
    pub fn release_on_owner(&self) {
        self.invalidate();
        #[cfg(feature = "cuda")]
        {
            let context = self
                .state
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .context
                .take();
            drop(context);
        }
    }
}

impl Drop for OwnerContext {
    fn drop(&mut self) {
        // Exceptional owner exits may bypass clean shutdown. Do not run a
        // CUDA destructor later on whichever thread drops the worker's Arc.
        #[cfg(feature = "cuda")]
        if let Some(context) = self
            .state
            .get_mut()
            .unwrap_or_else(|error| error.into_inner())
            .context
            .take()
        {
            std::mem::forget(context);
        }
    }
}

#[cfg(feature = "cuda")]
pub(crate) fn context_only_bytes(
    context: &cudarc::driver::CudaContext,
    attributed: Option<u64>,
) -> Option<u64> {
    use cudarc::driver::{result, sys};
    let attributed = attributed?;
    if !context.has_async_alloc() {
        return None;
    }
    context.preflight_raw_call().ok()?;
    // SAFETY: the retained context owns this live device and its allocation pool.
    let pool = unsafe { result::device::get_mem_pool(context.cu_device()).ok()? };
    let read = |attribute| {
        let mut bytes = 0u64;
        context.preflight_raw_call().ok()?;
        // SAFETY: memory-size pool attributes write a cuuint64_t, and both
        // the live pool and the writable destination outlive this call.
        unsafe {
            result::mem_pool::get_attribute(
                pool,
                attribute,
                (&mut bytes) as *mut u64 as *mut std::ffi::c_void,
            )
            .ok()?;
        }
        Some(bytes)
    };
    let used = read(sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_CURRENT)?;
    let reserved = read(sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT)?;
    // Any remaining pool allocation disproves our tensor-free boundary.
    // Unused reserves may be trimmed later, so they are never certified.
    tracing::info!(
        attributed_bytes = attributed,
        pool_used_bytes = used,
        pool_reserved_bytes = reserved,
        "Wan context baseline measurement"
    );
    certifiable_bytes(attributed, used, reserved)
}

#[cfg(any(feature = "cuda", test))]
fn certifiable_bytes(attributed: u64, pool_used: u64, pool_reserved: u64) -> Option<u64> {
    if pool_used != 0 {
        return None;
    }
    attributed
        .checked_sub(pool_reserved)
        .filter(|bytes| *bytes > 0)
}

#[cfg(test)]
mod lifecycle_tests {
    use super::*;

    #[test]
    fn releasable_pool_reserves_are_never_part_of_the_certificate() {
        assert_eq!(certifiable_bytes(3_000, 0, 1_000), Some(2_000));
        assert_eq!(certifiable_bytes(3_000, 1, 1_000), None);
        assert_eq!(certifiable_bytes(3_000, 0, 3_001), None);
        assert_eq!(certifiable_bytes(3_000, 0, 3_000), None);
    }

    #[test]
    fn non_wan_work_permanently_invalidates_the_owner_certificate() {
        let owner = OwnerContext::default();
        owner.state.lock().unwrap().baseline = Some(CertifiedBaseline {
            owner_epoch: 7,
            bytes: 2_000,
        });
        assert!(owner.snapshot(7, Some(3_000), 0).is_some());
        owner.invalidate();
        assert!(owner.snapshot(7, Some(3_000), 0).is_none());
        #[cfg(feature = "cuda")]
        assert!(!owner.accepts_certificate());
    }

    #[test]
    fn a_clean_owner_boundary_monotonically_refreshes_a_growing_context() {
        let owner = OwnerContext::default();
        owner.state.lock().unwrap().baseline = Some(CertifiedBaseline {
            owner_epoch: 7,
            bytes: 500,
        });

        // Lazy CUDA libraries can add context-owned allocations on a later
        // render. A newly synchronized tensor-free boundary must replace the
        // stale smaller certificate, while a lower sample must not shrink it.
        install_certificate(&mut owner.state.lock().unwrap(), 7, 2_500);
        assert_eq!(owner.snapshot(7, Some(3_000), 0).unwrap().bytes, 2_500);
        install_certificate(&mut owner.state.lock().unwrap(), 7, 2_000);
        assert_eq!(owner.snapshot(7, Some(3_000), 0).unwrap().bytes, 2_500);
    }
}
