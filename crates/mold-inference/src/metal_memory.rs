//! Native read-only Metal telemetry. No models are loaded by these probes.

use mold_core::metal_memory::MetalMemorySnapshot;

/// `None` means this binary/platform has no Metal backend, not a failed probe.
pub fn snapshot(ordinal: usize) -> Option<MetalMemorySnapshot> {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        use mold_core::metal_memory::MetalWiredLimit;
        let (wired_limit, mut error) = match read_wired_limit() {
            Ok(Some(0)) => (MetalWiredLimit::Automatic, None),
            Ok(Some(mib)) => (MetalWiredLimit::Explicit { mib }, None),
            Ok(None) => (MetalWiredLimit::Unsupported, None),
            Err(e) => (
                MetalWiredLimit::Unavailable,
                Some(format!("Cannot read Metal wired limit: {e}")),
            ),
        };
        let (recommended_bytes, allocated_bytes) = match crate::device::metal_device(ordinal) {
            Ok(device) => match device.as_metal_device() {
                Ok(device) => {
                    let device = device.device();
                    let recommended = device.recommended_max_working_set_size();
                    if recommended == 0 {
                        error =
                            Some("Metal reported an unavailable working-set recommendation".into());
                    }
                    (
                        (recommended > 0).then_some(recommended),
                        Some(device.current_allocated_size()),
                    )
                }
                Err(e) => {
                    error = Some(format!("Cannot read Metal device: {e}"));
                    (None, None)
                }
            },
            Err(e) => {
                error = Some(format!("Cannot open Metal device: {e}"));
                (None, None)
            }
        };
        let physical_bytes = crate::device::total_system_memory_bytes();
        let available_host_bytes = crate::device::available_system_memory_bytes();
        if physical_bytes.is_none() || available_host_bytes.is_none() {
            error = Some("Cannot read macOS host-memory budget".into());
        }
        Some(
            MetalMemorySnapshot {
                wired_limit,
                physical_bytes,
                available_host_bytes,
                recommended_bytes,
                allocated_bytes,
                effective_capacity_bytes: None,
                allocation_headroom_bytes: None,
                error,
            }
            .resolve(),
        )
    }
    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    {
        let _ = ordinal;
        None
    }
}

/// The kernel exposes an unsigned 32-bit integer in MiB. `None` means the key
/// is absent; permission/ABI/read failures are errors, never automatic mode.
#[cfg(target_os = "macos")]
pub fn read_wired_limit() -> std::io::Result<Option<u32>> {
    let mut value: u32 = 0;
    let mut len = std::mem::size_of::<u32>();
    // SAFETY: exact native uint ABI and writable, correctly sized storage.
    let result = unsafe {
        libc::sysctlbyname(
            c"iogpu.wired_limit_mb".as_ptr(),
            (&raw mut value).cast(),
            &raw mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if result != 0 {
        let error = std::io::Error::last_os_error();
        return if error.raw_os_error() == Some(libc::ENOENT) {
            Ok(None)
        } else {
            Err(error)
        };
    }
    if len != std::mem::size_of::<u32>() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "unexpected wired-limit kernel ABI",
        ));
    }
    Ok(Some(value))
}

#[cfg(all(test, target_os = "macos", feature = "metal"))]
mod tests {
    #[test]
    fn metal_memory_native_snapshot_is_read_only_and_policy_bounded() {
        let before = super::read_wired_limit().expect("read native sysctl");
        let sample = super::snapshot(0).expect("Metal build");
        assert!(sample.error.is_none(), "{sample:?}");
        let cap = sample.effective_capacity_bytes.expect("effective capacity");
        assert!(cap <= sample.recommended_bytes.unwrap());
        assert!(cap <= sample.physical_bytes.unwrap());
        assert!(sample.allocation_headroom_bytes.unwrap() <= cap);
        assert_eq!(super::read_wired_limit().unwrap(), before);
    }
}

#[cfg(all(test, not(all(target_os = "macos", feature = "metal"))))]
mod tests {
    #[test]
    fn metal_memory_no_backend_is_distinct_from_probe_failure() {
        assert!(super::snapshot(0).is_none());
    }
}
