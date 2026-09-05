//! Native read-only Metal telemetry. No models are loaded by these probes.

use mold_core::metal_memory::MetalMemorySnapshot;

#[cfg(test)]
thread_local! {
    static TEST_SNAPSHOT: std::cell::RefCell<Option<MetalMemorySnapshot>> = const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
pub(crate) fn with_test_snapshot<T>(sample: MetalMemorySnapshot, run: impl FnOnce() -> T) -> T {
    struct Restore(Option<MetalMemorySnapshot>);
    impl Drop for Restore {
        fn drop(&mut self) {
            TEST_SNAPSHOT.with(|value| {
                value.replace(self.0.take());
            });
        }
    }
    let _restore = Restore(TEST_SNAPSHOT.with(|value| value.replace(Some(sample))));
    run()
}

/// `None` means this binary/platform has no Metal backend, not a failed probe.
pub fn snapshot(ordinal: usize) -> Option<MetalMemorySnapshot> {
    #[cfg(test)]
    if let Some(value) = TEST_SNAPSHOT.with(|value| value.borrow().clone()) {
        return Some(value);
    }
    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        snapshot_with_host(
            ordinal,
            crate::device::total_system_memory_bytes(),
            crate::device::available_system_memory_bytes(),
        )
    }
    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    {
        let _ = ordinal;
        None
    }
}

/// Observe Metal against an existing authoritative host sample. Server resource
/// collection uses this so the host and device budgets share one Mach reading.
pub fn snapshot_with_host(
    ordinal: usize,
    physical_bytes: Option<u64>,
    available_host_bytes: Option<u64>,
) -> Option<MetalMemorySnapshot> {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        let wired = read_wired_limit().map_err(|error| error.to_string());
        let metal = crate::device::metal_device(ordinal)
            .and_then(|device| {
                let native = device.as_metal_device()?.device();
                Ok((
                    native.recommended_max_working_set_size() as u64,
                    native.current_allocated_size() as u64,
                ))
            })
            .map_err(|error| error.to_string());
        Some(from_readings(
            wired,
            metal,
            physical_bytes,
            available_host_bytes,
        ))
    }
    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    {
        let _ = (ordinal, physical_bytes, available_host_bytes);
        None
    }
}

/// Pure observation adapter shared by the native probe and injected tests.
#[cfg(any(test, all(target_os = "macos", feature = "metal")))]
fn from_readings(
    wired: Result<Option<u32>, String>,
    metal: Result<(u64, u64), String>,
    physical: Option<u64>,
    available: Option<u64>,
) -> MetalMemorySnapshot {
    use mold_core::metal_memory::MetalWiredLimit;
    let mut errors = Vec::new();
    let wired_limit = match wired {
        Ok(Some(0)) => MetalWiredLimit::Automatic,
        Ok(Some(mib)) => MetalWiredLimit::Explicit { mib },
        Ok(None) => MetalWiredLimit::Unsupported,
        Err(error) => {
            errors.push(format!("Cannot read Metal wired limit: {error}"));
            MetalWiredLimit::Unavailable
        }
    };
    let (recommended_bytes, allocated_bytes) = match metal {
        Ok((recommended, allocated)) => {
            if recommended == 0 {
                errors.push("Metal reported an unavailable working-set recommendation".into());
            }
            ((recommended > 0).then_some(recommended), Some(allocated))
        }
        Err(error) => {
            errors.push(format!("Cannot open Metal device: {error}"));
            (None, None)
        }
    };
    if physical.is_none() || available.is_none() {
        errors.push("Cannot read macOS host-memory budget".into());
    }
    MetalMemorySnapshot {
        wired_limit,
        physical_bytes: physical,
        available_host_bytes: available,
        recommended_bytes,
        allocated_bytes,
        effective_capacity_bytes: None,
        allocation_headroom_bytes: None,
        error: (!errors.is_empty()).then(|| errors.join("; ")),
    }
    .resolve()
}

#[cfg(test)]
mod observation_tests {
    use super::*;
    #[test]
    fn metal_memory_probe_preserves_all_failure_causes() {
        let sample = from_readings(
            Err("permission denied".into()),
            Err("no devices".into()),
            None,
            None,
        );
        let error = sample.error.unwrap();
        assert!(error.contains("permission denied"));
        assert!(error.contains("no devices"));
        assert!(error.contains("host-memory"));
        assert_eq!(sample.effective_capacity_bytes, None);
    }
    #[test]
    fn metal_memory_probe_distinguishes_optional_key_and_failed_probe() {
        let sample = from_readings(
            Ok(None),
            Ok((37 << 30, 4 << 30)),
            Some(48 << 30),
            Some(32 << 30),
        );
        assert_eq!(sample.effective_capacity_bytes, Some(37 << 30));
        assert!(sample.error.is_none());
        let missing = from_readings(
            Ok(Some(0)),
            Err("no Metal devices".into()),
            Some(48 << 30),
            Some(32 << 30),
        );
        assert!(missing.error.is_some());
        assert_eq!(missing.allocation_headroom_bytes, None);
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
    #[ignore = "native read-only qualification; not part of GPU-free unit tests"]
    fn metal_memory_native_snapshot_is_read_only_and_policy_bounded() {
        let before = super::read_wired_limit().expect("read native sysctl");
        let sample = super::snapshot(0).expect("Metal build");
        if let Some(cap) = sample.effective_capacity_bytes {
            assert!(cap <= sample.recommended_bytes.unwrap());
            assert!(cap <= sample.physical_bytes.unwrap());
            if let Some(headroom) = sample.allocation_headroom_bytes {
                assert!(headroom <= cap);
            }
        } else {
            assert!(sample.error.is_some());
        }
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
