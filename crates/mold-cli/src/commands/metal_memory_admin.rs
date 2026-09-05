//! Testable administration protocol. No process privileges or filesystem I/O.

pub trait WiredLimitAccess {
    fn read(&mut self) -> Result<u32, String>;
    fn write(&mut self, value: u32) -> Result<(), String>;
}

pub fn validate_limit(value: u32, maximum: u32) -> Result<(), String> {
    if value == 0 {
        return Err("use reset to select the automatic system limit".into());
    }
    if value > maximum {
        return Err(format!("{value} MiB exceeds the safe maximum {maximum} MiB (preserves max(15% RAM, 8 GiB) for the host)"));
    }
    Ok(())
}

/// Returns the previous raw setting only after the kernel confirms the write.
pub fn set_verified(access: &mut impl WiredLimitAccess, value: u32) -> Result<u32, String> {
    let old = access.read()?;
    access.write(value)?;
    let observed = access.read()?;
    if observed != value {
        return Err(format!("wired-limit verification failed: requested {value} MiB, observed {observed} MiB, previous {old} MiB; no success or rollback assumed"));
    }
    Ok(old)
}

/// Do not overwrite a change made by another administrator during rollback.
pub fn restore_if_unchanged(
    access: &mut impl WiredLimitAccess,
    applied: u32,
    previous: u32,
) -> Result<(), String> {
    let current = access.read()?;
    if current != applied {
        return Err(format!("rollback skipped: current setting changed to {current} MiB (our value was {applied} MiB)"));
    }
    set_verified(access, previous).map(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Fake {
        value: u32,
        ignore_write: bool,
        fail_write: bool,
    }
    impl WiredLimitAccess for Fake {
        fn read(&mut self) -> Result<u32, String> {
            Ok(self.value)
        }
        fn write(&mut self, value: u32) -> Result<(), String> {
            if self.fail_write {
                return Err("permission denied".into());
            }
            if !self.ignore_write {
                self.value = value
            }
            Ok(())
        }
    }
    #[test]
    fn metal_memory_admin_rejects_zero_and_unsafe_limits() {
        assert!(validate_limit(0, 16384).is_err());
        assert!(validate_limit(16385, 16384).is_err());
        assert!(validate_limit(16384, 16384).is_ok());
    }
    #[test]
    fn metal_memory_admin_verifies_kernel_readback() {
        let mut access = Fake {
            value: 0,
            ignore_write: true,
            fail_write: false,
        };
        assert!(set_verified(&mut access, 16384).is_err());
    }
    #[test]
    fn metal_memory_admin_returns_previous_setting_and_resets() {
        let mut access = Fake {
            value: 12288,
            ignore_write: false,
            fail_write: false,
        };
        assert_eq!(set_verified(&mut access, 16384).unwrap(), 12288);
        assert_eq!(set_verified(&mut access, 0).unwrap(), 16384);
        assert_eq!(access.value, 0);
    }
    #[test]
    fn metal_memory_admin_surfaces_permission_failure() {
        let mut access = Fake {
            value: 0,
            ignore_write: false,
            fail_write: true,
        };
        assert_eq!(
            set_verified(&mut access, 16384).unwrap_err(),
            "permission denied"
        );
        assert_eq!(access.value, 0);
    }
    #[test]
    fn metal_memory_admin_rollback_preserves_other_administrators_changes() {
        let mut access = Fake {
            value: 12288,
            ignore_write: false,
            fail_write: false,
        };
        assert!(restore_if_unchanged(&mut access, 16384, 0).is_err());
        assert_eq!(access.value, 12288);
        assert!(restore_if_unchanged(&mut access, 12288, 0).is_ok());
        assert_eq!(access.value, 0);
    }
}
