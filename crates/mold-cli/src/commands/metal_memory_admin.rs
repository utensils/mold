//! Testable administration protocol. No process privileges or filesystem I/O.

pub trait WiredLimitAccess {
    fn read(&mut self) -> Result<u32, String>;
    fn write(&mut self, value: u32) -> Result<(), String>;
}

pub trait BootPolicyAccess {
    fn read(&mut self) -> Result<Option<u32>, String>;
    fn replace(&mut self, value: Option<u32>, expected: Option<u32>) -> Result<(), String>;
    /// True when an existing registration was actually removed.
    fn unregister(&mut self, owned_file: bool) -> Result<bool, String>;
}

pub struct ChangeOutcome {
    pub previous: u32,
    pub previous_policy: Option<u32>,
    pub policy_warning: Option<String>,
}

/// Keep the privileged orchestration injectable; neither tests nor this
/// protocol know filesystem paths, process privileges or launchctl commands.
pub fn apply_verified(
    kernel: &mut impl WiredLimitAccess,
    policy: &mut impl BootPolicyAccess,
    value: u32,
    persist: bool,
) -> Result<ChangeOutcome, String> {
    let (previous_policy, policy_warning) = match policy.read() {
        Ok(value) => (value, None),
        Err(error) if !persist => (None, Some(error)),
        Err(error) => return Err(error),
    };
    let unloaded = persist && policy.unregister(previous_policy.is_some())?;
    let registration_note = if unloaded {
        " The previous boot registration was unloaded; its file still controls the next boot."
    } else {
        ""
    };
    let previous = set_verified(kernel, value)
        .map_err(|error| format!("{error}.{registration_note} Inspect status before retrying."))?;
    let target = (value != 0).then_some(value);
    if persist {
        if let Err(error) = policy.replace(target, previous_policy) {
            let policy_restore = match policy.read() {
                Ok(current) if current == previous_policy => Ok(()),
                Ok(current) if current == target => policy.replace(previous_policy, current),
                _ => Err("boot policy changed externally; left untouched".into()),
            };
            let kernel_restore = restore_if_unchanged(kernel, value, previous);
            let describe = |result: Result<(), String>| match result {
                Ok(()) => "previous state restored".into(),
                Err(error) => error,
            };
            return Err(format!("boot-policy update failed: {error}; boot-policy rollback: {}; kernel rollback: {}.{registration_note} Inspect status before retrying.", describe(policy_restore), describe(kernel_restore)));
        }
    }
    Ok(ChangeOutcome {
        previous,
        previous_policy,
        policy_warning,
    })
}

pub fn require_root(euid: u32) -> Result<(), String> {
    if euid == 0 {
        Ok(())
    } else {
        Err(
            "changing this machine-wide setting requires root; run this explicit command with sudo"
                .into(),
        )
    }
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
        assert!(require_root(501).is_err());
        assert!(require_root(0).is_ok());
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
    struct Boot {
        value: Option<u32>,
        foreign: bool,
        fail_replace: bool,
        fail_after_replace: bool,
        fail_unregister: bool,
        unloaded: bool,
        replacements: usize,
    }
    impl BootPolicyAccess for Boot {
        fn read(&mut self) -> Result<Option<u32>, String> {
            if self.foreign {
                Err("foreign policy".into())
            } else {
                Ok(self.value)
            }
        }
        fn replace(&mut self, value: Option<u32>, expected: Option<u32>) -> Result<(), String> {
            self.replacements += 1;
            assert_eq!(self.value, expected);
            if self.fail_replace {
                return Err("disk full".into());
            }
            self.value = value;
            if self.fail_after_replace {
                self.fail_after_replace = false;
                return Err("directory sync failed after replacement".into());
            }
            Ok(())
        }
        fn unregister(&mut self, _: bool) -> Result<bool, String> {
            if self.fail_unregister {
                return Err("bootout denied".into());
            }
            self.unloaded = true;
            Ok(true)
        }
    }
    fn boot() -> Boot {
        Boot {
            value: Some(12288),
            foreign: false,
            fail_replace: false,
            fail_after_replace: false,
            fail_unregister: false,
            unloaded: false,
            replacements: 0,
        }
    }
    fn kernel() -> Fake {
        Fake {
            value: 12288,
            ignore_write: false,
            fail_write: false,
        }
    }
    #[test]
    fn metal_memory_admin_persist_failure_restores_live_value() {
        let (mut k, mut b) = (kernel(), boot());
        b.fail_replace = true;
        let error = apply_verified(&mut k, &mut b, 16384, true).err().unwrap();
        assert_eq!(k.value, 12288);
        assert_eq!(b.value, Some(12288));
        assert!(error.contains("kernel rollback: previous state restored"));
        assert!(error.contains("registration was unloaded"));
    }
    #[test]
    fn metal_memory_admin_kernel_failure_preserves_file_and_reports_bootout() {
        let (mut k, mut b) = (kernel(), boot());
        k.fail_write = true;
        let error = apply_verified(&mut k, &mut b, 16384, true).err().unwrap();
        assert_eq!(b.replacements, 0);
        assert_eq!(b.value, Some(12288));
        assert!(error.contains("registration was unloaded"));
    }
    #[test]
    fn metal_memory_admin_bootout_failure_precedes_any_writes() {
        let (mut k, mut b) = (kernel(), boot());
        b.fail_unregister = true;
        assert!(apply_verified(&mut k, &mut b, 16384, true).is_err());
        assert_eq!(k.value, 12288);
        assert_eq!(b.replacements, 0);
    }
    #[test]
    fn metal_memory_admin_nonpersistent_foreign_policy_is_only_a_warning() {
        let (mut k, mut b) = (kernel(), boot());
        b.foreign = true;
        let outcome = apply_verified(&mut k, &mut b, 16384, false).unwrap();
        assert_eq!(k.value, 16384);
        assert!(outcome.policy_warning.is_some());
        assert!(!b.unloaded);
        assert_eq!(b.replacements, 0);
        assert!(apply_verified(&mut k, &mut b, 8192, true).is_err());
        assert_eq!(k.value, 16384);
    }
    #[test]
    fn metal_memory_admin_reset_removes_owned_boot_policy() {
        let (mut k, mut b) = (kernel(), boot());
        let outcome = apply_verified(&mut k, &mut b, 0, true).unwrap();
        assert_eq!(outcome.previous, 12288);
        assert_eq!(outcome.previous_policy, Some(12288));
        assert_eq!(k.value, 0);
        assert_eq!(b.value, None);
    }
    #[test]
    fn metal_memory_admin_partial_file_update_restores_both_verified_states() {
        let (mut k, mut b) = (kernel(), boot());
        b.fail_after_replace = true;
        let error = apply_verified(&mut k, &mut b, 16384, true).err().unwrap();
        assert_eq!(k.value, 12288);
        assert_eq!(b.value, Some(12288));
        assert_eq!(b.replacements, 2);
        assert!(error.contains("boot-policy rollback: previous state restored"));
    }
}
