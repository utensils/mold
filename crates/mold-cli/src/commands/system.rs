//! Explicit local machine administration; called before Config/DB initialization.

use anyhow::{bail, Result};
use clap::Subcommand;

#[derive(Subcommand)]
pub enum SystemAction {
    /// Inspect or configure this Mac's machine-wide GPU memory limit
    MetalMemory {
        #[command(subcommand)]
        action: MetalMemoryAction,
    },
}

#[derive(Subcommand)]
pub enum MetalMemoryAction {
    /// Show this machine's kernel setting and effective Metal budget (no server request)
    Status {
        #[arg(long)]
        json: bool,
    },
    /// Set the local machine-wide limit in MiB; requires root, preserves host headroom
    Set {
        #[arg(value_name = "MiB", value_parser = clap::value_parser!(u32).range(1..))]
        mib: u32,
        /// Also install a root-owned boot policy, applied at subsequent boots
        #[arg(long)]
        persist: bool,
    },
    /// Restore the automatic system limit; requires root
    Reset {
        /// Also remove Mold's owned boot policy and any loaded registration
        #[arg(long)]
        persist: bool,
    },
}

pub fn run(action: &SystemAction) -> Result<()> {
    let SystemAction::MetalMemory { action } = action;
    match action {
        MetalMemoryAction::Status { json } => status(*json),
        MetalMemoryAction::Set { mib, persist } => change(*mib, *persist),
        MetalMemoryAction::Reset { persist } => change(0, *persist),
    }
}

fn status(json: bool) -> Result<()> {
    let memory = mold_inference::metal_memory::snapshot(0);
    #[cfg(target_os = "macos")]
    let (raw, error) = match mold_inference::metal_memory::read_wired_limit() {
        Ok(value) => (value, None),
        Err(error) => (None, Some(error.to_string())),
    };
    #[cfg(not(target_os = "macos"))]
    let (raw, error): (Option<u32>, Option<String>) = (None, Some("macOS only".into()));
    #[cfg(target_os = "macos")]
    let (persistent, persistence_error) = {
        use super::metal_memory_persistence::{read_policy, DIRECTORY};
        match read_policy(std::path::Path::new(DIRECTORY), 0) {
            Ok(value) => (value, None),
            Err(error) => (None, Some(error.to_string())),
        }
    };
    #[cfg(not(target_os = "macos"))]
    let (persistent, persistence_error): (Option<u32>, Option<String>) = (None, None);
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "scope": "local_machine", "supported": if !cfg!(target_os = "macos") { Some(false) } else if error.is_some() { None } else { Some(raw.is_some()) },
                "wired_limit_mib": raw, "error": error,
                "persistent_limit_mib": persistent, "persistence_error": persistence_error,
                "memory": memory,
            }))?
        );
    } else {
        println!("Metal memory — this machine (system-wide)");
        match raw {
            Some(0) => println!("Kernel setting: automatic"),
            Some(mib) => println!("Kernel setting: {mib} MiB"),
            None if error.is_none() => println!("Kernel setting: unsupported"),
            None => println!(
                "Kernel setting: unavailable{}",
                error
                    .as_deref()
                    .map(|e| format!(" ({e})"))
                    .unwrap_or_default()
            ),
        }
        if let Some(memory) = memory {
            for (label, value) in [
                ("Installed RAM", memory.physical_bytes),
                ("Metal recommendation", memory.recommended_bytes),
                (
                    "Mold allocated (including cached buffers)",
                    memory.allocated_bytes,
                ),
                ("Effective capacity", memory.effective_capacity_bytes),
                ("Allocation headroom", memory.allocation_headroom_bytes),
            ] {
                println!(
                    "{label}: {}",
                    value
                        .map(|bytes| format!("{:.2} GiB", bytes as f64 / (1_u64 << 30) as f64))
                        .unwrap_or_else(|| "unavailable".into())
                );
            }
            if let Some(error) = memory.error {
                println!("Probe: {error}");
            }
            println!("Allocated/headroom values belong to this inspection process; use `mold gpu list --json` for a running server.");
        } else {
            println!("Metal budget unavailable in this build/platform.");
        }
        println!(
            "Boot policy: {}",
            persistent
                .map(|mib| format!("{mib} MiB"))
                .unwrap_or_else(|| if persistence_error.is_some() {
                    "unavailable"
                } else {
                    "none"
                }
                .into())
        );
        if let Some(error) = persistence_error {
            println!("Boot policy inspection: {error}");
        }
    }
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn change(_value: u32, _persist: bool) -> Result<()> {
    bail!("Metal wired-limit administration is available only on macOS")
}

#[cfg(target_os = "macos")]
struct Kernel;

#[cfg(target_os = "macos")]
impl super::metal_memory_admin::WiredLimitAccess for Kernel {
    fn read(&mut self) -> Result<u32, String> {
        mold_inference::metal_memory::read_wired_limit()
            .map_err(|e| e.to_string())?
            .ok_or_else(|| "iogpu.wired_limit_mb is unsupported on this Mac".into())
    }
    fn write(&mut self, mut value: u32) -> Result<(), String> {
        // SAFETY: fixed kernel key; exact uint ABI and valid input storage.
        let result = unsafe {
            libc::sysctlbyname(
                c"iogpu.wired_limit_mb".as_ptr(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                (&raw mut value).cast(),
                std::mem::size_of::<u32>(),
            )
        };
        if result != 0 {
            return Err(std::io::Error::last_os_error().to_string());
        }
        Ok(())
    }
}

#[cfg(target_os = "macos")]
fn change(value: u32, persist: bool) -> Result<()> {
    use super::metal_memory_admin::{
        apply_verified, require_root, validate_limit, WiredLimitAccess,
    };
    use super::metal_memory_persistence::{Store, DIRECTORY};
    use anyhow::Context;
    // SAFETY: geteuid has no pointer arguments or side effects.
    require_root(unsafe { libc::geteuid() }).map_err(anyhow::Error::msg)?;
    let mut kernel = Kernel;
    kernel.read().map_err(anyhow::Error::msg)?;
    if value != 0 {
        let total = mold_inference::device::total_system_memory_bytes()
            .context("cannot read installed RAM")?;
        let maximum = (total.saturating_sub(mold_core::metal_memory::host_safety_floor(total))
            / mold_core::metal_memory::MIB)
            .min(u64::from(u32::MAX)) as u32;
        validate_limit(value, maximum).map_err(anyhow::Error::msg)?;
    }
    // Also serializes non-persistent changes. No user/config/environment path.
    let store = Store::open(std::path::Path::new(DIRECTORY), 0)?;
    let result = apply_verified(&mut kernel, &mut BootPolicy(store), value, persist)
        .map_err(anyhow::Error::msg)?;
    let previous = result.previous;
    let previous_policy = result.previous_policy;
    if let Some(warning) = result.policy_warning {
        eprintln!("Boot policy left untouched; inspection failed: {warning}");
    }
    println!(
        "Verified local iogpu.wired_limit_mb: {previous} → {value}{}",
        if value == 0 { " (automatic)" } else { " MiB" }
    );
    if persist {
        println!(
            "Boot policy: {}",
            if value == 0 {
                if previous_policy.is_some() {
                    "removed"
                } else {
                    "none (already absent)"
                }
            } else {
                "installed for subsequent boots"
            }
        );
    } else if let Some(mib) = previous_policy {
        println!("Existing boot policy remains {mib} MiB; use --persist to change or remove it.");
    }
    status(false)?;
    println!("Restart an idle inference process if its Metal recommendation has not refreshed. This does not reserve memory against other applications.");
    Ok(())
}

#[cfg(target_os = "macos")]
struct BootPolicy(super::metal_memory_persistence::Store);

#[cfg(target_os = "macos")]
impl super::metal_memory_admin::BootPolicyAccess for BootPolicy {
    fn read(&mut self) -> Result<Option<u32>, String> {
        self.0.read().map_err(|error| error.to_string())
    }
    fn replace(&mut self, value: Option<u32>, expected: Option<u32>) -> Result<(), String> {
        self.0
            .replace(value, expected)
            .map_err(|error| error.to_string())
    }
    fn unregister(&mut self, owned_file: bool) -> Result<bool, String> {
        unregister_owned_boot_policy(owned_file).map_err(|error| error.to_string())
    }
}

#[cfg(target_os = "macos")]
fn unregister_owned_boot_policy(owned_file: bool) -> Result<bool> {
    use super::metal_memory_persistence::LABEL;
    let domain = format!("system/{LABEL}");
    let state = std::process::Command::new("/bin/launchctl")
        .args(["print", &domain])
        .output()?;
    if !state.status.success() {
        let stderr = String::from_utf8_lossy(&state.stderr);
        if state.status.code() == Some(113)
            && stderr.contains(&format!("Could not find service \"{LABEL}\""))
        {
            return Ok(false);
        }
        bail!(
            "cannot inspect Mold boot-policy registration: {}",
            stderr.trim()
        );
    }
    if !owned_file {
        bail!("a boot-policy registration exists without Mold's owned file; refusing to remove an unverified service. Inspect it with /bin/launchctl print system/io.utensils.mold.metal-memory; an administrator can remove a verified stale registration with /bin/launchctl bootout system/io.utensils.mold.metal-memory")
    }
    let output = std::process::Command::new("/bin/launchctl")
        .args(["bootout", &domain])
        .output()?;
    if !output.status.success() {
        bail!(
            "cannot unload Mold's boot-policy registration: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    #[derive(Parser)]
    struct Args {
        #[command(subcommand)]
        action: SystemAction,
    }
    #[test]
    fn metal_memory_cli_rejects_zero_negative_overflow_and_remote_arguments() {
        for value in ["0", "-1", "4294967296", "16GB"] {
            assert!(Args::try_parse_from(["mold", "metal-memory", "set", value]).is_err());
        }
        assert!(
            Args::try_parse_from(["mold", "metal-memory", "set", "16384", "--host", "remote"])
                .is_err()
        );
        assert!(
            Args::try_parse_from(["mold", "metal-memory", "set", "16384", "--persist"]).is_ok()
        );
        assert!(Args::try_parse_from(["mold", "metal-memory", "reset", "--persist"]).is_ok());
    }
}
