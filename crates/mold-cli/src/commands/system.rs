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
    /// Inspect or downgrade this machine's gallery archive authority storage
    GalleryAuthority {
        #[command(subcommand)]
        action: GalleryAuthorityAction,
    },
}

#[derive(Subcommand)]
pub enum GalleryAuthorityAction {
    /// Show the store's on-disk format and whether a downgrade is needed
    Status {
        /// Gallery directory; defaults to this machine's configured output dir
        #[arg(long, value_name = "PATH")]
        output_dir: Option<std::path::PathBuf>,
        #[arg(long)]
        json: bool,
    },
    /// Rewrite a version-3 store as version 2 so an older mold can publish
    ///
    /// Storage version 3 is opt-in (`gallery.authority_log`) because a mold
    /// older than 0.29 reads only version 2 and refuses to publish against a
    /// v3 store. Run this with the NEWER build, before rolling one back or
    /// before starting an older binary against a shared $MOLD_HOME.
    ///
    /// It refuses while any mold process holds this gallery's writer lease —
    /// stop `mold serve` (and any local `mold run` or desktop app) first.
    /// `status` reports whether one is live.
    Downgrade {
        /// Gallery directory; defaults to this machine's configured output dir
        #[arg(long, value_name = "PATH")]
        output_dir: Option<std::path::PathBuf>,
        #[arg(long)]
        json: bool,
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
    match action {
        SystemAction::MetalMemory { action } => match action {
            MetalMemoryAction::Status { json } => status(*json),
            MetalMemoryAction::Set { mib, persist } => change(*mib, *persist),
            MetalMemoryAction::Reset { persist } => change(0, *persist),
        },
        SystemAction::GalleryAuthority { action } => match action {
            GalleryAuthorityAction::Status { output_dir, json } => {
                gallery_authority_status(output_dir.as_deref(), *json)
            }
            GalleryAuthorityAction::Downgrade { output_dir, json } => {
                gallery_authority_downgrade(output_dir.as_deref(), *json)
            }
        },
    }
}

/// Resolve the gallery this machine publishes to.
///
/// `mold system` routes before config/DB startup and always targets THIS
/// machine — the store is a directory, not a server — so the config is loaded
/// here rather than inherited.
fn resolve_output_dir(explicit: Option<&std::path::Path>) -> Result<std::path::PathBuf> {
    if let Some(path) = explicit {
        return Ok(path.to_path_buf());
    }
    let config = mold_core::Config::load_or_default();
    let dir = config.effective_output_dir();
    anyhow::ensure!(
        !dir.as_os_str().is_empty(),
        "this machine has no gallery output directory configured; pass --output-dir"
    );
    Ok(dir)
}

fn gallery_authority_status(output_dir: Option<&std::path::Path>, json: bool) -> Result<()> {
    let dir = resolve_output_dir(output_dir)?;
    let status = mold_server::gallery_authority::storage_status(&dir)?;
    if json {
        println!("{}", serde_json::to_string_pretty(&status)?);
        return Ok(());
    }
    if !status.present {
        println!("No gallery archive authority in {}", dir.display());
        return Ok(());
    }
    println!("Gallery archive authority in {}", dir.display());
    println!(
        "  storage version: checkpoint {}, marker {}",
        status
            .checkpoint_version
            .map(|v| v.to_string())
            .unwrap_or_else(|| "unreadable".into()),
        status
            .marker_version
            .map(|v| v.to_string())
            .unwrap_or_else(|| "absent".into()),
    );
    println!(
        "  generation: {}",
        status
            .generation
            .map(|g| g.to_string())
            .unwrap_or_else(|| "unknown".into())
    );
    println!(
        "  mutation log: {} record(s), {} bytes",
        status.log_records, status.log_bytes
    );
    // The one line that says whether a downgrade will be allowed at all. The
    // three format facts above describe the store; this describes the machine.
    if status.live_writer {
        println!(
            "  live writer: yes{} — `downgrade` will refuse until it stops",
            match (status.live_writer_pid, status.live_writer_since_ms) {
                (Some(pid), Some(since)) if pid != 0 =>
                    format!(" (pid {pid}, lease taken at epoch ms {since})"),
                (Some(pid), None) if pid != 0 => format!(" (pid {pid})"),
                _ => String::new(),
            }
        );
    } else {
        println!("  live writer: none");
    }
    if status.pending_mutation {
        println!("  pending mutation: yes — start `mold serve` once to resolve it");
    }
    if status.torn_log_tail {
        println!("  torn log tail: yes — start `mold serve` once to resolve it");
    }
    // A home where both formats were written carries TWO indexes, and which
    // one is ahead decides whether a downgrade is safe at all. Reporting only
    // the active store hid exactly the divergence the switch exists to manage.
    for (label, facts) in [
        ("version-2 store", status.legacy_store),
        ("version-3 store", status.log_store),
    ] {
        let Some(facts) = facts else { continue };
        println!(
            "  other {label}: version {}, generation {}",
            facts
                .checkpoint_version
                .map(|v| v.to_string())
                .unwrap_or_else(|| "unreadable".into()),
            facts
                .generation
                .map(|g| g.to_string())
                .unwrap_or_else(|| "unknown".into()),
        );
    }
    if let (Some(legacy), Some(active)) = (
        status.legacy_store.and_then(|facts| facts.generation),
        status.generation,
    ) {
        if legacy > active {
            println!(
                "  the version-2 store is AHEAD (generation {legacy} against {active}) — an \
                 older binary has published here since the upgrade. `downgrade` will refuse \
                 rather than discard those prints."
            );
        }
    }
    if status.marker_version == Some(3) || status.checkpoint_version == Some(3) {
        println!(
            "  a mold older than 0.29 cannot publish against this store; \
             run `mold system gallery-authority downgrade` before rolling one back"
        );
    }
    Ok(())
}

fn gallery_authority_downgrade(output_dir: Option<&std::path::Path>, json: bool) -> Result<()> {
    let dir = resolve_output_dir(output_dir)?;
    let outcome = mold_server::gallery_authority::downgrade_to_legacy_storage(&dir)?;
    if json {
        println!("{}", serde_json::to_string_pretty(&outcome)?);
        return Ok(());
    }
    if outcome.already_legacy {
        println!(
            "{} already holds a version-2 gallery archive authority at generation {}",
            dir.display(),
            outcome.generation
        );
        return Ok(());
    }
    println!(
        "Downgraded the gallery archive authority in {} to storage version 2 at generation {} \
         ({} delta record(s) folded in). Verified by reading it back.",
        dir.display(),
        outcome.generation,
        outcome.replayed_records
    );
    Ok(())
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
