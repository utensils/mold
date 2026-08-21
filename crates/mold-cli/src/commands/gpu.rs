use std::collections::BTreeMap;

use anyhow::{bail, Context, Result};
use clap_complete::engine::CompletionCandidate;
use colored::Colorize;
use mold_core::{
    classify_server_error, DeviceActivity, DeviceAdminState, DeviceHealth, DeviceInfo,
    DeviceMemoryInfo, DeviceState, DeviceTelemetry, MoldClient, ServerAvailability,
    ServerCapabilities,
};

use crate::control::is_loopback_host;

fn ensure_supported(
    capabilities: &ServerCapabilities,
    device: &DeviceInfo,
    enabled: bool,
) -> Result<()> {
    if capabilities.devices.lifecycle && capabilities.dispatch.v2_authoritative {
        return Ok(());
    }
    if enabled && !device.desired_enabled && capabilities.devices.restart_enable {
        return Ok(());
    }
    bail!(
        "live GPU controls require Scheduler V2; only a persistently-disabled GPU can be enabled for the next server restart"
    )
}

fn resolve_device<'a>(devices: &'a [DeviceInfo], selector: &str) -> Result<&'a DeviceInfo> {
    if let Some(device) = devices.iter().find(|device| device.id == selector) {
        return Ok(device);
    }
    if let Ok(ordinal) = selector.parse::<usize>() {
        if let Some(device) = devices
            .iter()
            .find(|device| device.ordinal == Some(ordinal))
        {
            return Ok(device);
        }
    }
    bail!("unknown GPU {selector:?}; run `mold gpu list` for stable IDs")
}

fn human_state(device: &DeviceInfo) -> String {
    match (device.admin_state, device.health) {
        (DeviceAdminState::Draining, _) => "finishing current work".yellow().to_string(),
        (_, DeviceHealth::Unavailable | DeviceHealth::Poisoned) => {
            device.health.as_str().red().to_string()
        }
        (DeviceAdminState::Enabled, DeviceHealth::Healthy) => "enabled".green().to_string(),
        _ => device.admin_state.as_str().to_string(),
    }
}

pub(crate) fn format_device_line(device: &DeviceInfo) -> String {
    let ordinal = device
        .ordinal
        .map(|value| format!("GPU {value}"))
        .unwrap_or_else(|| "GPU —".into());
    let kind = if device.device_kind == mold_core::DeviceKind::Mig {
        format!(
            "MIG {}",
            device.mig_profile.as_deref().unwrap_or("profile unknown")
        )
    } else {
        device.backend.as_str().to_ascii_uppercase()
    };
    let vram = match (device.memory.used_bytes, device.memory.total_bytes) {
        (Some(used), Some(total)) => format!(
            "{:.1}/{:.1} GiB",
            used as f64 / 1024_f64.powi(3),
            total as f64 / 1024_f64.powi(3)
        ),
        _ => "—".into(),
    };
    let utilization = device
        .telemetry
        .utilization_percent
        .map(|value| format!("{value}%"))
        .unwrap_or_else(|| "—".into());
    format!(
        "{}  {}  {}  {}  {}  VRAM {}  util {}",
        device.id,
        ordinal,
        device.name,
        kind,
        human_state(device),
        vram,
        utilization
    )
}

pub async fn list(json: bool) -> Result<()> {
    let client = MoldClient::from_env();
    let (state, local) = match client.devices().await {
        Ok(state) => (state, false),
        Err(error)
            if is_loopback_host(client.host())
                && classify_server_error(&error) == ServerAvailability::FallbackLocal =>
        {
            (local_device_state()?, true)
        }
        Err(error) => return Err(error).context("failed to read server devices"),
    };
    if json {
        if local {
            eprintln!(
                "server unavailable at {}; returning local runtime inventory",
                client.host()
            );
        }
        println!("{}", serde_json::to_string_pretty(&state)?);
        return Ok(());
    }
    if local {
        println!("No mold server is running; showing this machine's runtime-visible devices.");
        println!("States below are startup preferences; live activity requires `mold serve`.\n");
    }
    if state.devices.is_empty() {
        println!("No runtime-visible compute devices on this machine.");
        if mold_inference::compiled_backend_label() == "cpu" {
            println!("This mold binary was built without a GPU backend.");
        }
        return Ok(());
    }
    for device in &state.devices {
        println!("{}", format_device_line(device));
    }
    Ok(())
}

pub async fn set(selector: &str, enabled: bool) -> Result<()> {
    let client = MoldClient::from_env();
    match set_enabled_with_client(&client, selector, enabled).await {
        Ok(()) => Ok(()),
        Err(error)
            if is_loopback_host(client.host())
                && classify_server_error(&error) == ServerAvailability::FallbackLocal =>
        {
            set_local_preference(selector, enabled)
        }
        Err(error) => Err(error),
    }
}

/// Dynamic completion candidates for local stable device IDs.
///
/// Completion deliberately performs no network request: it remains useful
/// with the server stopped and never substitutes local IDs for a remote host.
pub fn complete_device_id() -> Vec<CompletionCandidate> {
    complete_device_id_for_host(MoldClient::from_env().host())
}

fn complete_device_id_for_host(host: &str) -> Vec<CompletionCandidate> {
    if !is_loopback_host(host) {
        return Vec::new();
    }
    let mut ids = mold_inference::device::discover_gpus()
        .into_iter()
        .filter_map(|device| device.stable_id)
        .collect::<Vec<_>>();
    ids.sort();
    ids.into_iter().map(CompletionCandidate::new).collect()
}

pub(crate) fn local_device_state() -> Result<DeviceState> {
    let preferences = crate::metadata_db::handle()
        .map(|db| mold_db::DevicePreferences::new(db).list())
        .transpose()?
        .unwrap_or_default();
    Ok(project_local_devices(
        mold_inference::device::discover_gpus(),
        &preferences,
    ))
}

fn project_local_devices(
    discovered: Vec<mold_inference::device::DiscoveredGpu>,
    preferences: &BTreeMap<String, mold_db::DevicePreference>,
) -> DeviceState {
    let devices = discovered
        .into_iter()
        .map(|device| {
            let stable = device.stable_id.is_some();
            let id = device.stable_id.unwrap_or_else(|| {
                format!("{}:unavailable-{}", device.backend.as_str(), device.ordinal)
            });
            let desired_enabled = preferences
                .get(&id)
                .map(|preference| preference.desired_enabled)
                .unwrap_or(true);
            let total = (device.total_vram_bytes > 0).then_some(device.total_vram_bytes);
            let used = total.map(|total| total.saturating_sub(device.free_vram_bytes));
            let device_kind = match device.device_kind {
                Some(mold_inference::device::CudaDeviceKind::FullGpu) => {
                    mold_core::DeviceKind::FullGpu
                }
                Some(mold_inference::device::CudaDeviceKind::Mig) => mold_core::DeviceKind::Mig,
                Some(mold_inference::device::CudaDeviceKind::UnknownCuda) => {
                    mold_core::DeviceKind::UnknownCuda
                }
                None => mold_core::DeviceKind::Metal,
            };
            DeviceInfo {
                id,
                backend: device.backend,
                ordinal: Some(device.ordinal),
                device_kind,
                nvml_uuid: None,
                physical_uuid: None,
                mig_uuid: None,
                mig_parent_uuid: None,
                mig_profile: None,
                name: device.name,
                pci_bus_id: device.pci_bus_id,
                compute_capability: device
                    .compute_capability
                    .map(|(major, minor)| format!("{major}.{minor}")),
                memory: DeviceMemoryInfo {
                    total_bytes: total,
                    used_bytes: used,
                    mold_used_bytes: None,
                    other_used_bytes: None,
                },
                telemetry: DeviceTelemetry {
                    utilization_percent: None,
                    temperature_c: None,
                    power_w: None,
                },
                desired_enabled,
                restart_required: false,
                admin_state: if desired_enabled {
                    DeviceAdminState::Enabled
                } else {
                    DeviceAdminState::Disabled
                },
                health: if stable {
                    DeviceHealth::Healthy
                } else {
                    DeviceHealth::Unavailable
                },
                activity: DeviceActivity::Idle,
                schedulable: false,
                unschedulable_reason: Some(
                    device
                        .identity_error
                        .unwrap_or_else(|| "mold server is not running".into()),
                ),
                loaded_models: vec![],
                active_work_id: None,
                planned_work_ids: vec![],
            }
        })
        .collect();
    DeviceState {
        devices,
        plan_version: 0,
    }
}

fn set_local_preference(selector: &str, enabled: bool) -> Result<()> {
    let state = local_device_state()?;
    let device = resolve_device(&state.devices, selector)?;
    if device.health == DeviceHealth::Unavailable {
        bail!(
            "{} is visible but has no stable identity; its startup preference cannot be persisted",
            device.name
        );
    }
    let Some(db) = crate::metadata_db::handle() else {
        bail!(
            "cannot persist the device preference because the metadata DB is disabled or unavailable"
        );
    };
    let preferences = mold_db::DevicePreferences::new(db);
    let previous = preferences.get(&device.id)?;
    preferences.set(&device.id, enabled)?;
    let action = if enabled { "enabled" } else { "disabled" };
    if previous == Some(enabled) {
        println!("{}: already {action} for the next server start", device.id);
    } else {
        println!("{}: {action} for the next server start", device.id);
    }
    Ok(())
}

async fn set_enabled_with_client(client: &MoldClient, selector: &str, enabled: bool) -> Result<()> {
    let capabilities = client
        .server_capabilities()
        .await
        .context("failed to read server capabilities")?;
    let state = client
        .devices()
        .await
        .context("failed to read server devices")?;
    let device = resolve_device(&state.devices, selector)?;
    ensure_supported(&capabilities, device, enabled)?;
    let id = device.id.clone();
    let device = client
        .set_device_enabled(&id, enabled)
        .await
        .with_context(|| format!("failed to update device {id}"))?;
    if device.restart_required {
        println!("{}: enabled after server restart", device.name);
    } else {
        println!("{}: {}", device.id, human_state(&device));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use mold_core::{DeviceActivity, DeviceKind, DeviceMemoryInfo, DeviceTelemetry, GpuBackend};

    #[test]
    fn remote_completion_never_substitutes_local_device_ids() {
        assert!(complete_device_id_for_host("https://gpu-box:7680").is_empty());
    }

    fn device(id: &str, ordinal: usize) -> DeviceInfo {
        DeviceInfo {
            id: id.into(),
            backend: GpuBackend::Cuda,
            ordinal: Some(ordinal),
            device_kind: DeviceKind::FullGpu,
            nvml_uuid: None,
            physical_uuid: None,
            mig_uuid: None,
            mig_parent_uuid: None,
            mig_profile: None,
            name: "GPU".into(),
            pci_bus_id: None,
            compute_capability: None,
            memory: DeviceMemoryInfo {
                total_bytes: Some(24),
                used_bytes: Some(0),
                mold_used_bytes: Some(0),
                other_used_bytes: Some(0),
            },
            telemetry: DeviceTelemetry {
                utilization_percent: Some(0),
                temperature_c: None,
                power_w: None,
            },
            desired_enabled: true,
            restart_required: false,
            admin_state: DeviceAdminState::Enabled,
            health: DeviceHealth::Healthy,
            activity: DeviceActivity::Idle,
            schedulable: true,
            unschedulable_reason: None,
            loaded_models: vec![],
            active_work_id: None,
            planned_work_ids: vec![],
        }
    }

    #[test]
    fn resolves_stable_id_before_display_ordinal() {
        let devices = vec![device("cuda:stable", 7), device("7", 2)];
        assert_eq!(resolve_device(&devices, "7").unwrap().id, "7");
        assert_eq!(resolve_device(&devices, "2").unwrap().id, "7");
        assert!(resolve_device(&devices, "3").is_err());
    }

    #[test]
    fn gates_live_controls_but_allows_restart_recovery() {
        let mut legacy = ServerCapabilities::default();
        legacy.devices.available = true;
        legacy.devices.restart_enable = true;
        let mut disabled = device("cuda:disabled", 0);
        disabled.desired_enabled = false;
        disabled.admin_state = DeviceAdminState::Disabled;
        assert!(ensure_supported(&legacy, &disabled, true).is_ok());
        assert!(ensure_supported(&legacy, &disabled, false).is_err());

        let mut enabled = disabled.clone();
        enabled.desired_enabled = true;
        assert!(ensure_supported(&legacy, &enabled, true).is_err());

        legacy.devices.lifecycle = true;
        legacy.dispatch.v2_authoritative = false;
        assert!(ensure_supported(&legacy, &disabled, true).is_ok());
        assert!(ensure_supported(&legacy, &enabled, false).is_err());

        legacy.dispatch.v2_authoritative = true;
        assert!(ensure_supported(&legacy, &enabled, false).is_ok());
    }

    #[test]
    fn startup_excluded_uses_the_wire_spelling() {
        let mut excluded = device("cuda:excluded", 0);
        excluded.admin_state = DeviceAdminState::StartupExcluded;
        assert_eq!(human_state(&excluded), "startup_excluded");
    }

    #[test]
    fn formats_one_two_eight_and_sixty_four_devices_including_mig_and_lifecycle_states() {
        for count in [1, 2, 8, 64] {
            let lines = (0..count)
                .map(|ordinal| format_device_line(&device(&format!("cuda:{ordinal}"), ordinal)))
                .collect::<Vec<_>>();
            assert_eq!(lines.len(), count);
            assert!(lines.iter().all(|line| line.contains("GPU ")));
        }
        let mut mig = device("cuda:mig", 0);
        mig.device_kind = DeviceKind::Mig;
        mig.mig_profile = Some("1g.10gb".into());
        mig.admin_state = DeviceAdminState::Draining;
        assert!(format_device_line(&mig).contains("MIG 1g.10gb"));
        assert!(format_device_line(&mig).contains("finishing current work"));

        let mut disabled = device("cuda:disabled", 1);
        disabled.desired_enabled = false;
        disabled.admin_state = DeviceAdminState::Disabled;
        assert!(format_device_line(&disabled).contains("disabled"));

        let mut unavailable = device("cuda:unavailable", 2);
        unavailable.health = DeviceHealth::Unavailable;
        unavailable.schedulable = false;
        assert!(format_device_line(&unavailable).contains("unavailable"));
    }

    #[test]
    fn unknown_vram_is_not_rendered_as_zero() {
        let mut unknown = device("cuda:unknown", 0);
        unknown.memory.used_bytes = None;
        unknown.memory.total_bytes = None;
        assert!(format_device_line(&unknown).contains("VRAM —"));
        assert!(!format_device_line(&unknown).contains("0.0/0.0"));
    }

    #[test]
    fn local_projection_preserves_stable_identity_preferences_and_unknown_telemetry() {
        let discovered = mold_inference::device::DiscoveredGpu {
            ordinal: 2,
            stable_id: Some("cuda:stable".into()),
            raw_cuda_uuid: None,
            device_kind: Some(mold_inference::device::CudaDeviceKind::FullGpu),
            identity_error: None,
            backend: GpuBackend::Cuda,
            name: "Local GPU".into(),
            compute_capability: Some((8, 9)),
            pci_bus_id: Some("0000:01:00.0".into()),
            total_vram_bytes: 24 << 30,
            free_vram_bytes: 20 << 30,
        };
        let mut preferences = BTreeMap::new();
        preferences.insert(
            "cuda:stable".into(),
            mold_db::DevicePreference {
                device_id: "cuda:stable".into(),
                desired_enabled: false,
                updated_at: 1,
            },
        );

        let state = project_local_devices(vec![discovered], &preferences);
        let device = &state.devices[0];
        assert_eq!(device.id, "cuda:stable");
        assert_eq!(device.ordinal, Some(2));
        assert_eq!(device.compute_capability.as_deref(), Some("8.9"));
        assert_eq!(device.memory.used_bytes, Some(4 << 30));
        assert_eq!(device.telemetry.utilization_percent, None);
        assert!(!device.desired_enabled);
        assert_eq!(device.admin_state, DeviceAdminState::Disabled);
        assert!(!device.schedulable);
    }

    #[test]
    fn local_projection_keeps_identity_failures_visible_but_not_addressable() {
        let discovered = mold_inference::device::DiscoveredGpu {
            ordinal: 0,
            stable_id: None,
            raw_cuda_uuid: None,
            device_kind: Some(mold_inference::device::CudaDeviceKind::UnknownCuda),
            identity_error: Some("UUID unavailable".into()),
            backend: GpuBackend::Cuda,
            name: "Broken GPU".into(),
            compute_capability: None,
            pci_bus_id: None,
            total_vram_bytes: 0,
            free_vram_bytes: 0,
        };
        let state = project_local_devices(vec![discovered], &BTreeMap::new());
        assert_eq!(state.devices[0].health, DeviceHealth::Unavailable);
        assert_eq!(state.devices[0].memory.total_bytes, None);
        assert_eq!(
            state.devices[0].unschedulable_reason.as_deref(),
            Some("UUID unavailable")
        );
    }

    #[tokio::test]
    async fn legacy_live_disable_reads_the_target_but_never_sends_patch() {
        use wiremock::matchers::{method, path, path_regex};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path_regex(r"^/api/capabilities$"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "gallery": { "can_delete": true },
                "catalog": { "available": false, "families": [], "sort": [] },
                "devices": { "available": true, "lifecycle": false },
                "dispatch": {
                    "modes": ["legacy", "observe", "v2"],
                    "active_mode": "legacy",
                    "v2_authoritative": false,
                    "observes_v2_decisions": false
                }
            })))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/api/devices"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(mold_core::DeviceState {
                    devices: vec![device("cuda:stable", 0)],
                    plan_version: 0,
                }),
            )
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("PATCH"))
            .and(path_regex(r"^/api/devices/.*$"))
            .respond_with(ResponseTemplate::new(500))
            .expect(0)
            .mount(&server)
            .await;

        let error = set_enabled_with_client(&MoldClient::new(&server.uri()), "0", false)
            .await
            .unwrap_err()
            .to_string();
        assert!(error.contains("live GPU controls require Scheduler V2"));
    }
}
