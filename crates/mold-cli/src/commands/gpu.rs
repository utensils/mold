use anyhow::{bail, Context, Result};
use colored::Colorize;
use mold_core::{DeviceAdminState, DeviceHealth, DeviceInfo, MoldClient, ServerCapabilities};

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

fn format_device_line(device: &DeviceInfo) -> String {
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
    let used = device.memory.used_bytes.unwrap_or(0) as f64 / 1024_f64.powi(3);
    let total = device.memory.total_bytes.unwrap_or(0) as f64 / 1024_f64.powi(3);
    let utilization = device
        .telemetry
        .utilization_percent
        .map(|value| format!("{value}%"))
        .unwrap_or_else(|| "—".into());
    format!(
        "{}  {}  {}  {}  {}  VRAM {:.1}/{:.1} GiB  util {}",
        device.id,
        ordinal,
        device.name,
        kind,
        human_state(device),
        used,
        total,
        utilization
    )
}

pub async fn list(json: bool) -> Result<()> {
    let state = MoldClient::from_env()
        .devices()
        .await
        .context("failed to read server devices")?;
    if json {
        println!("{}", serde_json::to_string_pretty(&state)?);
        return Ok(());
    }
    if state.devices.is_empty() {
        println!("No compute devices visible.");
        return Ok(());
    }
    for device in &state.devices {
        println!("{}", format_device_line(device));
    }
    Ok(())
}

pub async fn set(selector: &str, enabled: bool) -> Result<()> {
    let client = MoldClient::from_env();
    set_enabled_with_client(&client, selector, enabled).await
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
