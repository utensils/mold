use anyhow::{bail, Result};
use mold_core::{DeviceAdminState, DeviceInfo, MoldClient, ServerCapabilities};

pub async fn list(json: bool) -> Result<()> {
    let devices = MoldClient::from_env().devices().await?;
    if json {
        println!("{}", serde_json::to_string_pretty(&devices)?);
        return Ok(());
    }
    if devices.devices.is_empty() {
        println!("No runtime-visible GPU devices.");
        return Ok(());
    }
    for device in devices.devices {
        let ordinal = device
            .ordinal
            .map(|ordinal| format!("GPU {ordinal}"))
            .unwrap_or_else(|| "GPU —".to_string());
        let memory = device
            .memory
            .total_bytes
            .map(|bytes| format!("{:.1} GiB", bytes as f64 / 1_073_741_824.0))
            .unwrap_or_else(|| "VRAM unknown".to_string());
        println!(
            "{ordinal:<7} {:<28} {:<18} {:<16} {memory}",
            device.name,
            short_id(&device.id),
            admin_label(device.admin_state),
        );
    }
    Ok(())
}

pub async fn set(selector: &str, enabled: bool) -> Result<()> {
    let client = MoldClient::from_env();
    let capabilities = client.server_capabilities().await?;
    let state = client.devices().await?;
    let device = resolve_device(&state.devices, selector)?;
    ensure_supported(&capabilities, device, enabled)?;
    let updated = client.set_device_enabled(&device.id, enabled).await?;
    if updated.restart_required {
        println!("{}: enabled after server restart", updated.name);
    } else {
        println!("{}: {}", updated.name, admin_label(updated.admin_state));
    }
    Ok(())
}

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
    if let Ok(ordinal) = selector.parse::<usize>() {
        return devices
            .iter()
            .find(|device| device.ordinal == Some(ordinal))
            .ok_or_else(|| anyhow::anyhow!("no runtime-visible GPU has ordinal {ordinal}"));
    }
    let matches = devices
        .iter()
        .filter(|device| device.id == selector)
        .collect::<Vec<_>>();
    match matches.as_slice() {
        [device] => Ok(*device),
        [] => bail!("unknown stable device ID '{selector}'"),
        _ => bail!("ambiguous device selector '{selector}'"),
    }
}

fn short_id(id: &str) -> String {
    if id.len() <= 18 {
        id.to_string()
    } else {
        format!("{}…{}", &id[..11], &id[id.len() - 6..])
    }
}

fn admin_label(state: DeviceAdminState) -> &'static str {
    match state {
        DeviceAdminState::StartupExcluded => "startup excluded",
        DeviceAdminState::Starting => "starting",
        DeviceAdminState::Enabled => "enabled",
        DeviceAdminState::Draining => "draining",
        DeviceAdminState::Disabled => "disabled",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mold_core::{
        DeviceActivity, DeviceHealth, DeviceKind, DeviceMemoryInfo, DeviceTelemetry, GpuBackend,
    };

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
            name: format!("GPU {ordinal}"),
            pci_bus_id: None,
            compute_capability: None,
            memory: DeviceMemoryInfo {
                total_bytes: None,
                used_bytes: None,
                mold_used_bytes: None,
                other_used_bytes: None,
            },
            telemetry: DeviceTelemetry {
                utilization_percent: None,
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
            loaded_models: Vec::new(),
            active_work_id: None,
            planned_work_ids: Vec::new(),
        }
    }

    #[test]
    fn resolves_stable_id_or_display_ordinal() {
        let devices = vec![device("cuda:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", 0)];
        assert_eq!(resolve_device(&devices, "0").unwrap().id, devices[0].id);
        assert_eq!(
            resolve_device(&devices, "cuda:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
                .unwrap()
                .id,
            devices[0].id
        );
        assert!(resolve_device(&devices, "1").is_err());
    }

    #[test]
    fn gates_live_controls_but_allows_restart_recovery() {
        let mut legacy = ServerCapabilities::default();
        legacy.devices.available = true;
        legacy.devices.restart_enable = true;
        let mut disabled = device("cuda:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", 0);
        disabled.desired_enabled = false;
        disabled.admin_state = DeviceAdminState::Disabled;
        assert!(ensure_supported(&legacy, &disabled, true).is_ok());
        assert!(ensure_supported(&legacy, &disabled, false).is_err());

        let mut enabled = disabled.clone();
        enabled.desired_enabled = true;
        assert!(ensure_supported(&legacy, &enabled, true).is_err());

        legacy.devices.lifecycle = true;
        legacy.dispatch.v2_authoritative = false;
        assert!(
            ensure_supported(&legacy, &disabled, true).is_ok(),
            "restart recovery remains available when lifecycle is advertised without authoritative dispatch"
        );
        assert!(
            ensure_supported(&legacy, &enabled, false).is_err(),
            "live mutation requires both lifecycle and authoritative dispatch"
        );

        legacy.devices.lifecycle = true;
        legacy.dispatch.v2_authoritative = true;
        assert!(ensure_supported(&legacy, &enabled, false).is_ok());
    }
}
