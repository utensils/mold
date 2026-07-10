use mold_core::runpod::{
    image_tag_for_gpu, CreatePodRequest, Datacenter, GpuType, NetworkVolume, Pod, RunPodClient,
    DEFAULT_ENDPOINT,
};
use serde::{Deserialize, Serialize};

use crate::commands::AppState;

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RunPodAccount {
    pub email: String,
    pub balance: f64,
    pub spend_per_hour: f64,
    pub spend_limit: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RunPodOverview {
    pub configured: bool,
    pub credential_source: Option<&'static str>,
    pub account: Option<RunPodAccount>,
    pub pods: Vec<Pod>,
    pub gpus: Vec<GpuType>,
    pub datacenters: Vec<Datacenter>,
    pub network_volumes: Vec<NetworkVolume>,
}

impl RunPodOverview {
    fn unconfigured() -> Self {
        Self {
            configured: false,
            credential_source: None,
            account: None,
            pods: vec![],
            gpus: vec![],
            datacenters: vec![],
            network_volumes: vec![],
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RunPodCreateInput {
    pub name: Option<String>,
    pub gpu_type_id: String,
    pub gpu_display_name: String,
    pub cloud_type: String,
    pub datacenter_id: Option<String>,
    pub container_disk_gb: u32,
    pub volume_gb: u32,
    pub network_volume_id: Option<String>,
    pub model: Option<String>,
    #[serde(default)]
    pub include_hf_token: bool,
}

fn client(state: &AppState) -> Result<Option<(RunPodClient, &'static str)>, String> {
    let config = mold_core::Config::load_or_default();
    let keychain = state
        .secrets
        .get("runpod-api-key")
        .map_err(|e| e.to_string())?
        .filter(|key| !key.is_empty());
    let environment = std::env::var("RUNPOD_API_KEY")
        .ok()
        .filter(|key| !key.is_empty());
    let configured = config.runpod.api_key.clone().filter(|key| !key.is_empty());
    let Some((key, source)) = keychain
        .map(|key| (key, "keychain"))
        .or_else(|| environment.map(|key| (key, "environment")))
        .or_else(|| configured.map(|key| (key, "config")))
    else {
        return Ok(None);
    };
    Ok(Some((
        RunPodClient::new(
            config
                .runpod
                .endpoint
                .as_deref()
                .unwrap_or(DEFAULT_ENDPOINT),
            key,
        ),
        source,
    )))
}

fn default_pod_name() -> String {
    let epoch = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format!("mold-{epoch}")
}

fn build_request(
    input: RunPodCreateInput,
    hf_token: Option<String>,
) -> Result<CreatePodRequest, String> {
    if input.gpu_type_id.trim().is_empty() {
        return Err("Choose a GPU before launching.".into());
    }
    let cloud_type = input.cloud_type.to_ascii_uppercase();
    if cloud_type != "SECURE" && cloud_type != "COMMUNITY" {
        return Err("Cloud type must be Secure or Community.".into());
    }
    if !(10..=1000).contains(&input.container_disk_gb) {
        return Err("Container disk must be between 10 and 1000 GB.".into());
    }
    if input.volume_gb > 10_000 {
        return Err("Workspace volume must be 10000 GB or smaller.".into());
    }

    let mut env = serde_json::Map::new();
    env.insert("MOLD_LOG".into(), "info".into());
    if let Some(model) = input.model.filter(|model| !model.trim().is_empty()) {
        env.insert("MOLD_DEFAULT_MODEL".into(), model.into());
    }
    if input.include_hf_token {
        if let Some(token) = hf_token.filter(|token| !token.is_empty()) {
            env.insert("HF_TOKEN".into(), token.into());
        }
    }

    Ok(CreatePodRequest {
        name: input
            .name
            .filter(|name| !name.trim().is_empty())
            .unwrap_or_else(default_pod_name),
        image_name: format!(
            "ghcr.io/utensils/mold:{}",
            image_tag_for_gpu(&input.gpu_display_name)
        ),
        gpu_type_ids: vec![input.gpu_type_id],
        cloud_type,
        data_center_ids: input.datacenter_id.map(|id| vec![id]),
        gpu_count: 1,
        container_disk_in_gb: input.container_disk_gb,
        volume_in_gb: input.volume_gb,
        volume_mount_path: "/workspace".into(),
        ports: vec!["7680/http".into(), "22/tcp".into()],
        env,
        network_volume_id: input.network_volume_id,
    })
}

#[tauri::command]
pub async fn runpod_overview(state: tauri::State<'_, AppState>) -> Result<RunPodOverview, String> {
    let Some((client, credential_source)) = client(&state)? else {
        return Ok(RunPodOverview::unconfigured());
    };

    let (user, pods, gpus, datacenters, network_volumes) = tokio::try_join!(
        client.user(),
        client.list_pods(),
        client.gpu_types(),
        client.datacenters(),
        client.network_volumes(),
    )
    .map_err(|e| format!("{e:#}"))?;

    Ok(RunPodOverview {
        configured: true,
        credential_source: Some(credential_source),
        account: Some(RunPodAccount {
            email: user.email,
            balance: user.client_balance,
            spend_per_hour: user.current_spend_per_hr,
            spend_limit: user.spend_limit,
        }),
        pods,
        gpus,
        datacenters,
        network_volumes,
    })
}

#[tauri::command]
pub async fn runpod_create(
    state: tauri::State<'_, AppState>,
    input: RunPodCreateInput,
) -> Result<Pod, String> {
    let (client, _) = client(&state)?.ok_or_else(|| "Add a RunPod API key first.".to_string())?;
    let hf_token = state.secrets.get("hf-token").map_err(|e| e.to_string())?;
    let request = build_request(input, hf_token)?;
    client
        .create_pod(&request)
        .await
        .map_err(|e| format!("{e:#}"))
}

#[tauri::command]
pub async fn runpod_start(state: tauri::State<'_, AppState>, id: String) -> Result<(), String> {
    let (client, _) = client(&state)?.ok_or_else(|| "Add a RunPod API key first.".to_string())?;
    client.start_pod(&id).await.map_err(|e| format!("{e:#}"))
}

#[tauri::command]
pub async fn runpod_stop(state: tauri::State<'_, AppState>, id: String) -> Result<(), String> {
    let (client, _) = client(&state)?.ok_or_else(|| "Add a RunPod API key first.".to_string())?;
    client.stop_pod(&id).await.map_err(|e| format!("{e:#}"))
}

#[tauri::command]
pub async fn runpod_delete(state: tauri::State<'_, AppState>, id: String) -> Result<(), String> {
    let (client, _) = client(&state)?.ok_or_else(|| "Add a RunPod API key first.".to_string())?;
    client.delete_pod(&id).await.map_err(|e| format!("{e:#}"))
}

#[tauri::command]
pub async fn runpod_logs(state: tauri::State<'_, AppState>, id: String) -> Result<String, String> {
    let (client, _) = client(&state)?.ok_or_else(|| "Add a RunPod API key first.".to_string())?;
    client.pod_logs(&id).await.map_err(|e| format!("{e:#}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input() -> RunPodCreateInput {
        RunPodCreateInput {
            name: Some("studio".into()),
            gpu_type_id: "NVIDIA GeForce RTX 5090".into(),
            gpu_display_name: "RTX 5090".into(),
            cloud_type: "secure".into(),
            datacenter_id: Some("US-KS-2".into()),
            container_disk_gb: 30,
            volume_gb: 80,
            network_volume_id: None,
            model: Some("flux-dev:q8".into()),
            include_hf_token: true,
        }
    }

    #[test]
    fn create_request_selects_image_ports_model_and_token() {
        let request = build_request(input(), Some("hf_secret".into())).unwrap();
        assert_eq!(request.image_name, "ghcr.io/utensils/mold:latest-sm120");
        assert_eq!(request.ports, ["7680/http", "22/tcp"]);
        assert_eq!(request.cloud_type, "SECURE");
        assert_eq!(request.data_center_ids, Some(vec!["US-KS-2".into()]));
        assert_eq!(request.env["MOLD_DEFAULT_MODEL"], "flux-dev:q8");
        assert_eq!(request.env["HF_TOKEN"], "hf_secret");
    }

    #[test]
    fn create_request_rejects_invalid_resource_values() {
        let mut invalid = input();
        invalid.container_disk_gb = 2;
        assert!(build_request(invalid, None)
            .unwrap_err()
            .contains("between 10 and 1000"));
    }
}
