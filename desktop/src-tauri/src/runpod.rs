use mold_core::runpod::{
    valid_network_volume_size, CreateNetworkVolumeRequest, CreatePodRequest, Datacenter, GpuType,
    NetworkVolume, Pod, RunPodClient, UpdateNetworkVolumeRequest, DEFAULT_ENDPOINT,
    NETWORK_VOLUME_MAX_GB, NETWORK_VOLUME_MIN_GB,
};
use serde::{Deserialize, Serialize};
use tokio::sync::OnceCell;

use crate::commands::AppState;

static SUPPORTED_POD_GPU_IDS: OnceCell<std::collections::HashSet<String>> = OnceCell::const_new();

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

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RunPodNetworkVolumeCreateInput {
    pub name: String,
    pub size_gb: u32,
    pub datacenter_id: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RunPodNetworkVolumeUpdateInput {
    pub id: String,
    pub name: Option<String>,
    pub size_gb: Option<u32>,
}

fn client(state: &AppState) -> Result<Option<(RunPodClient, &'static str)>, String> {
    let config = mold_core::Config::load_or_default();
    let environment = std::env::var("RUNPOD_API_KEY")
        .ok()
        .filter(|key| !key.is_empty());
    if let Some(key) = environment {
        return Ok(Some((
            RunPodClient::new(
                config
                    .runpod
                    .endpoint
                    .as_deref()
                    .unwrap_or(DEFAULT_ENDPOINT),
                key,
            ),
            "environment",
        )));
    }
    let stored = state
        .secrets
        .get("runpod-api-key")
        .map_err(|e| e.to_string())?
        .filter(|key| !key.is_empty());
    let configured = config.runpod.api_key.clone().filter(|key| !key.is_empty());
    let Some((key, source)) = stored
        .map(|key| (key, "app"))
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
    image_name: String,
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
        image_name,
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

fn apply_network_volume_constraints(input: &mut RunPodCreateInput, volume: &NetworkVolume) {
    input.cloud_type = "SECURE".into();
    input.datacenter_id = Some(volume.data_center_id.clone());
    input.network_volume_id = Some(volume.id.clone());
    // RunPod replaces the ordinary Pod volume with the attached network
    // volume. Sending zero makes the launch plan and billing intent explicit.
    input.volume_gb = 0;
}

#[tauri::command]
pub async fn runpod_overview(state: tauri::State<'_, AppState>) -> Result<RunPodOverview, String> {
    let Some((client, credential_source)) = client(&state)? else {
        return Ok(RunPodOverview::unconfigured());
    };

    let (user, mut pods, mut gpus, datacenters, network_volumes, supported_gpu_ids) =
        tokio::try_join!(
            client.user(),
            client.list_pods(),
            client.gpu_types(),
            client.datacenters(),
            client.network_volumes(),
            SUPPORTED_POD_GPU_IDS.get_or_try_init(|| client.supported_pod_gpu_type_ids()),
        )
        .map_err(|e| format!("{e:#}"))?;
    gpus.retain(|gpu| {
        gpu.id
            .as_deref()
            .or_else(|| (!gpu.gpu_id.is_empty()).then_some(gpu.gpu_id.as_str()))
            .is_some_and(|id| supported_gpu_ids.contains(id))
    });
    for pod in &mut pods {
        if pod.network_volume.is_none() {
            pod.network_volume = pod
                .network_volume_id
                .as_deref()
                .and_then(|id| network_volumes.iter().find(|volume| volume.id == id))
                .cloned();
        }
    }

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
    mut input: RunPodCreateInput,
) -> Result<Pod, String> {
    let (client, _) = client(&state)?.ok_or_else(|| "Add a RunPod API key first.".to_string())?;
    if let Some(volume_id) = input.network_volume_id.clone() {
        let volume = client
            .get_network_volume(&volume_id)
            .await
            .map_err(|e| format!("Could not load network volume {volume_id}: {e:#}"))?;
        apply_network_volume_constraints(&mut input, &volume);
    }
    let hf_token = if input.include_hf_token {
        state.secrets.get("hf-token").map_err(|e| e.to_string())?
    } else {
        None
    };
    let image_name = resolve_published_image_for_gpu(&input.gpu_display_name).await?;
    let request = build_request(input, hf_token, image_name)?;
    client
        .create_pod(&request)
        .await
        .map_err(|e| format!("{e:#}"))
}

async fn resolve_published_image_for_gpu(gpu_display_name: &str) -> Result<String, String> {
    mold_core::cuda_distribution::resolve_distribution_image_reference(
        mold_core::cuda_distribution::OFFICIAL_IMAGE_REPOSITORY,
        gpu_display_name,
        mold_core::cuda_distribution::distribution_image_version(),
    )
    .await
    .map_err(|error| format!("Could not resolve the Mold container: {error:#}"))
}

#[tauri::command]
pub async fn runpod_network_volume_create(
    state: tauri::State<'_, AppState>,
    input: RunPodNetworkVolumeCreateInput,
) -> Result<NetworkVolume, String> {
    let (client, _) = client(&state)?.ok_or_else(|| "Add a RunPod API key first.".to_string())?;
    let name = input.name.trim();
    let datacenter_id = input.datacenter_id.trim();
    if name.is_empty() {
        return Err("Enter a name for the network volume.".into());
    }
    if datacenter_id.is_empty() {
        return Err("Choose a datacenter for the network volume.".into());
    }
    if !valid_network_volume_size(input.size_gb) {
        return Err(format!(
            "Network volume size must be between {NETWORK_VOLUME_MIN_GB} and {NETWORK_VOLUME_MAX_GB} GB."
        ));
    }
    client
        .create_network_volume(&CreateNetworkVolumeRequest {
            name: name.into(),
            size: input.size_gb,
            data_center_id: datacenter_id.into(),
        })
        .await
        .map_err(|e| format!("{e:#}"))
}

#[tauri::command]
pub async fn runpod_network_volume_update(
    state: tauri::State<'_, AppState>,
    input: RunPodNetworkVolumeUpdateInput,
) -> Result<NetworkVolume, String> {
    let (client, _) = client(&state)?.ok_or_else(|| "Add a RunPod API key first.".to_string())?;
    if input.name.is_none() && input.size_gb.is_none() {
        return Err("Change the name or enter a larger size.".into());
    }
    let name = input.name.map(|name| name.trim().to_string());
    if name.as_deref().is_some_and(str::is_empty) {
        return Err("Network volume name cannot be empty.".into());
    }
    if let Some(size) = input.size_gb {
        if !valid_network_volume_size(size) {
            return Err(format!(
                "Network volume size must be between {NETWORK_VOLUME_MIN_GB} and {NETWORK_VOLUME_MAX_GB} GB."
            ));
        }
        let current = client
            .get_network_volume(&input.id)
            .await
            .map_err(|e| format!("{e:#}"))?;
        if size <= current.size {
            return Err(format!(
                "Network volumes can only grow. Current size is {} GB.",
                current.size
            ));
        }
    }
    client
        .update_network_volume(
            &input.id,
            &UpdateNetworkVolumeRequest {
                name,
                size: input.size_gb,
            },
        )
        .await
        .map_err(|e| format!("{e:#}"))
}

#[tauri::command]
pub async fn runpod_network_volume_delete(
    state: tauri::State<'_, AppState>,
    id: String,
) -> Result<(), String> {
    let (client, _) = client(&state)?.ok_or_else(|| "Add a RunPod API key first.".to_string())?;
    client
        .delete_network_volume_if_detached(&id)
        .await
        .map_err(|e| format!("Could not safely delete network volume: {e:#}"))
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
        let request = build_request(
            input(),
            Some("hf_secret".into()),
            "ghcr.io/utensils/mold:latest-sm120".into(),
        )
        .unwrap();
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
        assert!(build_request(invalid, None, "image".into())
            .unwrap_err()
            .contains("between 10 and 1000"));
    }

    #[tokio::test]
    async fn published_image_routing_rejects_grace_and_retains_b200() {
        for name in ["NVIDIA GH200", "NVIDIA GB200", "NVIDIA GB300"] {
            let error = resolve_published_image_for_gpu(name).await.unwrap_err();
            assert!(error.contains("linux/arm64"), "{name}: {error}");
        }
        assert_eq!(
            resolve_published_image_for_gpu("NVIDIA B200")
                .await
                .unwrap(),
            "ghcr.io/utensils/mold:latest-sm100"
        );
    }

    #[test]
    fn network_volume_forces_secure_cloud_and_its_datacenter() {
        let mut input = input();
        input.cloud_type = "COMMUNITY".into();
        input.datacenter_id = Some("EU-RO-1".into());
        input.network_volume_id = Some("nv-1".into());
        apply_network_volume_constraints(
            &mut input,
            &NetworkVolume {
                id: "nv-1".into(),
                name: "models".into(),
                data_center_id: "US-KS-2".into(),
                size: 100,
            },
        );
        let request = build_request(input, None, "image".into()).unwrap();
        assert_eq!(request.cloud_type, "SECURE");
        assert_eq!(request.data_center_ids, Some(vec!["US-KS-2".into()]));
        assert_eq!(request.network_volume_id.as_deref(), Some("nv-1"));
        assert_eq!(request.volume_in_gb, 0);
    }
}
