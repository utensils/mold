use mold_core::runpod::{
    canonical_supported_gpu_type_id, normalized_gpu_type_identity, valid_network_volume_size,
    CreateNetworkVolumeRequest, CreatePodRequest, Datacenter, GpuType, NetworkVolume, Pod,
    RunPodClient, UpdateNetworkVolumeRequest, DEFAULT_ENDPOINT, NETWORK_VOLUME_MAX_GB,
    NETWORK_VOLUME_MIN_GB,
};
use serde::{Deserialize, Serialize};
use std::future::Future;
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

trait RunPodLaunchClient {
    async fn gpu_types(&self) -> anyhow::Result<Vec<GpuType>>;
    async fn supported_pod_gpu_type_ids(&self)
        -> anyhow::Result<std::collections::HashSet<String>>;
    async fn create_pod(&self, request: &CreatePodRequest) -> anyhow::Result<Pod>;
}

impl RunPodLaunchClient for RunPodClient {
    async fn gpu_types(&self) -> anyhow::Result<Vec<GpuType>> {
        RunPodClient::gpu_types(self).await
    }

    async fn supported_pod_gpu_type_ids(
        &self,
    ) -> anyhow::Result<std::collections::HashSet<String>> {
        RunPodClient::supported_pod_gpu_type_ids(self).await
    }

    async fn create_pod(&self, request: &CreatePodRequest) -> anyhow::Result<Pod> {
        RunPodClient::create_pod(self, request).await
    }
}

fn normalized_provider_identity(value: &str) -> String {
    normalized_gpu_type_identity(value)
}

fn provider_gpu_id(gpu: &GpuType) -> Option<&str> {
    gpu.allocation_type_id()
}

fn normalize_supported_inventory_gpu(
    gpu: &mut GpuType,
    supported_gpu_ids: &std::collections::HashSet<String>,
) -> bool {
    let Some(provider_id) = gpu.allocation_type_id() else {
        return false;
    };
    let Some(supported_id) = canonical_supported_gpu_type_id(supported_gpu_ids, provider_id) else {
        return false;
    };
    gpu.id = Some(supported_id.to_string());
    true
}

fn resolve_provider_gpu<'a>(
    gpu_types: &'a [GpuType],
    submitted_id: &str,
    submitted_display_name: &str,
) -> Result<(&'a str, &'a str), String> {
    let submitted_id = normalized_provider_identity(submitted_id);
    if submitted_id.is_empty() {
        return Err("Choose a GPU before launching.".into());
    }
    if normalized_provider_identity(submitted_display_name).is_empty() {
        return Err(
            "The selected GPU name is missing. Refresh RunPod inventory and try again.".into(),
        );
    }

    let mut matches = gpu_types.iter().filter(|gpu| {
        provider_gpu_id(gpu).is_some_and(|id| normalized_provider_identity(id) == submitted_id)
    });
    let gpu = matches.next().ok_or_else(|| {
        "The selected GPU is no longer available. Refresh RunPod inventory and choose it again."
            .to_string()
    })?;
    if matches.next().is_some() {
        return Err(
            "RunPod returned duplicate GPU identities. Refresh inventory before launching.".into(),
        );
    }
    let provider_id = provider_gpu_id(gpu).expect("matched provider GPU has an id");
    let provider_name = gpu.display_name.trim();
    if provider_name.is_empty() {
        return Err("RunPod returned a GPU without a display name. Refresh inventory.".into());
    }
    if normalized_provider_identity(provider_name)
        != normalized_provider_identity(submitted_display_name)
    {
        return Err(format!(
            "The selected GPU name does not match RunPod's current inventory ({provider_name}). Refresh and choose it again."
        ));
    }

    // Both fields come from the fresh provider record. Checking the provider
    // ID as well as its display label prevents an inconsistent inventory
    // record from disguising a Grace CPU/GPU product as an amd64 B-series GPU.
    for authoritative_identity in [provider_id, provider_name] {
        mold_core::cuda_distribution::ensure_published_image_platform(authoritative_identity)
            .map_err(|error| error.to_string())?;
    }
    Ok((provider_id, provider_name))
}

async fn prepare_pod_request_with_image_resolver<C, F, Fut>(
    client: &C,
    mut input: RunPodCreateInput,
    hf_token: Option<String>,
    image_resolver: F,
) -> Result<CreatePodRequest, String>
where
    C: RunPodLaunchClient,
    F: FnOnce(&str) -> Fut,
    Fut: Future<Output = Result<String, String>>,
{
    let gpu_types = client
        .gpu_types()
        .await
        .map_err(|error| format!("Could not refresh RunPod GPU inventory: {error:#}"))?;
    let (provider_id, provider_name) =
        resolve_provider_gpu(&gpu_types, &input.gpu_type_id, &input.gpu_display_name)?;
    let supported_ids = client
        .supported_pod_gpu_type_ids()
        .await
        .map_err(|error| format!("Could not validate RunPod GPU launch support: {error:#}"))?;
    let Some(supported_provider_id) = canonical_supported_gpu_type_id(&supported_ids, provider_id)
    else {
        return Err(
            "The selected GPU is not currently accepted by RunPod's Pod API. Refresh inventory and choose another GPU."
                .into(),
        );
    };
    input.gpu_type_id = supported_provider_id.to_string();
    input.gpu_display_name = provider_name.to_string();
    let image_name = image_resolver(supported_provider_id).await?;
    build_request(input, hf_token, image_name)
}

async fn launch_pod_with_image_resolver<C, F, Fut>(
    client: &C,
    input: RunPodCreateInput,
    hf_token: Option<String>,
    image_resolver: F,
) -> Result<Pod, String>
where
    C: RunPodLaunchClient,
    F: FnOnce(&str) -> Fut,
    Fut: Future<Output = Result<String, String>>,
{
    let request =
        prepare_pod_request_with_image_resolver(client, input, hf_token, image_resolver).await?;
    client
        .create_pod(&request)
        .await
        .map_err(|error| format!("{error:#}"))
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
    gpus.retain_mut(|gpu| normalize_supported_inventory_gpu(gpu, supported_gpu_ids));
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
    launch_pod_with_image_resolver(&client, input, hf_token, |gpu_name| {
        let gpu_name = gpu_name.to_string();
        async move { resolve_published_image_for_gpu(&gpu_name).await }
    })
    .await
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
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Default)]
    struct FakeLaunchClient {
        gpu_types: Vec<GpuType>,
        supported_ids: std::collections::HashSet<String>,
        create_calls: AtomicUsize,
    }

    impl RunPodLaunchClient for FakeLaunchClient {
        async fn gpu_types(&self) -> anyhow::Result<Vec<GpuType>> {
            Ok(self.gpu_types.clone())
        }

        async fn supported_pod_gpu_type_ids(
            &self,
        ) -> anyhow::Result<std::collections::HashSet<String>> {
            Ok(self.supported_ids.clone())
        }

        async fn create_pod(&self, _request: &CreatePodRequest) -> anyhow::Result<Pod> {
            self.create_calls.fetch_add(1, Ordering::SeqCst);
            unreachable!("rejected provider identity must not create a pod")
        }
    }

    fn gpu(id: &str, display_name: &str) -> GpuType {
        GpuType {
            id: Some(id.into()),
            display_name: display_name.into(),
            gpu_id: String::new(),
            memory_in_gb: Some(192),
            secure_cloud: true,
            community_cloud: false,
            stock_status: Some("High".into()),
            available: true,
            secure_price: None,
            community_price: None,
        }
    }

    #[test]
    fn inventory_filter_uses_trimmed_primary_then_alternate_then_display_fallback() {
        let supported = [
            "NVIDIA A100-SXM4-80GB".to_string(),
            "NVIDIA GeForce RTX 5090".to_string(),
        ]
        .into_iter()
        .collect();

        let mut alternate = gpu(" \t ", "Misleading display");
        alternate.gpu_id = "  nvidia a100-sxm4-80gb  ".into();
        assert!(normalize_supported_inventory_gpu(
            &mut alternate,
            &supported
        ));
        assert_eq!(alternate.id.as_deref(), Some("NVIDIA A100-SXM4-80GB"));

        let mut legacy = gpu("", "RTX 5090");
        legacy.id = None;
        assert!(normalize_supported_inventory_gpu(&mut legacy, &supported));
        assert_eq!(legacy.id.as_deref(), Some("NVIDIA GeForce RTX 5090"));
    }

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
        for name in ["NVIDIA B200", "NVIDIA B300"] {
            assert_eq!(
                resolve_published_image_for_gpu(name).await.unwrap(),
                "ghcr.io/utensils/mold:latest-sm100"
            );
        }
    }

    #[tokio::test]
    async fn launch_rejects_untrusted_provider_identity_before_create_pod() {
        let cases = [
            ("", "NVIDIA B200", "Choose a GPU"),
            ("stale-id", "NVIDIA B200", "no longer available"),
            ("b200-id", "", "GPU name is missing"),
            ("b200-id", "NVIDIA GB200", "does not match"),
            ("gb200-id", "NVIDIA B200", "does not match"),
            ("b200-id", "NVIDIA B200", "not currently accepted"),
            ("gh200-id", "NVIDIA GH200", "linux/arm64"),
            ("gb200-id", "NVIDIA GB200", "linux/arm64"),
            ("gb300-id", "NVIDIA GB300", "linux/arm64"),
        ];

        for (submitted_id, submitted_name, expected_error) in cases {
            let client = FakeLaunchClient {
                gpu_types: vec![
                    gpu("b200-id", "NVIDIA B200"),
                    gpu("gh200-id", "NVIDIA GH200"),
                    gpu("gb200-id", "NVIDIA GB200"),
                    gpu("gb300-id", "NVIDIA GB300"),
                ],
                ..Default::default()
            };
            let mut launch_input = input();
            launch_input.gpu_type_id = submitted_id.into();
            launch_input.gpu_display_name = submitted_name.into();

            let error =
                launch_pod_with_image_resolver(&client, launch_input, None, |_name| async {
                    panic!("rejected identity must not resolve an image")
                })
                .await
                .unwrap_err();

            assert!(
                error.contains(expected_error),
                "{submitted_id:?}/{submitted_name:?}: {error}"
            );
            assert_eq!(client.create_calls.load(Ordering::SeqCst), 0);
        }
    }

    #[tokio::test]
    async fn launch_normalizes_provider_id_and_uses_authoritative_b200_identity() {
        let client = FakeLaunchClient {
            gpu_types: vec![gpu("  B200-ID  ", "NVIDIA B200")],
            supported_ids: ["B200-ID".into()].into_iter().collect(),
            ..Default::default()
        };
        let mut launch_input = input();
        launch_input.gpu_type_id = " b200-id ".into();
        launch_input.gpu_display_name = "  NVIDIA   B200 ".into();

        let result = prepare_pod_request_with_image_resolver(&client, launch_input, None, |name| {
            let name = name.to_string();
            async move {
                assert_eq!(name, "B200-ID");
                Ok("ghcr.io/utensils/mold:latest-sm100".into())
            }
        })
        .await
        .unwrap();

        assert_eq!(result.gpu_type_ids, ["B200-ID"]);
        assert_eq!(result.image_name, "ghcr.io/utensils/mold:latest-sm100");
        assert_eq!(client.create_calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn desktop_image_routing_uses_the_same_authoritative_id_as_allocation() {
        let client = FakeLaunchClient {
            gpu_types: vec![gpu("NVIDIA A100 80GB PCIe", "RTX 5090")],
            supported_ids: ["NVIDIA A100 80GB PCIe".into()].into_iter().collect(),
            ..Default::default()
        };
        let mut launch_input = input();
        launch_input.gpu_type_id = "NVIDIA A100 80GB PCIe".into();
        launch_input.gpu_display_name = "RTX 5090".into();

        let result =
            prepare_pod_request_with_image_resolver(&client, launch_input, None, |identity| {
                let identity = identity.to_string();
                async move {
                    assert_eq!(identity, "NVIDIA A100 80GB PCIe");
                    Ok("ghcr.io/utensils/mold:latest-sm80".into())
                }
            })
            .await
            .unwrap();

        assert_eq!(result.gpu_type_ids, ["NVIDIA A100 80GB PCIe"]);
        assert_eq!(result.image_name, "ghcr.io/utensils/mold:latest-sm80");
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
