//! RunPod REST API client.
//!
//! Wraps `https://rest.runpod.io/v1/` for pod lifecycle management from within
//! the mold CLI. Uses Bearer-token auth via `RUNPOD_API_KEY` env var or an
//! explicit key. All methods are async.

use crate::error::MoldError;
use anyhow::Result;
use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::time::Duration;

/// Default REST base URL.
pub const DEFAULT_ENDPOINT: &str = "https://rest.runpod.io/v1";

/// GraphQL endpoint (used for /user since REST doesn't expose it).
pub const GRAPHQL_ENDPOINT: &str = "https://api.runpod.io/graphql";

/// Environment variable that holds the RunPod API key.
pub const API_KEY_ENV: &str = "RUNPOD_API_KEY";

/// Persisted configuration under `[runpod]` in `config.toml`.
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct RunPodSettings {
    /// API key stored in config. Env var `RUNPOD_API_KEY` takes precedence.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,

    /// Preferred GPU, e.g. `"NVIDIA GeForce RTX 5090"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_gpu: Option<String>,

    /// Preferred datacenter id, e.g. `"EUR-IS-2"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_datacenter: Option<String>,

    /// Attach this network volume to new pods (id from RunPod console).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_network_volume_id: Option<String>,

    /// When `true`, `mold runpod run` deletes the pod after generating.
    /// When `false`, the pod is left running for reuse. Default `false`.
    #[serde(default)]
    pub auto_teardown: bool,

    /// After this many minutes of idle time, background reap deletes the pod.
    /// `0` disables the idle reaper. Default `20`.
    #[serde(default = "default_auto_teardown_idle_mins")]
    pub auto_teardown_idle_mins: u32,

    /// Fail UAT or `run` if cumulative pod spend for the session exceeds this
    /// many USD. `0.0` disables the guard. Default `0.0`.
    #[serde(default)]
    pub cost_alert_usd: f64,

    /// Override the REST endpoint (mostly for testing).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub endpoint: Option<String>,
}

fn default_auto_teardown_idle_mins() -> u32 {
    20
}

/// Redact `api_key` when logging.
impl RunPodSettings {
    pub fn redacted_debug(&self) -> String {
        format!(
            "RunPodSettings {{ api_key: {}, default_gpu: {:?}, default_datacenter: {:?}, \
             default_network_volume_id: {:?}, auto_teardown: {}, auto_teardown_idle_mins: {}, \
             cost_alert_usd: {}, endpoint: {:?} }}",
            if self.api_key.is_some() {
                "Some(\"<redacted>\")"
            } else {
                "None"
            },
            self.default_gpu,
            self.default_datacenter,
            self.default_network_volume_id,
            self.auto_teardown,
            self.auto_teardown_idle_mins,
            self.cost_alert_usd,
            self.endpoint,
        )
    }
}

// ─── API response types ────────────────────────────────────────────────────

/// `GET /user` response.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct UserInfo {
    pub id: String,
    pub email: String,
    #[serde(default)]
    pub client_balance: f64,
    #[serde(default)]
    pub current_spend_per_hr: f64,
    #[serde(default)]
    pub spend_limit: Option<f64>,
}

/// One entry from `GET /gputypes`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct GpuType {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(rename = "displayName", default)]
    pub display_name: String,
    #[serde(rename = "gpuId", default)]
    pub gpu_id: String,
    #[serde(rename = "memoryInGb", default)]
    pub memory_in_gb: u32,
    #[serde(rename = "secureCloud", default)]
    pub secure_cloud: bool,
    #[serde(rename = "communityCloud", default)]
    pub community_cloud: bool,
    #[serde(rename = "stockStatus", default)]
    pub stock_status: Option<String>,
    #[serde(default)]
    pub available: bool,
}

/// One entry from `GET /datacenters`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Datacenter {
    pub id: String,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub location: Option<String>,
    #[serde(rename = "gpuAvailability", default)]
    pub gpu_availability: Vec<GpuAvailability>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct GpuAvailability {
    #[serde(rename = "displayName", default)]
    pub display_name: String,
    #[serde(rename = "gpuId", default)]
    pub gpu_id: String,
    #[serde(rename = "stockStatus", default)]
    pub stock_status: Option<String>,
}

/// `GET /pods` / `GET /pods/{id}` response.
///
/// Only fields we actually use are deserialized — anything else is allowed via
/// `#[serde(default)]` on the struct to avoid breaking on RunPod API drift.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Pod {
    pub id: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(rename = "desiredStatus", default)]
    pub desired_status: String,
    #[serde(rename = "imageName", default)]
    pub image_name: Option<String>,
    #[serde(rename = "gpuCount", default)]
    pub gpu_count: u32,
    #[serde(rename = "costPerHr", default)]
    pub cost_per_hr: f64,
    #[serde(rename = "uptimeSeconds", default)]
    pub uptime_seconds: u64,
    #[serde(rename = "lastStatusChange", default)]
    pub last_status_change: Option<String>,
    #[serde(rename = "memoryInGb", default)]
    pub memory_in_gb: u32,
    #[serde(rename = "vcpuCount", default)]
    pub vcpu_count: u32,
    #[serde(rename = "volumeInGb", default)]
    pub volume_in_gb: u32,
    #[serde(rename = "volumeMountPath", default)]
    pub volume_mount_path: Option<String>,
    #[serde(default)]
    pub ports: serde_json::Value,
    #[serde(default)]
    pub env: serde_json::Value,
    #[serde(default)]
    pub machine: Option<PodMachine>,
    #[serde(default)]
    pub runtime: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PodMachine {
    #[serde(rename = "gpuDisplayName", default)]
    pub gpu_display_name: Option<String>,
    #[serde(default)]
    pub location: Option<String>,
}

/// Body for `POST /pods`.
#[derive(Debug, Clone, Serialize, Default)]
pub struct CreatePodRequest {
    pub name: String,
    #[serde(rename = "imageName")]
    pub image_name: String,
    #[serde(rename = "gpuTypeIds")]
    pub gpu_type_ids: Vec<String>,
    #[serde(rename = "cloudType")]
    pub cloud_type: String,
    #[serde(rename = "dataCenterIds", skip_serializing_if = "Option::is_none")]
    pub data_center_ids: Option<Vec<String>>,
    #[serde(rename = "gpuCount")]
    pub gpu_count: u32,
    #[serde(rename = "containerDiskInGb")]
    pub container_disk_in_gb: u32,
    #[serde(rename = "volumeInGb")]
    pub volume_in_gb: u32,
    #[serde(rename = "volumeMountPath")]
    pub volume_mount_path: String,
    pub ports: Vec<String>,
    pub env: serde_json::Map<String, serde_json::Value>,
    #[serde(rename = "networkVolumeId", skip_serializing_if = "Option::is_none")]
    pub network_volume_id: Option<String>,
}

/// One entry from `GET /networkvolumes`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct NetworkVolume {
    pub id: String,
    pub name: String,
    #[serde(rename = "dataCenterId", default)]
    pub data_center_id: String,
    pub size: u32,
}

// ─── Client ────────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct RunPodClient {
    endpoint: String,
    api_key: String,
    http: Client,
}

impl fmt::Debug for RunPodClient {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RunPodClient")
            .field("endpoint", &self.endpoint)
            .field("api_key", &"<redacted>")
            .finish()
    }
}

impl RunPodClient {
    /// Construct with explicit endpoint + key.
    pub fn new(endpoint: impl Into<String>, api_key: impl Into<String>) -> Self {
        let http = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .unwrap_or_default();
        Self {
            endpoint: endpoint.into(),
            api_key: api_key.into(),
            http,
        }
    }

    /// Construct from config + environment. `RUNPOD_API_KEY` overrides
    /// `settings.api_key`. Returns `RunPodAuth` error if no key is available.
    pub fn from_settings(settings: &RunPodSettings) -> std::result::Result<Self, MoldError> {
        let key = std::env::var(API_KEY_ENV)
            .ok()
            .filter(|k| !k.is_empty())
            .or_else(|| settings.api_key.clone())
            .ok_or_else(|| {
                MoldError::RunPodAuth(format!(
                    "RunPod API key not set — export {API_KEY_ENV} or run \
                     `mold config set runpod.api_key <key>`"
                ))
            })?;
        let endpoint = settings
            .endpoint
            .clone()
            .unwrap_or_else(|| DEFAULT_ENDPOINT.to_string());
        Ok(Self::new(endpoint, key))
    }

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.endpoint.trim_end_matches('/'), path)
    }

    async fn get_json<T: for<'de> Deserialize<'de>>(&self, path: &str) -> Result<T> {
        let resp = self
            .http
            .get(self.url(path))
            .bearer_auth(&self.api_key)
            .send()
            .await
            .map_err(|e| MoldError::RunPod(format!("RunPod {path}: {e}")))?;
        let status = resp.status();
        if status.is_success() {
            let body = resp
                .text()
                .await
                .map_err(|e| MoldError::RunPod(format!("RunPod {path} body: {e}")))?;
            serde_json::from_str(&body).map_err(|e| {
                MoldError::RunPod(format!(
                    "RunPod {path}: failed to parse response: {e} — body: {}",
                    truncate_for_error(&body)
                ))
                .into()
            })
        } else {
            Err(http_error(path, status, resp).await.into())
        }
    }

    async fn post_json<B: Serialize, T: for<'de> Deserialize<'de>>(
        &self,
        path: &str,
        body: &B,
    ) -> Result<T> {
        let resp = self
            .http
            .post(self.url(path))
            .bearer_auth(&self.api_key)
            .json(body)
            .send()
            .await
            .map_err(|e| MoldError::RunPod(format!("RunPod {path}: {e}")))?;
        let status = resp.status();
        if status.is_success() {
            let text = resp
                .text()
                .await
                .map_err(|e| MoldError::RunPod(format!("RunPod {path} body: {e}")))?;
            serde_json::from_str(&text).map_err(|e| {
                MoldError::RunPod(format!(
                    "RunPod {path}: failed to parse response: {e} — body: {}",
                    truncate_for_error(&text)
                ))
                .into()
            })
        } else {
            Err(http_error(path, status, resp).await.into())
        }
    }

    async fn post_empty(&self, path: &str) -> Result<()> {
        let resp = self
            .http
            .post(self.url(path))
            .bearer_auth(&self.api_key)
            .send()
            .await
            .map_err(|e| MoldError::RunPod(format!("RunPod {path}: {e}")))?;
        let status = resp.status();
        if status.is_success() {
            Ok(())
        } else {
            Err(http_error(path, status, resp).await.into())
        }
    }

    async fn delete(&self, path: &str) -> Result<()> {
        let resp = self
            .http
            .delete(self.url(path))
            .bearer_auth(&self.api_key)
            .send()
            .await
            .map_err(|e| MoldError::RunPod(format!("RunPod {path}: {e}")))?;
        let status = resp.status();
        if status.is_success() {
            Ok(())
        } else {
            Err(http_error(path, status, resp).await.into())
        }
    }

    async fn get_text(&self, path: &str) -> Result<String> {
        let resp = self
            .http
            .get(self.url(path))
            .bearer_auth(&self.api_key)
            .send()
            .await
            .map_err(|e| MoldError::RunPod(format!("RunPod {path}: {e}")))?;
        let status = resp.status();
        if status.is_success() {
            Ok(resp
                .text()
                .await
                .map_err(|e| MoldError::RunPod(format!("RunPod {path} body: {e}")))?)
        } else {
            Err(http_error(path, status, resp).await.into())
        }
    }

    // ─── Typed endpoints ────────────────────────────────────────────

    /// User/account info isn't exposed by the REST API, so we fall back to
    /// the GraphQL endpoint (same API key works for both).
    pub async fn user(&self) -> Result<UserInfo> {
        let query = serde_json::json!({
            "query": "query { myself { id email clientBalance currentSpendPerHr spendLimit } }"
        });
        let resp = self
            .http
            .post(GRAPHQL_ENDPOINT)
            .bearer_auth(&self.api_key)
            .json(&query)
            .send()
            .await
            .map_err(|e| MoldError::RunPod(format!("RunPod graphql /user: {e}")))?;
        let status = resp.status();
        if !status.is_success() {
            return Err(http_error("graphql /user", status, resp).await.into());
        }
        let body: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| MoldError::RunPod(format!("RunPod graphql /user json: {e}")))?;
        if let Some(errs) = body.get("errors") {
            return Err(MoldError::RunPod(format!("RunPod graphql errors: {errs}")).into());
        }
        let myself = body
            .get("data")
            .and_then(|d| d.get("myself"))
            .ok_or_else(|| MoldError::RunPod("graphql: missing data.myself".into()))?;
        let info = UserInfo {
            id: myself
                .get("id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            email: myself
                .get("email")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            client_balance: myself
                .get("clientBalance")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0),
            current_spend_per_hr: myself
                .get("currentSpendPerHr")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0),
            spend_limit: myself.get("spendLimit").and_then(|v| v.as_f64()),
        };
        Ok(info)
    }

    /// Query GPU types via GraphQL (not exposed in REST v1).
    /// Stock status is aggregated: the highest stock level across all DCs.
    pub async fn gpu_types(&self) -> Result<Vec<GpuType>> {
        let query = serde_json::json!({
            "query": "query { gpuTypes { id displayName memoryInGb secureCloud communityCloud } dataCenters { gpuAvailability { displayName stockStatus } } }"
        });
        let body = self.graphql(&query).await?;
        let data = body
            .get("data")
            .ok_or_else(|| MoldError::RunPod("graphql: missing data".into()))?;
        let types: Vec<GpuType> = serde_json::from_value(
            data.get("gpuTypes")
                .cloned()
                .unwrap_or(serde_json::Value::Array(vec![])),
        )
        .map_err(|e| MoldError::RunPod(format!("parse gpuTypes: {e}")))?;
        // Aggregate stock across datacenters.
        let mut best_stock: std::collections::HashMap<String, String> =
            std::collections::HashMap::new();
        if let Some(dcs) = data.get("dataCenters").and_then(|v| v.as_array()) {
            for dc in dcs {
                if let Some(avail) = dc.get("gpuAvailability").and_then(|v| v.as_array()) {
                    for a in avail {
                        if let (Some(name), Some(stock)) = (
                            a.get("displayName").and_then(|v| v.as_str()),
                            a.get("stockStatus").and_then(|v| v.as_str()),
                        ) {
                            let current = best_stock.get(name).cloned().unwrap_or_default();
                            if stock_rank(stock) > stock_rank(&current) {
                                best_stock.insert(name.to_string(), stock.to_string());
                            }
                        }
                    }
                }
            }
        }
        let mut out = types;
        for g in out.iter_mut() {
            if let Some(s) = best_stock.get(&g.display_name) {
                if !s.is_empty() {
                    g.stock_status = Some(s.clone());
                }
            }
            g.available = g.stock_status.as_deref().is_some_and(|s| s != "None");
        }
        Ok(out)
    }

    /// Query datacenters with per-GPU availability via GraphQL.
    pub async fn datacenters(&self) -> Result<Vec<Datacenter>> {
        let query = serde_json::json!({
            "query": "query { dataCenters { id name listed gpuAvailability { id displayName stockStatus } } }"
        });
        let body = self.graphql(&query).await?;
        let arr = body
            .get("data")
            .and_then(|d| d.get("dataCenters"))
            .cloned()
            .unwrap_or(serde_json::Value::Array(vec![]));
        // Map GraphQL `id` → `gpuId` so we can reuse the same Datacenter type.
        let arr = match arr {
            serde_json::Value::Array(mut dcs) => {
                for dc in dcs.iter_mut() {
                    if let Some(avail) =
                        dc.get_mut("gpuAvailability").and_then(|v| v.as_array_mut())
                    {
                        for a in avail.iter_mut() {
                            if let Some(id) = a.get("id").and_then(|v| v.as_str()) {
                                let id = id.to_string();
                                if let Some(obj) = a.as_object_mut() {
                                    obj.insert("gpuId".into(), serde_json::Value::String(id));
                                }
                            }
                        }
                    }
                }
                serde_json::Value::Array(dcs)
            }
            other => other,
        };
        let dcs: Vec<Datacenter> = serde_json::from_value(arr)
            .map_err(|e| MoldError::RunPod(format!("parse dataCenters: {e}")))?;
        Ok(dcs)
    }

    async fn graphql(&self, query: &serde_json::Value) -> Result<serde_json::Value> {
        let resp = self
            .http
            .post(GRAPHQL_ENDPOINT)
            .bearer_auth(&self.api_key)
            .json(query)
            .send()
            .await
            .map_err(|e| MoldError::RunPod(format!("RunPod graphql: {e}")))?;
        let status = resp.status();
        if !status.is_success() {
            return Err(http_error("graphql", status, resp).await.into());
        }
        let body: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| MoldError::RunPod(format!("graphql body: {e}")))?;
        if let Some(errs) = body
            .get("errors")
            .filter(|e| !e.as_array().map(|a| a.is_empty()).unwrap_or(true))
        {
            return Err(MoldError::RunPod(format!("graphql errors: {errs}")).into());
        }
        Ok(body)
    }

    pub async fn list_pods(&self) -> Result<Vec<Pod>> {
        self.get_json("/pods").await
    }

    pub async fn get_pod(&self, id: &str) -> Result<Pod> {
        self.get_json(&format!("/pods/{id}")).await
    }

    pub async fn create_pod(&self, req: &CreatePodRequest) -> Result<Pod> {
        self.post_json("/pods", req).await
    }

    pub async fn stop_pod(&self, id: &str) -> Result<()> {
        self.post_empty(&format!("/pods/{id}/stop")).await
    }

    pub async fn start_pod(&self, id: &str) -> Result<()> {
        self.post_empty(&format!("/pods/{id}/start")).await
    }

    pub async fn delete_pod(&self, id: &str) -> Result<()> {
        self.delete(&format!("/pods/{id}")).await
    }

    pub async fn pod_logs(&self, id: &str) -> Result<String> {
        self.get_text(&format!("/pods/{id}/logs")).await
    }

    pub async fn network_volumes(&self) -> Result<Vec<NetworkVolume>> {
        self.get_json("/networkvolumes").await
    }
}

// ─── Helpers ────────────────────────────────────────────────────────────────

async fn http_error(path: &str, status: StatusCode, resp: reqwest::Response) -> MoldError {
    let body = resp.text().await.unwrap_or_default();
    let msg = truncate_for_error(&body);
    match status {
        StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => {
            MoldError::RunPodAuth(format!("RunPod {path} {status}: {msg}"))
        }
        StatusCode::NOT_FOUND => {
            MoldError::RunPodNotFound(format!("RunPod {path} {status}: {msg}"))
        }
        StatusCode::CONFLICT | StatusCode::SERVICE_UNAVAILABLE
            if msg.to_lowercase().contains("does not have the resources") =>
        {
            MoldError::RunPodNoStock(format!("RunPod {path} {status}: {msg}"))
        }
        _ => MoldError::RunPod(format!("RunPod {path} {status}: {msg}")),
    }
}

fn stock_rank(s: &str) -> u8 {
    match s {
        "High" => 3,
        "Medium" => 2,
        "Low" => 1,
        _ => 0,
    }
}

fn truncate_for_error(s: &str) -> String {
    const MAX: usize = 400;
    let s = s.trim();
    if s.len() <= MAX {
        s.to_string()
    } else {
        format!("{}…", &s[..MAX])
    }
}

/// Map a RunPod GPU `displayName` (e.g. `"RTX 4090"`) to the matching
/// `ghcr.io/utensils/mold` image tag.
pub fn image_tag_for_gpu(display_name: &str) -> &'static str {
    let d = display_name.to_lowercase();
    if d.contains("5090") || d.contains("blackwell") || d.contains("b200") {
        "latest-sm120"
    } else if d.contains("a100") || d.contains("3090") || d.contains("a40") || d.contains("ampere")
    {
        "latest-sm80"
    } else {
        // Ada (4090, L40, L40S) and fallback
        "latest"
    }
}

/// Ranked preference when auto-picking GPUs. Higher index = more preferred.
pub const GPU_PREFERENCE: &[&str] = &[
    "A100 PCIe",
    "L40",
    "L40S",
    "RTX A6000",
    "RTX 5090",
    "RTX 4090",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_tag_mapping() {
        assert_eq!(image_tag_for_gpu("RTX 4090"), "latest");
        assert_eq!(image_tag_for_gpu("NVIDIA GeForce RTX 4090"), "latest");
        assert_eq!(image_tag_for_gpu("L40S"), "latest");
        assert_eq!(image_tag_for_gpu("RTX 5090"), "latest-sm120");
        assert_eq!(image_tag_for_gpu("NVIDIA GeForce RTX 5090"), "latest-sm120");
        assert_eq!(image_tag_for_gpu("A100 80GB"), "latest-sm80");
        assert_eq!(image_tag_for_gpu("A100 PCIe"), "latest-sm80");
        assert_eq!(image_tag_for_gpu("RTX 3090"), "latest-sm80");
    }

    #[test]
    fn redacted_debug_hides_api_key() {
        let s = RunPodSettings {
            api_key: Some("secret-key".to_string()),
            ..Default::default()
        };
        let out = s.redacted_debug();
        assert!(!out.contains("secret-key"));
        assert!(out.contains("<redacted>"));
    }

    #[test]
    fn from_settings_requires_key() {
        std::env::remove_var(API_KEY_ENV);
        let err = RunPodClient::from_settings(&RunPodSettings::default()).unwrap_err();
        assert!(matches!(err, MoldError::RunPodAuth(_)));
    }

    #[test]
    fn truncate_for_error_boundary() {
        let short = "short";
        assert_eq!(truncate_for_error(short), "short");
        let long = "x".repeat(500);
        let truncated = truncate_for_error(&long);
        assert!(truncated.ends_with('…'));
        assert!(truncated.chars().count() <= 401);
    }

    #[test]
    fn runpod_settings_toml_roundtrip() {
        let original = RunPodSettings {
            api_key: Some("k".to_string()),
            default_gpu: Some("RTX 5090".to_string()),
            default_datacenter: Some("EUR-IS-2".to_string()),
            default_network_volume_id: Some("nv-123".to_string()),
            auto_teardown: true,
            auto_teardown_idle_mins: 30,
            cost_alert_usd: 3.5,
            endpoint: None,
        };
        let toml_s = toml::to_string(&original).unwrap();
        let round: RunPodSettings = toml::from_str(&toml_s).unwrap();
        assert_eq!(round.api_key, original.api_key);
        assert_eq!(round.default_gpu, original.default_gpu);
        assert_eq!(round.default_datacenter, original.default_datacenter);
        assert_eq!(
            round.default_network_volume_id,
            original.default_network_volume_id
        );
        assert_eq!(round.auto_teardown, original.auto_teardown);
        assert_eq!(
            round.auto_teardown_idle_mins,
            original.auto_teardown_idle_mins
        );
        assert_eq!(round.cost_alert_usd, original.cost_alert_usd);
    }

    #[test]
    fn default_auto_teardown_idle_mins_is_20() {
        let s: RunPodSettings = toml::from_str("").unwrap();
        assert_eq!(s.auto_teardown_idle_mins, 20);
    }
}
