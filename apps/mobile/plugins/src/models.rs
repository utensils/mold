use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HostKeyRequest {
    pub host_id: String,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SetApiKeyRequest {
    pub host_id: String,
    pub api_key: String,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GetApiKeyResponse {
    pub api_key: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DiscoveryRequest {
    pub timeout_ms: u32,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DiscoveredHost {
    pub name: String,
    pub host: String,
    pub port: u16,
    pub auth_required: bool,
    pub instance_id: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DiscoveryResponse {
    pub hosts: Vec<DiscoveredHost>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageDataRequest {
    pub data_b64: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct VideoUrlRequest {
    pub url: String,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ShareAnimationRequest {
    pub url: String,
    pub api_key: Option<String>,
    pub request_json: String,
    pub filename: String,
    pub reuse_key: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ShareAnimationResponse {
    pub outcome: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct AppearanceRequest {
    pub appearance: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct IdentityPhotoRequest {
    pub source: String,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct IdentityPhotoResponse {
    pub cancelled: bool,
    pub filename: Option<String>,
    pub mime_type: Option<String>,
    pub size_bytes: Option<u64>,
    pub data_b64: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct PairingScanResponse {
    pub content: String,
}
