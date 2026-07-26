//! Lambda Cloud API client and deployment helpers.

use crate::error::MoldError;
use anyhow::Result;
use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};
use std::time::Duration;

pub const DEFAULT_ENDPOINT: &str = "https://cloud.lambda.ai/api/v1";
pub const API_KEY_ENV: &str = "LAMBDA_API_KEY";
pub const DEFAULT_IMAGE_REPOSITORY: &str = "ghcr.io/utensils/mold";

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LambdaSettings {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    #[serde(
        default = "default_endpoint_opt",
        skip_serializing_if = "Option::is_none"
    )]
    pub endpoint: Option<String>,
    #[serde(
        default = "default_image_repository_opt",
        skip_serializing_if = "Option::is_none"
    )]
    pub image_repository: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ssh_key_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ssh_private_key_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filesystem_prefix: Option<String>,
    #[serde(default = "default_filesystem_mount_path")]
    pub filesystem_mount_path: String,
    #[serde(default = "default_confirm_hourly_usd")]
    pub confirm_hourly_usd: f64,
    #[serde(default = "default_local_port")]
    pub local_port: u16,
}

impl Default for LambdaSettings {
    fn default() -> Self {
        Self {
            api_key: None,
            endpoint: default_endpoint_opt(),
            image_repository: default_image_repository_opt(),
            ssh_key_name: None,
            ssh_private_key_path: None,
            filesystem_prefix: None,
            filesystem_mount_path: default_filesystem_mount_path(),
            confirm_hourly_usd: default_confirm_hourly_usd(),
            local_port: default_local_port(),
        }
    }
}

fn default_endpoint_opt() -> Option<String> {
    Some(DEFAULT_ENDPOINT.to_string())
}

fn default_image_repository_opt() -> Option<String> {
    Some(DEFAULT_IMAGE_REPOSITORY.to_string())
}

fn default_filesystem_mount_path() -> String {
    "/data/mold".to_string()
}

fn default_confirm_hourly_usd() -> f64 {
    5.0
}

fn default_local_port() -> u16 {
    7680
}

impl LambdaSettings {
    pub fn resolved_api_key(&self) -> Option<String> {
        std::env::var(API_KEY_ENV)
            .ok()
            .filter(|s| !s.is_empty())
            .or_else(|| self.api_key.clone())
    }

    pub fn endpoint(&self) -> &str {
        self.endpoint.as_deref().unwrap_or(DEFAULT_ENDPOINT)
    }

    pub fn image_repository(&self) -> &str {
        self.image_repository
            .as_deref()
            .unwrap_or(DEFAULT_IMAGE_REPOSITORY)
    }

    pub fn redacted_debug(&self) -> String {
        format!(
            "LambdaSettings {{ api_key: {}, endpoint: {:?}, image_repository: {:?}, \
             ssh_key_name: {:?}, ssh_private_key_path: {:?}, filesystem_prefix: {:?}, \
             filesystem_mount_path: {:?}, confirm_hourly_usd: {}, local_port: {} }}",
            if self.api_key.is_some() {
                "Some(\"<redacted>\")"
            } else {
                "None"
            },
            self.endpoint,
            self.image_repository,
            self.ssh_key_name,
            self.ssh_private_key_path,
            self.filesystem_prefix,
            self.filesystem_mount_path,
            self.confirm_hourly_usd,
            self.local_port,
        )
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ApiList<T> {
    #[serde(default)]
    pub data: Vec<T>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ApiItem<T> {
    pub data: T,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq)]
pub struct Region {
    pub name: String,
    #[serde(default)]
    pub description: String,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq)]
pub struct InstanceTypeSpecs {
    #[serde(default)]
    pub gpus: u32,
    #[serde(default)]
    pub gpu_description: String,
    #[serde(default)]
    pub memory_gib: u32,
    #[serde(default)]
    pub storage_gib: u32,
    #[serde(default)]
    pub vcpus: u32,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq)]
pub struct InstanceType {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub gpu_description: String,
    #[serde(default)]
    pub price_cents_per_hour: u32,
    #[serde(default)]
    pub specs: InstanceTypeSpecs,
    #[serde(default)]
    pub regions_with_capacity_available: Vec<Region>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq)]
pub struct SshKey {
    pub id: String,
    pub name: String,
    pub public_key: String,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq)]
pub struct Filesystem {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub mount_point: String,
    #[serde(default)]
    pub region: Option<Region>,
    #[serde(default)]
    pub bytes_used: Option<u64>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq)]
pub struct Tag {
    pub key: String,
    pub value: String,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq)]
pub struct Instance {
    pub id: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub status: String,
    #[serde(default)]
    pub ip: Option<String>,
    #[serde(default)]
    pub private_ip: Option<String>,
    #[serde(default)]
    pub instance_type: Option<InstanceType>,
    #[serde(default)]
    pub region: Option<Region>,
    #[serde(default)]
    pub ssh_key_names: Vec<String>,
    #[serde(default)]
    pub file_system_names: Vec<String>,
    #[serde(default)]
    pub tags: Vec<Tag>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct CreateSshKeyRequest {
    pub name: String,
    pub public_key: String,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct CreateFilesystemRequest {
    pub name: String,
    pub region: String,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct LaunchInstancesRequest {
    pub region_name: String,
    pub instance_type_name: String,
    pub ssh_key_names: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub file_system_names: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub file_system_mounts: Vec<FilesystemMount>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hostname: Option<String>,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image: Option<LaunchImage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_data: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub tags: Vec<Tag>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq)]
pub struct InstanceLaunchResponse {
    #[serde(default)]
    pub instance_ids: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct InstanceTypesResponse {
    #[serde(default)]
    data: std::collections::BTreeMap<String, InstanceTypeOffering>,
}

#[derive(Debug, Clone, Deserialize)]
struct InstanceTypeOffering {
    instance_type: InstanceType,
    #[serde(default)]
    regions_with_capacity_available: Vec<Region>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FilesystemMount {
    pub mount_point: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_system_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_system_id: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LaunchImage {
    pub id: String,
}

pub struct LaunchRequestInput<'a> {
    pub region_name: &'a str,
    pub instance_type_name: &'a str,
    pub ssh_key_name: &'a str,
    pub filesystem_name: &'a str,
    pub filesystem_id: Option<&'a str>,
    pub filesystem_mount_path: &'a str,
    pub instance_name: &'a str,
    pub image_id: Option<&'a str>,
    pub user_data: &'a str,
}

pub fn build_launch_request(input: LaunchRequestInput<'_>) -> LaunchInstancesRequest {
    LaunchInstancesRequest {
        region_name: input.region_name.to_string(),
        instance_type_name: input.instance_type_name.to_string(),
        ssh_key_names: vec![input.ssh_key_name.to_string()],
        file_system_names: vec![input.filesystem_name.to_string()],
        file_system_mounts: vec![FilesystemMount {
            mount_point: input.filesystem_mount_path.to_string(),
            file_system_name: input
                .filesystem_id
                .is_none()
                .then(|| input.filesystem_name.to_string()),
            file_system_id: input.filesystem_id.map(str::to_string),
        }],
        hostname: None,
        name: input.instance_name.to_string(),
        image: input.image_id.map(|id| LaunchImage { id: id.to_string() }),
        user_data: Some(input.user_data.to_string()),
        tags: vec![Tag {
            key: "managed-by".to_string(),
            value: "mold".to_string(),
        }],
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct AvailabilityRow {
    pub instance_type: String,
    pub region: String,
    pub gpu_description: String,
    pub gpu_count: u32,
    pub generation_slots: u32,
    pub price_per_hour_usd: f64,
    pub memory_gib: u32,
    pub storage_gib: u32,
    pub image: String,
}

impl AvailabilityRow {
    pub fn from_instance_type(
        instance_type: &InstanceType,
        image_repository: &str,
        version: &str,
    ) -> Self {
        let region = instance_type
            .regions_with_capacity_available
            .first()
            .map(|r| r.name.clone())
            .unwrap_or_default();
        let image = if gpu_uses_unsupported_linux_arm64(&instance_type.specs.gpu_description)
            || gpu_uses_unsupported_linux_arm64(&instance_type.name)
        {
            "unsupported: linux/arm64 host".to_string()
        } else {
            let tag = image_tag_for_gpu(&instance_type.specs.gpu_description, version);
            format!("{image_repository}:{tag}")
        };
        Self {
            instance_type: instance_type.name.clone(),
            region,
            gpu_description: instance_type.specs.gpu_description.clone(),
            gpu_count: instance_type.specs.gpus,
            generation_slots: instance_type.specs.gpus,
            price_per_hour_usd: instance_type.price_cents_per_hour as f64 / 100.0,
            memory_gib: instance_type.specs.memory_gib,
            storage_gib: instance_type.specs.storage_gib,
            image,
        }
    }
}

pub fn image_tag_for_gpu(gpu_description: &str, version: &str) -> String {
    crate::cuda_distribution::image_tag_for_gpu_name(gpu_description, version)
}

pub fn gpu_uses_unsupported_linux_arm64(gpu_description: &str) -> bool {
    gpu_description.to_ascii_lowercase().contains("gh200")
}

pub fn filesystem_name(settings: &LambdaSettings, region: &str) -> String {
    let prefix = settings.filesystem_prefix.as_deref().unwrap_or("mold");
    format!("{prefix}-{region}")
}

#[derive(Debug, Clone)]
pub struct CloudInitOptions {
    pub image: String,
    pub mount_path: String,
    pub env_file: String,
}

pub fn render_cloud_init(opts: &CloudInitOptions) -> String {
    format!(
        r#"#cloud-config
write_files:
  - path: /etc/systemd/system/mold-lambda.service
    permissions: '0644'
    content: |
      [Unit]
      Description=mold Lambda container
      After=docker.service network-online.target
      Wants=network-online.target

      [Service]
      Restart=always
      RestartSec=10
      ExecStartPre=-/usr/bin/docker rm -f mold
      ExecStartPre=/usr/bin/docker pull {image}
      ExecStart=/usr/bin/docker run --name mold --gpus all --restart unless-stopped --env-file {env_file} -e MOLD_PORT=7680 -p 127.0.0.1:7680:7680 -v {mount_path}:/workspace {image}
      ExecStop=/usr/bin/docker stop mold

      [Install]
      WantedBy=multi-user.target
runcmd:
  - [ mkdir, -p, /etc/mold ]
  - [ sh, -c, "touch {env_file} && chmod 600 {env_file}" ]
  - [ systemctl, daemon-reload ]
  - [ systemctl, enable, --now, mold-lambda.service ]
"#,
        image = opts.image,
        mount_path = opts.mount_path,
        env_file = opts.env_file,
    )
}

#[derive(Clone)]
pub struct LambdaClient {
    client: Client,
    endpoint: String,
    api_key: String,
}

impl LambdaClient {
    pub fn from_settings(settings: &LambdaSettings) -> Result<Self> {
        let api_key = settings.resolved_api_key().ok_or_else(|| {
            MoldError::Config("missing Lambda API key; set LAMBDA_API_KEY or lambda.api_key".into())
        })?;
        Ok(Self {
            client: Client::builder().timeout(Duration::from_secs(60)).build()?,
            endpoint: settings.endpoint().trim_end_matches('/').to_string(),
            api_key,
        })
    }

    pub fn new(endpoint: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            endpoint: endpoint.into().trim_end_matches('/').to_string(),
            api_key: api_key.into(),
        }
    }

    async fn get_list<T: for<'de> Deserialize<'de> + Default>(&self, path: &str) -> Result<Vec<T>> {
        let resp = self
            .client
            .get(format!("{}{}", self.endpoint, path))
            .basic_auth(&self.api_key, Some(""))
            .send()
            .await?;
        decode_list(resp).await
    }

    async fn post_item<B: Serialize, T: for<'de> Deserialize<'de>>(
        &self,
        path: &str,
        body: &B,
    ) -> Result<T> {
        let resp = self
            .client
            .post(format!("{}{}", self.endpoint, path))
            .basic_auth(&self.api_key, Some(""))
            .json(body)
            .send()
            .await?;
        decode_item(resp).await
    }

    pub async fn list_instance_types(&self) -> Result<Vec<InstanceType>> {
        let resp = self
            .client
            .get(format!("{}/instance-types", self.endpoint))
            .basic_auth(&self.api_key, Some(""))
            .send()
            .await?;
        if !resp.status().is_success() {
            return Err(lambda_error(resp).await.into());
        }
        decode_instance_types_body(&resp.text().await?)
    }

    pub async fn list_instances(&self) -> Result<Vec<Instance>> {
        self.get_list("/instances").await
    }

    pub async fn get_instance(&self, id: &str) -> Result<Instance> {
        let resp = self
            .client
            .get(format!("{}/instances/{id}", self.endpoint))
            .basic_auth(&self.api_key, Some(""))
            .send()
            .await?;
        decode_item(resp).await
    }

    pub async fn launch_instance(
        &self,
        req: &LaunchInstancesRequest,
    ) -> Result<InstanceLaunchResponse> {
        self.post_item("/instance-operations/launch", req).await
    }

    pub async fn terminate_instance(&self, id: &str) -> Result<()> {
        let body = serde_json::json!({ "instance_ids": [id] });
        let resp = self
            .client
            .post(format!("{}/instance-operations/terminate", self.endpoint))
            .basic_auth(&self.api_key, Some(""))
            .json(&body)
            .send()
            .await?;
        ensure_success(resp).await
    }

    pub async fn list_ssh_keys(&self) -> Result<Vec<SshKey>> {
        self.get_list("/ssh-keys").await
    }

    pub async fn create_ssh_key(&self, req: &CreateSshKeyRequest) -> Result<SshKey> {
        self.post_item("/ssh-keys", req).await
    }

    pub async fn list_filesystems(&self) -> Result<Vec<Filesystem>> {
        self.get_list("/file-systems").await
    }

    pub async fn create_filesystem(&self, req: &CreateFilesystemRequest) -> Result<Filesystem> {
        self.post_item("/filesystems", req).await
    }

    pub async fn delete_filesystem(&self, id: &str) -> Result<()> {
        let resp = self
            .client
            .delete(format!("{}/filesystems/{id}", self.endpoint))
            .basic_auth(&self.api_key, Some(""))
            .send()
            .await?;
        ensure_success(resp).await
    }
}

async fn decode_list<T: for<'de> Deserialize<'de> + Default>(
    resp: reqwest::Response,
) -> Result<Vec<T>> {
    if !resp.status().is_success() {
        return Err(lambda_error(resp).await.into());
    }
    Ok(resp.json::<ApiList<T>>().await?.data)
}

async fn decode_item<T: for<'de> Deserialize<'de>>(resp: reqwest::Response) -> Result<T> {
    if !resp.status().is_success() {
        return Err(lambda_error(resp).await.into());
    }
    Ok(resp.json::<ApiItem<T>>().await?.data)
}

async fn ensure_success(resp: reqwest::Response) -> Result<()> {
    if !resp.status().is_success() {
        return Err(lambda_error(resp).await.into());
    }
    Ok(())
}

async fn lambda_error(resp: reqwest::Response) -> MoldError {
    let status = resp.status();
    let body = resp.text().await.unwrap_or_default();
    let message = if status == StatusCode::UNAUTHORIZED {
        "Lambda API authentication failed".to_string()
    } else {
        format!(
            "Lambda API request failed with {status}: {}",
            truncate(&body)
        )
    };
    MoldError::Config(message)
}

fn truncate(s: &str) -> String {
    const MAX: usize = 400;
    if s.chars().count() <= MAX {
        return s.to_string();
    }
    let mut out = s.chars().take(MAX).collect::<String>();
    out.push('…');
    out
}

pub fn decode_instance_types_body(body: &str) -> Result<Vec<InstanceType>> {
    let response: InstanceTypesResponse = serde_json::from_str(body)?;
    Ok(response
        .data
        .into_values()
        .map(|offering| {
            let mut instance_type = offering.instance_type;
            if instance_type.specs.gpu_description.is_empty() {
                instance_type.specs.gpu_description = instance_type.gpu_description.clone();
            }
            instance_type.regions_with_capacity_available =
                offering.regions_with_capacity_available;
            instance_type
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn instance_types_decode_lambda_map_shape() {
        let body = r#"{
          "data": {
            "gpu_1x_a10": {
              "instance_type": {
                "name": "gpu_1x_a10",
                "description": "1x A10",
                "gpu_description": "A10",
                "price_cents_per_hour": 75,
                "specs": {
                  "vcpus": 30,
                  "memory_gib": 200,
                  "storage_gib": 1400,
                  "gpus": 1
                }
              },
              "regions_with_capacity_available": [
                {"name": "us-west-1", "description": "California"}
              ]
            }
          }
        }"#;

        let decoded = decode_instance_types_body(body).unwrap();
        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0].name, "gpu_1x_a10");
        assert_eq!(decoded[0].specs.gpu_description, "A10");
        assert_eq!(
            decoded[0].regions_with_capacity_available[0].name,
            "us-west-1"
        );
    }

    #[test]
    fn lambda_settings_toml_roundtrip_and_defaults() {
        let settings: LambdaSettings = toml::from_str("").unwrap();
        assert_eq!(settings.endpoint.as_deref(), Some(DEFAULT_ENDPOINT));
        assert_eq!(
            settings.image_repository.as_deref(),
            Some(DEFAULT_IMAGE_REPOSITORY)
        );
        assert_eq!(settings.filesystem_mount_path, "/data/mold");
        assert_eq!(settings.confirm_hourly_usd, 5.0);
        assert_eq!(settings.local_port, 7680);

        let original = LambdaSettings {
            api_key: Some("secret".into()),
            endpoint: Some("http://localhost:9999".into()),
            image_repository: Some("ghcr.io/example/mold".into()),
            ssh_key_name: Some("mold-key".into()),
            ssh_private_key_path: Some("~/.ssh/mold_lambda_ed25519".into()),
            filesystem_prefix: Some("mold".into()),
            filesystem_mount_path: "/mnt/mold".into(),
            confirm_hourly_usd: 9.5,
            local_port: 7777,
        };
        let encoded = toml::to_string(&original).unwrap();
        let decoded: LambdaSettings = toml::from_str(&encoded).unwrap();
        assert_eq!(decoded.api_key, original.api_key);
        assert_eq!(decoded.filesystem_mount_path, "/mnt/mold");
        assert_eq!(decoded.local_port, 7777);
    }

    #[test]
    fn auth_prefers_lambda_api_key_env_over_config() {
        let _guard = crate::test_support::ENV_LOCK.lock().unwrap();
        std::env::set_var(API_KEY_ENV, "from-env");
        let settings = LambdaSettings {
            api_key: Some("from-config".into()),
            ..Default::default()
        };
        assert_eq!(settings.resolved_api_key().as_deref(), Some("from-env"));
        std::env::remove_var(API_KEY_ENV);
    }

    #[test]
    fn image_tag_maps_gpu_generations() {
        for name in ["NVIDIA A30", "NVIDIA A100-SXM4-80GB", "Ampere"] {
            assert_eq!(image_tag_for_gpu(name, "0.10.0"), "0.10.0-sm80", "{name}");
        }
        for name in [
            "NVIDIA A10",
            "NVIDIA A40",
            "NVIDIA RTX A6000",
            "NVIDIA A16",
            "NVIDIA A2",
            "NVIDIA RTX 3090",
        ] {
            assert_eq!(image_tag_for_gpu(name, "0.10.0"), "0.10.0-sm86", "{name}");
        }
        assert_eq!(image_tag_for_gpu("NVIDIA L40S", "0.10.0"), "0.10.0");
        for name in ["NVIDIA H100 PCIe", "NVIDIA H200 SXM"] {
            assert_eq!(image_tag_for_gpu(name, "0.10.0"), "0.10.0-sm90", "{name}");
        }
        for name in ["NVIDIA B200", "NVIDIA GB200", "NVIDIA B300", "NVIDIA GB300"] {
            assert_eq!(image_tag_for_gpu(name, "0.10.0"), "0.10.0-sm100", "{name}");
        }
        for name in ["NVIDIA RTX PRO 6000", "NVIDIA GeForce RTX 5090"] {
            assert_eq!(image_tag_for_gpu(name, "v0.10.0"), "0.10.0-sm120", "{name}");
        }
        assert_eq!(
            image_tag_for_gpu("NVIDIA Blackwell", "latest"),
            "latest",
            "generic Blackwell must not guess between incompatible sm100 and sm120"
        );
    }

    #[test]
    fn gh200_is_not_supported_by_published_linux_images() {
        assert!(gpu_uses_unsupported_linux_arm64("GH200 (96 GB)"));
        assert!(gpu_uses_unsupported_linux_arm64("gpu_1x_gh200"));
        assert!(!gpu_uses_unsupported_linux_arm64("NVIDIA H100 PCIe"));
        assert!(!gpu_uses_unsupported_linux_arm64("NVIDIA A100-SXM4-80GB"));
    }

    #[test]
    fn availability_marks_gh200_as_unsupported() {
        let ty = InstanceType {
            name: "gpu_1x_gh200".into(),
            description: "1x GH200".into(),
            gpu_description: "GH200 (96 GB)".into(),
            price_cents_per_hour: 229,
            specs: InstanceTypeSpecs {
                gpus: 1,
                gpu_description: "GH200 (96 GB)".into(),
                memory_gib: 432,
                storage_gib: 4096,
                ..Default::default()
            },
            regions_with_capacity_available: vec![Region {
                name: "us-east-3".into(),
                description: "Austin".into(),
            }],
        };
        let row = AvailabilityRow::from_instance_type(&ty, "ghcr.io/utensils/mold", "0.10.0");
        assert_eq!(row.image, "unsupported: linux/arm64 host");
    }

    #[test]
    fn availability_row_uses_gpu_count_as_generation_slots() {
        let ty = InstanceType {
            name: "gpu_8x_h100".into(),
            description: "8x H100".into(),
            gpu_description: "NVIDIA H100".into(),
            price_cents_per_hour: 15920,
            specs: InstanceTypeSpecs {
                gpus: 8,
                gpu_description: "NVIDIA H100".into(),
                memory_gib: 1800,
                storage_gib: 200,
                ..Default::default()
            },
            regions_with_capacity_available: vec![Region {
                name: "us-east-1".into(),
                description: "Virginia".into(),
            }],
        };
        let row = AvailabilityRow::from_instance_type(&ty, "ghcr.io/utensils/mold", "0.10.0");
        assert_eq!(row.generation_slots, 8);
        assert_eq!(row.image, "ghcr.io/utensils/mold:0.10.0-sm90");
        assert_eq!(row.price_per_hour_usd, 159.20);
    }

    #[test]
    fn filesystem_name_defaults_to_prefix_region() {
        let settings = LambdaSettings::default();
        assert_eq!(filesystem_name(&settings, "us-west-1"), "mold-us-west-1");
        let custom = LambdaSettings {
            filesystem_prefix: Some("team-mold".into()),
            ..Default::default()
        };
        assert_eq!(filesystem_name(&custom, "us-east-1"), "team-mold-us-east-1");
    }

    #[test]
    fn launch_request_contains_expected_shape() {
        let req = build_launch_request(LaunchRequestInput {
            region_name: "us-west-1",
            instance_type_name: "gpu_1x_a10",
            ssh_key_name: "mold-laptop",
            filesystem_name: "mold-us-west-1",
            filesystem_id: None,
            filesystem_mount_path: "/data/mold",
            instance_name: "mold-us-west-1",
            image_id: None,
            user_data: "#cloud-config\n",
        });
        let json = serde_json::to_value(req).unwrap();
        assert_eq!(json["region_name"], "us-west-1");
        assert_eq!(json["ssh_key_names"], serde_json::json!(["mold-laptop"]));
        assert_eq!(
            json["file_system_names"],
            serde_json::json!(["mold-us-west-1"])
        );
        assert_eq!(json["file_system_mounts"][0]["mount_point"], "/data/mold");
        assert_eq!(json["tags"][0]["key"], "managed-by");
        assert_eq!(json["tags"][0]["value"], "mold");
    }

    #[test]
    fn launch_request_uses_filesystem_id_when_available() {
        let req = build_launch_request(LaunchRequestInput {
            region_name: "us-west-1",
            instance_type_name: "gpu_1x_a10",
            ssh_key_name: "mold-laptop",
            filesystem_name: "mold-us-west-1",
            filesystem_id: Some("fs-123"),
            filesystem_mount_path: "/data/mold",
            instance_name: "mold-us-west-1",
            image_id: None,
            user_data: "#cloud-config\n",
        });
        let mount = &req.file_system_mounts[0];
        assert_eq!(mount.file_system_id.as_deref(), Some("fs-123"));
        assert!(mount.file_system_name.is_none());
    }

    #[test]
    fn create_filesystem_request_uses_lambda_region_field() {
        let req = CreateFilesystemRequest {
            name: "mold-us-east-1".into(),
            region: "us-east-1".into(),
        };
        let json = serde_json::to_value(req).unwrap();
        assert_eq!(json["name"], "mold-us-east-1");
        assert_eq!(json["region"], "us-east-1");
        assert!(json.get("region_name").is_none());
    }

    #[test]
    fn cloud_init_keeps_service_private_and_omits_secrets_by_default() {
        let rendered = render_cloud_init(&CloudInitOptions {
            image: "ghcr.io/utensils/mold:0.10.0-sm90".into(),
            mount_path: "/data/mold".into(),
            env_file: "/etc/mold/lambda.env".into(),
        });
        assert!(rendered.contains("-p 127.0.0.1:7680:7680"));
        assert!(rendered.contains("-v /data/mold:/workspace"));
        assert!(rendered.contains("--gpus all"));
        assert!(rendered.contains("ghcr.io/utensils/mold:0.10.0-sm90"));
        assert!(!rendered.contains("HF_TOKEN"));
        assert!(!rendered.contains("CIVITAI_TOKEN"));
    }
}
