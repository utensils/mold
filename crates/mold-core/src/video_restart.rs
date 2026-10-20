//! Portable new attempts preserve the original request and prefix, not child progress.
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct VideoRestartOptions {
    pub source_instance_id: String,
    pub source_revision: String,
    pub restart: crate::job_restart::RestartAvailability,
    pub send: crate::job_restart::RestartAvailability,
}

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct VideoRestartExport {
    pub destination_instance_id: String,
    pub source_instance_id: String,
    pub source_revision: String,
    /// Canonical literal JSON. Forward unchanged; never persist media bodies.
    pub request_json: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefix: Option<VideoRestartPrefix>,
}

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct VideoRestartPrefix {
    pub filename: String,
    pub archive_identity_sha256: String,
    pub sha256: String,
    #[serde(with = "crate::video_job::decimal_u64")]
    #[schema(value_type = String)]
    pub size_bytes: u64,
    /// An opaque, source-revision-bound locator. It is never a credential.
    pub source_ref: String,
    /// Completed local retention, issued only when source and destination
    /// are this same installation. Remote destinations must relay bytes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retained_upload: Option<crate::video_job::VideoSourceUploadRef>,
    /// Forward unchanged to upload begin; its metadata can contain full u64s.
    pub upload_spec_json: String,
}

impl crate::MoldClient {
    pub async fn video_restart_options(&self, id: &str) -> anyhow::Result<VideoRestartOptions> {
        let path = crate::client::encode_path_segment(id);
        Ok(serde_json::from_value(
            self.queue_transfer_json(&format!("/api/video-jobs/{path}/restart-options"), None)
                .await?,
        )?)
    }
    pub async fn export_video_restart(
        &self,
        id: &str,
        authority: &crate::job_restart::RestartExportRequest,
    ) -> anyhow::Result<VideoRestartExport> {
        let path = crate::client::encode_path_segment(id);
        Ok(serde_json::from_value(
            self.queue_transfer_json(
                &format!("/api/video-jobs/{path}/restart-export"),
                Some(&serde_json::to_value(authority)?),
            )
            .await?,
        )?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn portable_prefix_keeps_metadata_opaque_and_sizes_lossless() {
        let raw = r#"{"filename":"source.mp4","metadata":{"seed":18446744073709551615,"prompt":"literal\\nscene"}}"#;
        let value:VideoRestartPrefix=serde_json::from_value(serde_json::json!({"filename":"source.mp4","archive_identity_sha256":"a".repeat(64),"sha256":"b".repeat(64),"size_bytes":"9007199254740993","source_ref":"opaque","upload_spec_json":raw})).unwrap();
        assert_eq!(value.upload_spec_json, raw);
        assert_eq!(
            serde_json::to_value(value).unwrap()["size_bytes"],
            "9007199254740993"
        );
    }
}
