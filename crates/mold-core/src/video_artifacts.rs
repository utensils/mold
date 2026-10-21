//! Bounded published-video provenance and immutable archive artifact descriptors.
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct VideoArtifactFile {
    pub sha256: String,
    #[serde(with = "crate::video_job::decimal_u64")]
    #[schema(value_type = String)]
    pub size_bytes: u64,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct VideoPcmArtifact {
    pub file: VideoArtifactFile,
    pub sample_rate: u32,
    pub channels: u16,
    #[serde(with = "crate::video_job::decimal_u64")]
    #[schema(value_type = String)]
    pub frames: u64,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct VideoJobArtifacts {
    pub version: u32,
    /// Opaque private archive namespace, never a path or URL.
    pub bundle_id: String,
    pub provenance: VideoArtifactFile,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pcm: Option<VideoPcmArtifact>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct VideoJobProvenance {
    pub id: String,
    pub manifest_sha256: String,
    #[serde(with = "crate::video_job::decimal_u64")]
    #[schema(value_type = String)]
    pub clip_count: u64,
    #[serde(with = "crate::video_job::decimal_u64")]
    #[schema(value_type = String)]
    pub part_count: u64,
    #[serde(with = "crate::video_job::decimal_u64")]
    #[schema(value_type = String)]
    pub added_frames: u64,
    #[serde(with = "crate::video_job::decimal_u64")]
    #[schema(value_type = String)]
    pub source_frames: u64,
    #[serde(with = "crate::video_job::decimal_u64")]
    #[schema(value_type = String)]
    pub total_frames: u64,
    pub join_mode: crate::video_job::VideoJoinMode,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub authoring: Option<crate::video_job::VideoJobAuthoring>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_archive_identity_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifacts: Option<VideoJobArtifacts>,
}

fn digest(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}
impl VideoJobProvenance {
    pub fn validate(&self) -> Result<(), String> {
        if uuid::Uuid::parse_str(&self.id).is_err()
            || !digest(&self.manifest_sha256)
            || self
                .source_archive_identity_sha256
                .as_ref()
                .is_some_and(|value| !digest(value))
        {
            return Err("invalid video provenance identity".into());
        }
        if self.clip_count == 0
            || self.part_count == 0
            || self.added_frames == 0
            || self.added_frames.checked_add(self.source_frames) != Some(self.total_frames)
        {
            return Err("invalid video provenance frame counts".into());
        }
        if let Some(authoring) = &self.authoring {
            authoring.validate()?;
        }
        if let Some(artifacts) = &self.artifacts {
            artifacts.validate()?;
        }
        Ok(())
    }
}
impl VideoJobArtifacts {
    pub fn validate(&self) -> Result<(), String> {
        if self.version != 1
            || !digest(&self.bundle_id)
            || !digest(&self.provenance.sha256)
            || self.provenance.size_bytes == 0
        {
            return Err("invalid video artifact descriptor".into());
        }
        if let Some(pcm) = &self.pcm {
            let size = pcm
                .frames
                .checked_mul(u64::from(pcm.channels))
                .and_then(|frames| frames.checked_mul(4));
            if !digest(&pcm.file.sha256)
                || pcm.frames == 0
                || pcm.sample_rate == 0
                || pcm.sample_rate > 192000
                || !(1..=2).contains(&pcm.channels)
                || size != Some(pcm.file.size_bytes)
            {
                return Err("invalid exact PCM archive descriptor".into());
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn published_authoring_preserves_literal_scenes_duration_and_full_seed() {
        let raw = serde_json::json!({
            "id":"c7f8554e-53e7-4bd1-a203-346b72032e31", "manifest_sha256":"a".repeat(64),
            "clip_count":"2", "part_count":"2", "added_frames":"144", "source_frames":"720", "total_frames":"864", "join_mode":"continuous",
            "authoring":{"duration_ms":"9007199254740993", "boundary_frames":107, "input_frames":97, "seed":"18446744073709551615", "scenes":[{"prompt":"literal\\n\r\nnext", "duration_ms":"9007199254740993"}]}
        });
        let parsed: VideoJobProvenance = serde_json::from_value(raw.clone()).unwrap();
        parsed.validate().unwrap();
        let encoded = serde_json::to_value(&parsed).unwrap();
        assert_eq!(encoded["authoring"], raw["authoring"]);
    }
    #[test]
    fn long_movie_provenance_stays_scalar_and_pcm_has_no_inline_media_budget() {
        let value:VideoJobProvenance=serde_json::from_value(serde_json::json!({"id":"c7f8554e-53e7-4bd1-a203-346b72032e31","manifest_sha256":"a".repeat(64),"clip_count":"9007199254740993","part_count":"562949953421313","added_frames":"9007199254740993","source_frames":"0","total_frames":"9007199254740993","join_mode":"blended_shots","artifacts":{"version":1,"bundle_id":"b".repeat(64),"provenance":{"sha256":"c".repeat(64),"size_bytes":"1200"},"pcm":{"file":{"sha256":"d".repeat(64),"size_bytes":"2764800000"},"sample_rate":48000,"channels":2,"frames":"345600000"}}})).unwrap();
        value.validate().unwrap();
        assert_eq!(
            serde_json::to_value(&value).unwrap()["clip_count"],
            "9007199254740993"
        );
        let mut invalid = value;
        invalid
            .artifacts
            .as_mut()
            .unwrap()
            .pcm
            .as_mut()
            .unwrap()
            .frames += 1;
        assert!(invalid.validate().is_err());
    }
}
