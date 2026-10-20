//! Durable arbitrary-duration video parent contracts.

use serde::{Deserialize, Deserializer, Serialize};

/// A new protocol: prompt strings have ordinary literal JSON semantics.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct VideoJobRequest {
    #[serde(deserialize_with = "deserialize_literal_generation")]
    pub generation: crate::GenerateRequest,
    /// Generated time, or added time when extending a source.
    #[serde(with = "decimal_u64")]
    #[schema(value_type = String)]
    pub duration_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<VideoJobSource>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scenes: Option<Vec<VideoJobScene>>,
    /// Read-only validation proof, required by creation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub validation_revision: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct VideoJobSource {
    pub filename: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_archive_identity_sha256: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct VideoJobScene {
    pub prompt: String,
    #[serde(with = "decimal_u64")]
    #[schema(value_type = String)]
    pub duration_ms: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum VideoJoinMode {
    Continuous,
    BlendedShots,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct VideoJobPlan {
    pub validation_revision: String,
    pub model: String,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub single_clip_max_frames: u32,
    pub overlap_frames: u32,
    pub join_mode: VideoJoinMode,
    #[serde(with = "decimal_u64")]
    #[schema(value_type = String)]
    pub clip_count: u64,
    #[serde(with = "decimal_u64")]
    #[schema(value_type = String)]
    pub part_count: u64,
    #[serde(with = "decimal_u64")]
    #[schema(value_type = String)]
    pub added_frames: u64,
    #[serde(with = "decimal_u64")]
    #[schema(value_type = String)]
    pub source_frames: u64,
    #[serde(with = "decimal_u64")]
    #[schema(value_type = String)]
    pub total_frames: u64,
    #[serde(with = "decimal_u64")]
    #[schema(value_type = String)]
    pub added_duration_ms: u64,
    #[serde(with = "decimal_u64")]
    #[schema(value_type = String)]
    pub total_duration_ms: u64,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum VideoJobState {
    Queued,
    Running,
    Paused,
    Completed,
    Failed,
    Interrupted,
    Cancelled,
}

impl VideoJobState {
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Completed | Self::Failed | Self::Interrupted | Self::Cancelled
        )
    }
    pub fn can_cancel(self) -> bool {
        matches!(self, Self::Queued | Self::Running | Self::Paused)
    }
    pub fn can_resume(self) -> bool {
        matches!(self, Self::Paused | Self::Interrupted | Self::Failed)
    }
}

/// Public projection; retained media and private manifests never ride status/SSE.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct VideoJob {
    pub id: String,
    pub state: VideoJobState,
    pub plan: VideoJobPlan,
    #[serde(with = "decimal_u64")]
    #[schema(value_type = String)]
    pub completed_clips: u64,
    pub active_child_id: Option<String>,
    pub output_filename: Option<String>,
    pub error: Option<String>,
    pub error_detail: Option<String>,
    pub created_at_ms: i64,
    pub updated_at_ms: i64,
    pub can_cancel: bool,
    pub can_resume: bool,
    pub can_delete: bool,
}

impl VideoJobRequest {
    pub fn validate_shape(&self) -> Result<(), String> {
        if self.duration_ms == 0 {
            return Err("duration_ms must be greater than zero".into());
        }
        if let Some(source) = &self.source {
            let name = &source.filename;
            if name.is_empty()
                || name.trim() != name
                || name.contains(['/', '\\'])
                || name == "."
                || name == ".."
                || name.chars().any(char::is_control)
            {
                return Err("source must be one exact Library filename".into());
            }
            if source
                .expected_archive_identity_sha256
                .as_ref()
                .is_some_and(|hash| {
                    hash.len() != 64 || !hash.bytes().all(|b| b.is_ascii_hexdigit())
                })
            {
                return Err("source archive identity must be a SHA-256 digest".into());
            }
        }
        if let Some(scenes) = &self.scenes {
            if scenes.is_empty() || scenes.len() > 4096 {
                return Err("scenes must contain 1 to 4096 entries".into());
            }
            let sum = scenes.iter().try_fold(0u64, |sum, scene| {
                if scene.duration_ms == 0
                    || scene.prompt.trim().is_empty()
                    || scene.prompt.len() > 77000
                {
                    return None;
                }
                sum.checked_add(scene.duration_ms)
            });
            if sum != Some(self.duration_ms) {
                return Err("scene durations must be positive and sum exactly to duration_ms; prompts must be nonempty and at most 77000 bytes".into());
            }
        }
        Ok(())
    }
}

// The legacy GenerateRequest reader normalizes escaped newlines. Restore only
// prompt fields from this new protocol's literal JSON after typed validation.
fn deserialize_literal_generation<'de, D: Deserializer<'de>>(
    d: D,
) -> Result<crate::GenerateRequest, D::Error> {
    let value = serde_json::Value::deserialize(d)?;
    let mut generation: crate::GenerateRequest =
        serde_json::from_value(value.clone()).map_err(serde::de::Error::custom)?;
    let string = |key: &str| {
        value
            .get(key)
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned)
    };
    generation.prompt = string("prompt")
        .ok_or_else(|| serde::de::Error::custom("generation.prompt must be a string"))?;
    generation.negative_prompt = string("negative_prompt");
    generation.original_prompt = string("original_prompt");
    if let (Some(transform), Some(raw)) = (
        generation.prompt_transform.as_mut(),
        value.get("prompt_transform"),
    ) {
        transform.root_prompt = raw
            .get("root_prompt")
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned);
        transform.source_prompt = raw
            .get("source_prompt")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| serde::de::Error::custom("source_prompt must be a string"))?
            .to_owned();
    }
    Ok(generation)
}

/// JSON integers are accepted for callers; output is always lossless in JS.
pub mod decimal_u64 {
    use serde::{Deserialize, Deserializer, Serializer};
    pub fn serialize<S: Serializer>(value: &u64, s: S) -> Result<S::Ok, S::Error> {
        s.collect_str(value)
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<u64, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Value {
            Integer(u64),
            Decimal(String),
        }
        match Value::deserialize(d)? {
            Value::Integer(v) => Ok(v),
            Value::Decimal(v) if !v.is_empty() && v.bytes().all(|b| b.is_ascii_digit()) => {
                v.parse().map_err(serde::de::Error::custom)
            }
            _ => Err(serde::de::Error::custom(
                "expected unsigned integer or decimal string",
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(duration: serde_json::Value) -> VideoJobRequest {
        serde_json::from_value(serde_json::json!({
            "generation": {"width":512,"height":512,"steps":4,"model":"ltx-2-19b-distilled:fp8", "prompt":r"a sign \new"},
            "duration_ms":duration
        })).unwrap()
    }

    #[test]
    fn new_protocol_preserves_literal_prompt_bytes_and_full_range_seed() {
        let value = serde_json::json!({"generation":{
            "width":512,"height":512,"steps":4,"model":"ltx-2-19b-distilled:fp8", "prompt":"one\\new\r\ntwo",
            "negative_prompt":r"blur\n", "original_prompt":r"root\r", "seed":u64::MAX
        },"duration_ms":"18446744073709551615"});
        let parsed: VideoJobRequest = serde_json::from_value(value.clone()).unwrap();
        assert_eq!(
            parsed.generation.prompt,
            value["generation"]["prompt"].as_str().unwrap()
        );
        assert_eq!(
            parsed.generation.negative_prompt.as_deref(),
            Some(r"blur\n")
        );
        assert_eq!(
            parsed.generation.original_prompt.as_deref(),
            Some(r"root\r")
        );
        assert_eq!(parsed.generation.seed, Some(u64::MAX));
        let again: VideoJobRequest =
            serde_json::from_str(&serde_json::to_string(&parsed).unwrap()).unwrap();
        assert_eq!(again.generation.prompt, parsed.generation.prompt);
        assert_eq!(again.duration_ms, u64::MAX);
    }

    #[test]
    fn durations_accept_integer_or_decimal_string_and_emit_strings() {
        for value in [serde_json::json!(30000), serde_json::json!("30000")] {
            let parsed = request(value);
            assert_eq!(parsed.duration_ms, 30000);
            assert_eq!(
                serde_json::to_value(parsed).unwrap()["duration_ms"],
                "30000"
            );
        }
        for invalid in [
            serde_json::json!(-1),
            serde_json::json!(1.5),
            serde_json::json!("+1"),
            serde_json::json!(" 1"),
        ] {
            assert!(serde_json::from_value::<VideoJobRequest>(
                serde_json::json!({"generation":{"model":"x","prompt":"x"},"duration_ms":invalid})
            )
            .is_err());
        }
    }

    #[test]
    fn scenes_are_optional_but_sum_exactly_without_overflow() {
        let mut req = request(serde_json::json!(30000));
        assert!(req.validate_shape().is_ok());
        req.scenes = Some(vec![VideoJobScene {
            prompt: "next".into(),
            duration_ms: 30000,
        }]);
        assert!(req.validate_shape().is_ok());
        req.scenes.as_mut().unwrap()[0].duration_ms = 29999;
        assert!(req.validate_shape().is_err());
        req.scenes = Some(vec![]);
        assert!(req.validate_shape().is_err());
        req.duration_ms = 0;
        req.scenes = None;
        assert!(req.validate_shape().is_err());
    }

    #[test]
    fn library_source_is_an_exact_filename_and_hash_not_a_path() {
        let mut req = request(serde_json::json!(30000));
        for name in ["../a.mp4", "a/b.mp4", "a\\b.mp4", " a.mp4"] {
            req.source = Some(VideoJobSource {
                filename: name.into(),
                expected_archive_identity_sha256: None,
            });
            assert!(req.validate_shape().is_err());
        }
        req.source = Some(VideoJobSource {
            filename: "a.mp4".into(),
            expected_archive_identity_sha256: Some("a".repeat(64)),
        });
        assert!(req.validate_shape().is_ok());
        req.source
            .as_mut()
            .unwrap()
            .expected_archive_identity_sha256 = Some("not a digest".into());
        assert!(req.validate_shape().is_err());
    }
}
