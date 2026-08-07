use serde::Deserialize;
use thiserror::Error;

pub const H3_SELECTED_LANGUAGE_LAYERS: usize = 50;
pub const H3_FULL_LANGUAGE_LAYERS: usize = 64;

/// Tensor payload of the full BF16 Qwen3-VL-32B checkpoint.
pub const H3_FULL_CHECKPOINT_BYTES: u64 = 66_714_780_128;
/// BF16 tensor payload materialized by the layer-50 H3 conditioner.
pub const H3_BF16_PARAMETER_BYTES: u64 = 51_506_191_840;
/// Size of Comfy's pruned single-file safetensors object, including its header.
pub const H3_COMFY_SAFETENSORS_BYTES: u64 = 51_506_295_256;

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct H3ConditionerConfig {
    pub architectures: Vec<String>,
    pub image_token_id: u32,
    pub video_token_id: u32,
    pub vision_start_token_id: u32,
    pub vision_end_token_id: u32,
    pub text_config: H3TextConfig,
    pub vision_config: H3VisionConfig,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct H3TextConfig {
    pub model_type: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub max_position_embeddings: usize,
    pub hidden_act: String,
    pub attention_bias: bool,
    pub rope_scaling: H3RopeScaling,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct H3RopeScaling {
    pub rope_type: String,
    pub mrope_interleaved: bool,
    pub mrope_section: [usize; 3],
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct H3VisionConfig {
    pub model_type: String,
    pub depth: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub in_channels: usize,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub spatial_merge_size: usize,
    pub out_hidden_size: usize,
    pub num_position_embeddings: usize,
    pub deepstack_visual_indexes: Vec<usize>,
    pub hidden_act: String,
}

#[derive(Debug, Error, PartialEq)]
pub enum ConfigError {
    #[error("invalid MiniMax H3 Qwen3-VL config JSON: {0}")]
    Json(String),
    #[error("wrong conditioner variant: {0}")]
    WrongVariant(String),
}

impl H3ConditionerConfig {
    pub fn from_json(bytes: &[u8]) -> Result<Self, ConfigError> {
        let config: Self =
            serde_json::from_slice(bytes).map_err(|error| ConfigError::Json(error.to_string()))?;
        config.validate()?;
        Ok(config)
    }

    /// Validate the exact released Qwen3-VL-32B architecture before any
    /// checkpoint tensor is mapped or allocated.
    pub fn validate(&self) -> Result<(), ConfigError> {
        let wrong = |field: &str, found: String, expected: &str| {
            ConfigError::WrongVariant(format!("{field} is {found}, expected {expected}"))
        };
        macro_rules! exact {
            ($field:expr, $found:expr, $expected:expr) => {
                if $found != $expected {
                    return Err(wrong(
                        $field,
                        format!("{:?}", $found),
                        &format!("{:?}", $expected),
                    ));
                }
            };
        }

        exact!(
            "architectures",
            self.architectures.as_slice(),
            ["Qwen3VLForConditionalGeneration"]
        );
        exact!("image_token_id", self.image_token_id, 151_655);
        exact!("video_token_id", self.video_token_id, 151_656);
        exact!("vision_start_token_id", self.vision_start_token_id, 151_652);
        exact!("vision_end_token_id", self.vision_end_token_id, 151_653);

        let text = &self.text_config;
        exact!("text.model_type", text.model_type.as_str(), "qwen3_vl_text");
        exact!("text.vocab_size", text.vocab_size, 151_936);
        exact!("text.hidden_size", text.hidden_size, 5_120);
        exact!("text.intermediate_size", text.intermediate_size, 25_600);
        exact!(
            "text.num_hidden_layers",
            text.num_hidden_layers,
            H3_FULL_LANGUAGE_LAYERS
        );
        exact!("text.num_attention_heads", text.num_attention_heads, 64);
        exact!("text.num_key_value_heads", text.num_key_value_heads, 8);
        exact!("text.head_dim", text.head_dim, 128);
        exact!("text.rms_norm_eps", text.rms_norm_eps, 1e-6);
        exact!("text.rope_theta", text.rope_theta, 5_000_000.0);
        exact!(
            "text.max_position_embeddings",
            text.max_position_embeddings,
            262_144
        );
        exact!("text.hidden_act", text.hidden_act.as_str(), "silu");
        exact!("text.attention_bias", text.attention_bias, false);
        exact!(
            "text.rope_type",
            text.rope_scaling.rope_type.as_str(),
            "default"
        );
        exact!(
            "text.mrope_interleaved",
            text.rope_scaling.mrope_interleaved,
            true
        );
        exact!(
            "text.mrope_section",
            text.rope_scaling.mrope_section,
            [24, 20, 20]
        );

        let vision = &self.vision_config;
        exact!("vision.model_type", vision.model_type.as_str(), "qwen3_vl");
        exact!("vision.depth", vision.depth, 27);
        exact!("vision.hidden_size", vision.hidden_size, 1_152);
        exact!("vision.intermediate_size", vision.intermediate_size, 4_304);
        exact!("vision.num_heads", vision.num_heads, 16);
        exact!("vision.in_channels", vision.in_channels, 3);
        exact!("vision.patch_size", vision.patch_size, 16);
        exact!("vision.temporal_patch_size", vision.temporal_patch_size, 2);
        exact!("vision.spatial_merge_size", vision.spatial_merge_size, 2);
        exact!("vision.out_hidden_size", vision.out_hidden_size, 5_120);
        exact!(
            "vision.num_position_embeddings",
            vision.num_position_embeddings,
            2_304
        );
        exact!(
            "vision.deepstack_visual_indexes",
            vision.deepstack_visual_indexes.as_slice(),
            [8, 16, 24]
        );
        exact!(
            "vision.hidden_act",
            vision.hidden_act.as_str(),
            "gelu_pytorch_tanh"
        );
        Ok(())
    }

    pub fn materialized_parameter_count(&self) -> u64 {
        let text = &self.text_config;
        let vision = &self.vision_config;
        let h = text.hidden_size as u64;
        let q = (text.num_attention_heads * text.head_dim) as u64;
        let kv = (text.num_key_value_heads * text.head_dim) as u64;
        let intermediate = text.intermediate_size as u64;
        let per_language_layer =
            h * q + 2 * h * kv + q * h + 3 * h * intermediate + 2 * h + 2 * text.head_dim as u64;
        let language =
            text.vocab_size as u64 * h + H3_SELECTED_LANGUAGE_LAYERS as u64 * per_language_layer;

        let vh = vision.hidden_size as u64;
        let vi = vision.intermediate_size as u64;
        let patch = vision.in_channels as u64
            * vision.temporal_patch_size as u64
            * vision.patch_size as u64
            * vision.patch_size as u64
            * vh
            + vh;
        let positions = vision.num_position_embeddings as u64 * vh;
        let per_vision_block = 4 * vh * vh + 4 * vh + 2 * vh * vi + vi + vh + 4 * vh;
        let merged = vh * vision.spatial_merge_size.pow(2) as u64;
        let merger = 2 * vh
            + merged * merged
            + merged
            + merged * vision.out_hidden_size as u64
            + vision.out_hidden_size as u64;
        let deepstack_merger = 2 * merged
            + merged * merged
            + merged
            + merged * vision.out_hidden_size as u64
            + vision.out_hidden_size as u64;
        language
            + patch
            + positions
            + vision.depth as u64 * per_vision_block
            + merger
            + vision.deepstack_visual_indexes.len() as u64 * deepstack_merger
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn released_config() -> String {
        r#"{
              "architectures":["Qwen3VLForConditionalGeneration"],
              "image_token_id":151655,"video_token_id":151656,
              "vision_start_token_id":151652,"vision_end_token_id":151653,
              "text_config":{"model_type":"qwen3_vl_text","vocab_size":151936,
                "hidden_size":5120,"intermediate_size":25600,"num_hidden_layers":64,
                "num_attention_heads":64,"num_key_value_heads":8,"head_dim":128,
                "rms_norm_eps":0.000001,"rope_theta":5000000.0,
                "max_position_embeddings":262144,"hidden_act":"silu","attention_bias":false,
                "rope_scaling":{"rope_type":"default","mrope_interleaved":true,
                  "mrope_section":[24,20,20]}},
              "vision_config":{"model_type":"qwen3_vl","depth":27,"hidden_size":1152,
                "intermediate_size":4304,"num_heads":16,"in_channels":3,"patch_size":16,
                "temporal_patch_size":2,"spatial_merge_size":2,"out_hidden_size":5120,
                "num_position_embeddings":2304,"deepstack_visual_indexes":[8,16,24],
                "hidden_act":"gelu_pytorch_tanh"}
            }"#
        .to_string()
    }

    #[test]
    fn released_config_has_exact_layer_50_charge() {
        let config = H3ConditionerConfig::from_json(released_config().as_bytes()).unwrap();
        assert_eq!(config.materialized_parameter_count(), 25_753_095_920);
        assert_eq!(
            config.materialized_parameter_count() * 2,
            H3_BF16_PARAMETER_BYTES
        );
        assert_eq!(
            H3_COMFY_SAFETENSORS_BYTES - H3_BF16_PARAMETER_BYTES,
            103_416
        );
    }

    #[test]
    fn rejects_ordinary_qwen_variant_before_loading() {
        let wrong = released_config().replace("\"hidden_size\":5120", "\"hidden_size\":4096");
        let error = H3ConditionerConfig::from_json(wrong.as_bytes()).unwrap_err();
        assert!(error.to_string().contains("text.hidden_size"));
    }
}
