use anyhow::{bail, Result};

use crate::ltx2::model::LtxRopeType;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CaptionProjectionPlacement {
    Transformer,
    TextEncoderConnector,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GemmaFeatureExtractorKind {
    V1SharedAv,
    V2DualAv,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GemmaProfile {
    pub(crate) hidden_size: usize,
    pub(crate) num_hidden_layers: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) num_key_value_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) intermediate_size: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ConnectorProfile {
    pub(crate) video_num_attention_heads: usize,
    pub(crate) video_attention_head_dim: usize,
    pub(crate) video_num_layers: usize,
    pub(crate) audio_num_attention_heads: usize,
    pub(crate) audio_attention_head_dim: usize,
    pub(crate) audio_num_layers: usize,
    pub(crate) positional_embedding_theta: f64,
    pub(crate) positional_embedding_max_pos: &'static [usize],
    pub(crate) rope_type: LtxRopeType,
    pub(crate) double_precision_rope: bool,
    pub(crate) num_learnable_registers: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TransformerProfile {
    pub(crate) num_attention_heads: usize,
    pub(crate) attention_head_dim: usize,
    pub(crate) num_layers: usize,
    pub(crate) in_channels: usize,
    pub(crate) out_channels: usize,
    pub(crate) cross_attention_dim: usize,
    pub(crate) audio_num_attention_heads: usize,
    pub(crate) audio_attention_head_dim: usize,
    pub(crate) audio_in_channels: usize,
    pub(crate) audio_out_channels: usize,
    pub(crate) audio_cross_attention_dim: usize,
    pub(crate) cross_attention_adaln: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Ltx2ModelPreset {
    pub(crate) name: &'static str,
    pub(crate) caption_projection: CaptionProjectionPlacement,
    pub(crate) feature_extractor: GemmaFeatureExtractorKind,
    pub(crate) transformer: TransformerProfile,
    pub(crate) connectors: ConnectorProfile,
    pub(crate) gemma: GemmaProfile,
    pub(crate) supports_spatial_upscale_x1_5: bool,
    pub(crate) supports_spatial_upscale_x2: bool,
    pub(crate) supports_temporal_upscale_x2: bool,
    pub(crate) streaming_prefetch_count: u32,
}

impl Ltx2ModelPreset {
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn transformer_inner_dim(self) -> usize {
        self.transformer.num_attention_heads * self.transformer.attention_head_dim
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn audio_transformer_inner_dim(self) -> usize {
        self.transformer.audio_num_attention_heads * self.transformer.audio_attention_head_dim
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn gemma_flat_dim(self) -> usize {
        self.gemma.hidden_size * (self.gemma.num_hidden_layers + 1)
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn video_connector_inner_dim(self) -> usize {
        self.connectors.video_num_attention_heads * self.connectors.video_attention_head_dim
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn audio_connector_inner_dim(self) -> usize {
        self.connectors.audio_num_attention_heads * self.connectors.audio_attention_head_dim
    }
}

const GEMMA_PROFILE: GemmaProfile = GemmaProfile {
    hidden_size: 3840,
    num_hidden_layers: 48,
    num_attention_heads: 16,
    num_key_value_heads: 8,
    head_dim: 256,
    intermediate_size: 15360,
};

const CONNECTOR_PROFILE_19B: ConnectorProfile = ConnectorProfile {
    video_num_attention_heads: 30,
    video_attention_head_dim: 128,
    video_num_layers: 2,
    audio_num_attention_heads: 30,
    audio_attention_head_dim: 128,
    audio_num_layers: 2,
    positional_embedding_theta: 10_000.0,
    positional_embedding_max_pos: &[4096],
    rope_type: LtxRopeType::Split,
    double_precision_rope: true,
    num_learnable_registers: Some(128),
};

const CONNECTOR_PROFILE_22B: ConnectorProfile = ConnectorProfile {
    video_num_attention_heads: 30,
    video_attention_head_dim: 128,
    video_num_layers: 2,
    audio_num_attention_heads: 32,
    audio_attention_head_dim: 64,
    audio_num_layers: 2,
    positional_embedding_theta: 10_000.0,
    positional_embedding_max_pos: &[4096],
    rope_type: LtxRopeType::Split,
    double_precision_rope: true,
    num_learnable_registers: Some(128),
};

const TRANSFORMER_PROFILE: TransformerProfile = TransformerProfile {
    num_attention_heads: 32,
    attention_head_dim: 128,
    num_layers: 48,
    in_channels: 128,
    out_channels: 128,
    cross_attention_dim: 4096,
    audio_num_attention_heads: 32,
    audio_attention_head_dim: 64,
    audio_in_channels: 128,
    audio_out_channels: 128,
    audio_cross_attention_dim: 2048,
    cross_attention_adaln: false,
};

const PRESET_19B: Ltx2ModelPreset = Ltx2ModelPreset {
    name: "ltx-2-19b",
    caption_projection: CaptionProjectionPlacement::Transformer,
    feature_extractor: GemmaFeatureExtractorKind::V1SharedAv,
    transformer: TRANSFORMER_PROFILE,
    connectors: CONNECTOR_PROFILE_19B,
    gemma: GEMMA_PROFILE,
    supports_spatial_upscale_x1_5: false,
    supports_spatial_upscale_x2: true,
    supports_temporal_upscale_x2: true,
    streaming_prefetch_count: 2,
};

const PRESET_22B: Ltx2ModelPreset = Ltx2ModelPreset {
    name: "ltx-2.3-22b",
    caption_projection: CaptionProjectionPlacement::TextEncoderConnector,
    feature_extractor: GemmaFeatureExtractorKind::V2DualAv,
    transformer: TRANSFORMER_PROFILE,
    connectors: CONNECTOR_PROFILE_22B,
    gemma: GEMMA_PROFILE,
    supports_spatial_upscale_x1_5: true,
    supports_spatial_upscale_x2: true,
    supports_temporal_upscale_x2: true,
    streaming_prefetch_count: 2,
};

pub(crate) fn preset_for_model(model_name: &str) -> Result<Ltx2ModelPreset> {
    if model_name.contains("ltx-2.3") {
        Ok(PRESET_22B)
    } else if model_name.contains("ltx-2") {
        Ok(PRESET_19B)
    } else {
        bail!("unsupported LTX-2 preset for model '{model_name}'");
    }
}

#[cfg(test)]
mod tests {
    use super::{preset_for_model, CaptionProjectionPlacement, GemmaFeatureExtractorKind};

    #[test]
    fn preset_selection_distinguishes_19b_and_22b_profiles() {
        let preset_19b = preset_for_model("ltx-2-19b-distilled:fp8").unwrap();
        assert_eq!(preset_19b.name, "ltx-2-19b");
        assert_eq!(
            preset_19b.caption_projection,
            CaptionProjectionPlacement::Transformer
        );
        assert_eq!(
            preset_19b.feature_extractor,
            GemmaFeatureExtractorKind::V1SharedAv
        );
        assert!(!preset_19b.supports_spatial_upscale_x1_5);
        assert_eq!(preset_19b.transformer_inner_dim(), 4096);
        assert_eq!(preset_19b.audio_transformer_inner_dim(), 2048);
        assert_eq!(preset_19b.video_connector_inner_dim(), 3840);
        assert_eq!(preset_19b.audio_connector_inner_dim(), 3840);
        assert_eq!(preset_19b.gemma_flat_dim(), 188_160);

        let preset_22b = preset_for_model("ltx-2.3-22b-dev:fp8").unwrap();
        assert_eq!(preset_22b.name, "ltx-2.3-22b");
        assert_eq!(
            preset_22b.caption_projection,
            CaptionProjectionPlacement::TextEncoderConnector
        );
        assert_eq!(
            preset_22b.feature_extractor,
            GemmaFeatureExtractorKind::V2DualAv
        );
        assert!(preset_22b.supports_spatial_upscale_x1_5);
        assert_eq!(preset_22b.streaming_prefetch_count, 2);
        assert_eq!(preset_22b.video_connector_inner_dim(), 3840);
        assert_eq!(preset_22b.audio_connector_inner_dim(), 2048);
    }
}
