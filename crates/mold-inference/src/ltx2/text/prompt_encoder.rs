#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]

use anyhow::{Context, Result};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;

use super::connectors::{
    Embeddings1DConnector, EmbeddingsFeatureExtractor, EmbeddingsProcessor,
    EmbeddingsProcessorOutput, FeatureExtractorV1, FeatureExtractorV2, PaddingSide, Projection,
};
use super::encoder::GemmaHiddenStateEncoder;
use super::gemma::{EncodedPromptPair, GemmaAssets};
use crate::ltx2::model::LtxRopeType;
use crate::ltx2::preset::{GemmaFeatureExtractorKind, Ltx2ModelPreset};

#[derive(Debug, Clone)]
pub struct NativePromptEncoding {
    pub conditional: EmbeddingsProcessorOutput,
    pub unconditional: EmbeddingsProcessorOutput,
}

pub struct NativePromptEncoder {
    gemma: GemmaHiddenStateEncoder,
    embeddings_processor: EmbeddingsProcessor,
    padding_side: PaddingSide,
}

impl NativePromptEncoder {
    pub fn new(
        gemma: GemmaHiddenStateEncoder,
        embeddings_processor: EmbeddingsProcessor,
        padding_side: PaddingSide,
    ) -> Self {
        Self {
            gemma,
            embeddings_processor,
            padding_side,
        }
    }

    pub fn load(
        gemma_root: &std::path::Path,
        checkpoint_path: &std::path::Path,
        preset: &Ltx2ModelPreset,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let assets = GemmaAssets::discover(gemma_root)?;
        let gemma = GemmaHiddenStateEncoder::load_from_assets(&assets, device, dtype)?;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                std::slice::from_ref(&checkpoint_path),
                dtype,
                device,
            )?
        };
        let embeddings_processor = build_embeddings_processor(
            vb,
            preset.feature_extractor,
            preset.gemma.hidden_size,
            preset.gemma.num_hidden_layers,
            preset.video_connector_inner_dim(),
            Some(match preset.feature_extractor {
                GemmaFeatureExtractorKind::V1SharedAv => preset.video_connector_inner_dim(),
                GemmaFeatureExtractorKind::V2DualAv => preset.audio_transformer_inner_dim(),
            }),
            ConnectorSpec {
                prefix: "model.diffusion_model.video_embeddings_connector.",
                num_attention_heads: preset.connectors.video_num_attention_heads,
                attention_head_dim: preset.connectors.video_attention_head_dim,
                num_layers: preset.connectors.video_num_layers,
                apply_gated_attention: preset.connectors.apply_gated_attention,
                positional_embedding_theta: preset.connectors.positional_embedding_theta,
                positional_embedding_max_pos: preset.connectors.positional_embedding_max_pos,
                rope_type: preset.connectors.rope_type,
                double_precision_rope: preset.connectors.double_precision_rope,
                num_learnable_registers: preset.connectors.num_learnable_registers,
            },
            Some(ConnectorSpec {
                prefix: "model.diffusion_model.audio_embeddings_connector.",
                num_attention_heads: match preset.feature_extractor {
                    GemmaFeatureExtractorKind::V1SharedAv => {
                        preset.connectors.video_num_attention_heads
                    }
                    GemmaFeatureExtractorKind::V2DualAv => {
                        preset.connectors.audio_num_attention_heads
                    }
                },
                attention_head_dim: match preset.feature_extractor {
                    GemmaFeatureExtractorKind::V1SharedAv => {
                        preset.connectors.video_attention_head_dim
                    }
                    GemmaFeatureExtractorKind::V2DualAv => {
                        preset.connectors.audio_attention_head_dim
                    }
                },
                num_layers: preset.connectors.audio_num_layers,
                apply_gated_attention: preset.connectors.apply_gated_attention,
                positional_embedding_theta: preset.connectors.positional_embedding_theta,
                positional_embedding_max_pos: preset.connectors.positional_embedding_max_pos,
                rope_type: preset.connectors.rope_type,
                double_precision_rope: preset.connectors.double_precision_rope,
                num_learnable_registers: preset.connectors.num_learnable_registers,
            }),
        )
        .with_context(|| {
            format!(
                "failed to build native LTX-2 embeddings processor from '{}'",
                checkpoint_path.display()
            )
        })?;

        Ok(Self::new(gemma, embeddings_processor, PaddingSide::Left))
    }

    pub fn encode_prompt_pair(&mut self, pair: &EncodedPromptPair) -> Result<NativePromptEncoding> {
        self.encode_prompt_pair_with_unconditional(pair, true)
    }

    pub fn encode_prompt_pair_with_unconditional(
        &mut self,
        pair: &EncodedPromptPair,
        encode_unconditional: bool,
    ) -> Result<NativePromptEncoding> {
        let conditional = self
            .encode_prompt_tokens(&pair.conditional)
            .context("failed to build conditional native LTX-2 embeddings")?;
        let unconditional = if encode_unconditional {
            self.encode_prompt_tokens(&pair.unconditional)
                .context("failed to build unconditional native LTX-2 embeddings")?
        } else {
            conditional.clone()
        };

        Ok(NativePromptEncoding {
            conditional,
            unconditional,
        })
    }

    pub fn device(&self) -> &Device {
        self.gemma.device()
    }

    fn encode_prompt_tokens(
        &mut self,
        tokens: &super::gemma::PromptTokens,
    ) -> Result<EmbeddingsProcessorOutput> {
        let encoded = self
            .gemma
            .encode_prompt_tokens(tokens)
            .context("failed to encode Gemma prompt tokens")?;
        self.embeddings_processor
            .process_hidden_states(
                &encoded.hidden_states,
                &encoded.attention_mask,
                self.padding_side,
            )
            .context("failed to project native LTX-2 prompt embeddings")
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct ConnectorSpec<'a> {
    pub(crate) prefix: &'a str,
    pub(crate) num_attention_heads: usize,
    pub(crate) attention_head_dim: usize,
    pub(crate) num_layers: usize,
    pub(crate) apply_gated_attention: bool,
    pub(crate) positional_embedding_theta: f64,
    pub(crate) positional_embedding_max_pos: &'a [usize],
    pub(crate) rope_type: LtxRopeType,
    pub(crate) double_precision_rope: bool,
    pub(crate) num_learnable_registers: Option<usize>,
}

pub(crate) fn build_embeddings_processor(
    vb: VarBuilder,
    feature_extractor_kind: GemmaFeatureExtractorKind,
    gemma_hidden_size: usize,
    gemma_num_hidden_layers: usize,
    video_out_dim: usize,
    audio_out_dim: Option<usize>,
    video_connector: ConnectorSpec<'_>,
    audio_connector: Option<ConnectorSpec<'_>>,
) -> Result<EmbeddingsProcessor> {
    let flat_dim = gemma_hidden_size * (gemma_num_hidden_layers + 1);
    let feature_extractor = match feature_extractor_kind {
        GemmaFeatureExtractorKind::V1SharedAv => {
            let weight = vb.get(
                (video_out_dim, flat_dim),
                "text_embedding_projection.aggregate_embed.weight",
            )?;
            EmbeddingsFeatureExtractor::V1(FeatureExtractorV1::new(
                Projection::new(weight, None),
                true,
            ))
        }
        GemmaFeatureExtractorKind::V2DualAv => {
            let video_weight = vb.get(
                (video_out_dim, flat_dim),
                "text_embedding_projection.video_aggregate_embed.weight",
            )?;
            let video_bias = vb.get(
                video_out_dim,
                "text_embedding_projection.video_aggregate_embed.bias",
            )?;
            let audio_out_dim = audio_out_dim.expect("V2 feature extractor requires audio output");
            let audio_weight = vb.get(
                (audio_out_dim, flat_dim),
                "text_embedding_projection.audio_aggregate_embed.weight",
            )?;
            let audio_bias = vb.get(
                audio_out_dim,
                "text_embedding_projection.audio_aggregate_embed.bias",
            )?;
            EmbeddingsFeatureExtractor::V2(FeatureExtractorV2::new(
                Projection::new(video_weight, Some(video_bias)),
                Some(Projection::new(audio_weight, Some(audio_bias))),
                gemma_hidden_size,
            ))
        }
    };

    let video_connector = build_connector(vb.clone(), video_connector)?;
    let audio_connector = audio_connector
        .map(|spec| build_connector(vb.clone(), spec))
        .transpose()?;

    Ok(EmbeddingsProcessor::new(
        feature_extractor,
        video_connector,
        audio_connector,
    ))
}

fn build_connector(vb: VarBuilder, spec: ConnectorSpec<'_>) -> Result<Embeddings1DConnector> {
    let prefix = spec.prefix.to_string();
    let vb = vb.rename_f(move |name| format!("{prefix}{name}"));
    Embeddings1DConnector::new(
        spec.num_attention_heads,
        spec.attention_head_dim,
        spec.num_layers,
        spec.positional_embedding_theta,
        spec.positional_embedding_max_pos.to_vec(),
        spec.rope_type,
        spec.double_precision_rope,
        spec.num_learnable_registers,
        spec.apply_gated_attention,
        vb,
    )
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;

    use super::{build_embeddings_processor, ConnectorSpec, NativePromptEncoder};
    use crate::ltx2::model::LtxRopeType;
    use crate::ltx2::preset::GemmaFeatureExtractorKind;
    use crate::ltx2::text::connectors::PaddingSide;
    use crate::ltx2::text::encoder::{GemmaConfig, GemmaHiddenStateEncoder};
    use crate::ltx2::text::gemma::{EncodedPromptPair, PromptTokens};

    fn tiny_gemma_config() -> GemmaConfig {
        GemmaConfig {
            attention_bias: false,
            head_dim: 4,
            hidden_activation: candle_nn::Activation::GeluPytorchTanh,
            hidden_size: 8,
            intermediate_size: 16,
            num_attention_heads: 2,
            num_hidden_layers: 2,
            num_key_value_heads: 1,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            rope_local_base_freq: 10_000.0,
            vocab_size: 16,
            final_logit_softcapping: None,
            attn_logit_softcapping: None,
            query_pre_attn_scalar: 4,
            sliding_window: 4,
            sliding_window_pattern: 2,
            max_position_embeddings: 32,
        }
    }

    fn zero_gemma_var_builder(cfg: &GemmaConfig) -> VarBuilder<'static> {
        let mut tensors = HashMap::new();
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            Tensor::zeros((cfg.vocab_size, cfg.hidden_size), DType::F32, &Device::Cpu).unwrap(),
        );
        for layer in 0..cfg.num_hidden_layers {
            for name in [
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
                "mlp.gate_proj",
                "mlp.up_proj",
                "mlp.down_proj",
            ] {
                let (rows, cols) = match name {
                    "self_attn.q_proj" => (cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size),
                    "self_attn.k_proj" | "self_attn.v_proj" => {
                        (cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size)
                    }
                    "self_attn.o_proj" => (cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim),
                    "mlp.gate_proj" | "mlp.up_proj" => (cfg.intermediate_size, cfg.hidden_size),
                    "mlp.down_proj" => (cfg.hidden_size, cfg.intermediate_size),
                    _ => unreachable!(),
                };
                tensors.insert(
                    format!("model.layers.{layer}.{name}.weight"),
                    Tensor::zeros((rows, cols), DType::F32, &Device::Cpu).unwrap(),
                );
            }
            for name in [
                "self_attn.q_norm",
                "self_attn.k_norm",
                "input_layernorm",
                "pre_feedforward_layernorm",
                "post_feedforward_layernorm",
                "post_attention_layernorm",
            ] {
                let dim = if name.contains("q_norm") || name.contains("k_norm") {
                    cfg.head_dim
                } else {
                    cfg.hidden_size
                };
                tensors.insert(
                    format!("model.layers.{layer}.{name}.weight"),
                    Tensor::zeros(dim, DType::F32, &Device::Cpu).unwrap(),
                );
            }
        }
        tensors.insert(
            "model.norm.weight".to_string(),
            Tensor::zeros(cfg.hidden_size, DType::F32, &Device::Cpu).unwrap(),
        );
        VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu)
    }

    fn zero_connector_source_var_builder() -> VarBuilder<'static> {
        let mut tensors = HashMap::new();
        tensors.insert(
            "text_embedding_projection.video_aggregate_embed.weight".to_string(),
            Tensor::zeros((8, 24), DType::F32, &Device::Cpu).unwrap(),
        );
        tensors.insert(
            "text_embedding_projection.video_aggregate_embed.bias".to_string(),
            Tensor::zeros(8, DType::F32, &Device::Cpu).unwrap(),
        );
        tensors.insert(
            "text_embedding_projection.audio_aggregate_embed.weight".to_string(),
            Tensor::zeros((4, 24), DType::F32, &Device::Cpu).unwrap(),
        );
        tensors.insert(
            "text_embedding_projection.audio_aggregate_embed.bias".to_string(),
            Tensor::zeros(4, DType::F32, &Device::Cpu).unwrap(),
        );
        for (prefix, dim) in [
            ("model.diffusion_model.video_embeddings_connector", 8usize),
            ("model.diffusion_model.audio_embeddings_connector", 4usize),
        ] {
            for linear_name in ["attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0"] {
                tensors.insert(
                    format!("{prefix}.transformer_1d_blocks.0.{linear_name}.weight"),
                    Tensor::zeros((dim, dim), DType::F32, &Device::Cpu).unwrap(),
                );
                tensors.insert(
                    format!("{prefix}.transformer_1d_blocks.0.{linear_name}.bias"),
                    Tensor::zeros(dim, DType::F32, &Device::Cpu).unwrap(),
                );
            }
            for norm_name in ["attn1.q_norm", "attn1.k_norm"] {
                tensors.insert(
                    format!("{prefix}.transformer_1d_blocks.0.{norm_name}.weight"),
                    Tensor::ones(dim, DType::F32, &Device::Cpu).unwrap(),
                );
            }
            tensors.insert(
                format!("{prefix}.transformer_1d_blocks.0.ff.net.0.proj.weight"),
                Tensor::zeros((dim * 4, dim), DType::F32, &Device::Cpu).unwrap(),
            );
            tensors.insert(
                format!("{prefix}.transformer_1d_blocks.0.ff.net.0.proj.bias"),
                Tensor::zeros(dim * 4, DType::F32, &Device::Cpu).unwrap(),
            );
            tensors.insert(
                format!("{prefix}.transformer_1d_blocks.0.ff.net.2.weight"),
                Tensor::zeros((dim, dim * 4), DType::F32, &Device::Cpu).unwrap(),
            );
            tensors.insert(
                format!("{prefix}.transformer_1d_blocks.0.ff.net.2.bias"),
                Tensor::zeros(dim, DType::F32, &Device::Cpu).unwrap(),
            );
            tensors.insert(
                format!("{prefix}.learnable_registers"),
                Tensor::zeros((128, dim), DType::F32, &Device::Cpu).unwrap(),
            );
        }
        VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu)
    }

    fn prompt_pair() -> EncodedPromptPair {
        EncodedPromptPair {
            conditional: PromptTokens {
                input_ids: vec![0, 0, 5],
                attention_mask: vec![0, 0, 1],
            },
            unconditional: PromptTokens {
                input_ids: vec![0, 0, 0],
                attention_mask: vec![0, 0, 0],
            },
            pad_token_id: 0,
            eos_token_id: Some(1),
            max_length: 3,
        }
    }

    #[test]
    fn native_prompt_encoder_wires_gemma_and_embeddings_processor() {
        let cfg = tiny_gemma_config();
        let gemma = GemmaHiddenStateEncoder::new(&cfg, zero_gemma_var_builder(&cfg)).unwrap();
        let processor = build_embeddings_processor(
            zero_connector_source_var_builder(),
            GemmaFeatureExtractorKind::V2DualAv,
            cfg.hidden_size,
            cfg.num_hidden_layers,
            8,
            Some(4),
            ConnectorSpec {
                prefix: "model.diffusion_model.video_embeddings_connector.",
                num_attention_heads: 2,
                attention_head_dim: 4,
                num_layers: 1,
                apply_gated_attention: false,
                positional_embedding_theta: 10_000.0,
                positional_embedding_max_pos: &[32],
                rope_type: LtxRopeType::Split,
                double_precision_rope: true,
                num_learnable_registers: Some(128),
            },
            Some(ConnectorSpec {
                prefix: "model.diffusion_model.audio_embeddings_connector.",
                num_attention_heads: 1,
                attention_head_dim: 4,
                num_layers: 1,
                apply_gated_attention: false,
                positional_embedding_theta: 10_000.0,
                positional_embedding_max_pos: &[32],
                rope_type: LtxRopeType::Split,
                double_precision_rope: true,
                num_learnable_registers: Some(128),
            }),
        )
        .unwrap();
        let mut prompt_encoder = NativePromptEncoder::new(gemma, processor, PaddingSide::Left);

        let output = prompt_encoder.encode_prompt_pair(&prompt_pair()).unwrap();

        assert_eq!(
            output.conditional.video_encoding.dims3().unwrap(),
            (1, 3, 8)
        );
        assert_eq!(
            output.conditional.audio_encoding.unwrap().dims3().unwrap(),
            (1, 3, 4)
        );
        assert_eq!(
            output.conditional.attention_mask.to_vec2::<u8>().unwrap(),
            vec![vec![1, 1, 1]]
        );
        assert_eq!(
            output.unconditional.attention_mask.to_vec2::<u8>().unwrap(),
            vec![vec![1, 1, 1]]
        );
    }

    #[test]
    fn native_prompt_encoder_can_skip_unconditional_pass_for_guidance_free_paths() {
        let cfg = tiny_gemma_config();
        let gemma = GemmaHiddenStateEncoder::new(&cfg, zero_gemma_var_builder(&cfg)).unwrap();
        let processor = build_embeddings_processor(
            zero_connector_source_var_builder(),
            GemmaFeatureExtractorKind::V2DualAv,
            cfg.hidden_size,
            cfg.num_hidden_layers,
            8,
            Some(4),
            ConnectorSpec {
                prefix: "model.diffusion_model.video_embeddings_connector.",
                num_attention_heads: 2,
                attention_head_dim: 4,
                num_layers: 1,
                apply_gated_attention: false,
                positional_embedding_theta: 10_000.0,
                positional_embedding_max_pos: &[32],
                rope_type: LtxRopeType::Split,
                double_precision_rope: true,
                num_learnable_registers: Some(128),
            },
            Some(ConnectorSpec {
                prefix: "model.diffusion_model.audio_embeddings_connector.",
                num_attention_heads: 1,
                attention_head_dim: 4,
                num_layers: 1,
                apply_gated_attention: false,
                positional_embedding_theta: 10_000.0,
                positional_embedding_max_pos: &[32],
                rope_type: LtxRopeType::Split,
                double_precision_rope: true,
                num_learnable_registers: Some(128),
            }),
        )
        .unwrap();
        let mut prompt_encoder = NativePromptEncoder::new(gemma, processor, PaddingSide::Left);

        let output = prompt_encoder
            .encode_prompt_pair_with_unconditional(&prompt_pair(), false)
            .unwrap();

        assert_eq!(
            output.conditional.video_encoding.to_vec3::<f32>().unwrap(),
            output
                .unconditional
                .video_encoding
                .to_vec3::<f32>()
                .unwrap()
        );
        assert_eq!(
            output.conditional.attention_mask.to_vec2::<u8>().unwrap(),
            output.unconditional.attention_mask.to_vec2::<u8>().unwrap()
        );
    }
}
