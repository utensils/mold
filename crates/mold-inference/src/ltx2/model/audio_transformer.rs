//! Audio-only LTX-2 transformer — the denoiser behind text-to-audio (T2A).
//!
//! Upstream ships this as a separate *configurator* over the same checkpoint
//! (`LTXAudioOnlyModelConfigurator` in
//! `packages/ltx-core/src/ltx_core/model/transformer/model_configurator.py:126`,
//! selected by `packages/ltx-pipelines/src/ltx_pipelines/t2a_one_stage.py:86`),
//! paired with an SDOps map that restricts checkpoint reads to the `audio_*`
//! keys so the video weights are never even read from disk
//! (`model_configurator.py:207`).
//!
//! Mold mirrors that split rather than making [`super::video_transformer`]'s
//! AV hot path optional. Two reasons, in order:
//!
//! 1. The six shipped LTX-2 pipelines all run through
//!    `LtxAvTransformerBlock::forward`, and `Option`-ifying its video branch
//!    would put every one of them behind a new set of `if let` arms for a
//!    feature none of them use.
//! 2. It is the difference between T2A being fast and being pointless. The
//!    audio branch is roughly a quarter of the per-block parameters
//!    (`audio_dim` 2048 against `inner_dim` 4096) and both cross-modal
//!    attentions are gone, so an audio-only build is ~3.5 GB eagerly resident
//!    instead of ~19 GB streamed block-by-block — for a denoise over ~126
//!    tokens.
//!
//! What keeps the two from drifting is
//! `audio_only_block_matches_av_block_audio_branch` at the bottom of this
//! file: it runs this block and `LtxAvTransformerBlock::forward(idx, None,
//! Some(&audio), …)` — already-trusted production code — over the same
//! weights and inputs and asserts the outputs match element-wise.

use candle_core::{DType, Result, Tensor};
use candle_nn as nn;
use nn::{Module, VarBuilder};
use std::sync::Arc;

use crate::ltx2::guidance::{BatchedPerturbationConfig, PerturbationType};
use crate::ltx2::lora::Ltx2LoraRegistry;

use super::video_transformer::{
    ada_component, gate_tokens, log_detail_tensor, lora_adapters_for, ltx2_block_debug_enabled,
    ltx2_load_debug_enabled, modulate_tokens, rms_norm_tensor, tensor_debug_stats,
    AdaLayerNormSingle, FeedForward, LayerNormNoParams, Ltx2AvTransformer3DModel,
    Ltx2VideoRotaryPosEmbed, Ltx2VideoTransformer3DModelConfig, LtxAttention, LtxLinear,
    LtxPreparedModality, LtxPreparedModalityStatic, Nvfp4LinearCache, PixArtAlphaTextProjection,
};

/// One audio-only transformer block: self-attention, text cross-attention,
/// feed-forward. Structurally the `audio` half of [`LtxAvTransformerBlock`]
/// with the audio↔video cross-attention pair and every `video_*` weight
/// omitted.
///
/// [`LtxAvTransformerBlock`]: super::video_transformer::LtxAvTransformerBlock
#[derive(Clone, Debug)]
pub(crate) struct LtxAudioTransformerBlock {
    audio_attn1: LtxAttention,
    audio_attn2: LtxAttention,
    audio_ff: FeedForward,
    audio_scale_shift_table: Tensor,
    norm_eps: f64,
    cross_attention_adaln: bool,
    audio_prompt_scale_shift_table: Option<Tensor>,
}

impl LtxAudioTransformerBlock {
    pub(crate) fn new(
        config: &Ltx2VideoTransformer3DModelConfig,
        vb: VarBuilder,
        lora_registry: Option<&Ltx2LoraRegistry>,
        block_key: &str,
        nvfp4_cache: Option<&Nvfp4LinearCache>,
    ) -> Result<Self> {
        let audio_dim = config.audio_num_attention_heads * config.audio_attention_head_dim;

        let audio_attn1 = LtxAttention::new(
            audio_dim,
            config.audio_num_attention_heads,
            config.audio_num_attention_heads,
            config.audio_attention_head_dim,
            0.0,
            config.attention_bias,
            None,
            config.attention_out_bias,
            &config.qk_norm,
            config.rope_type,
            config.apply_gated_attention,
            vb.pp("audio_attn1"),
            lora_registry,
            &format!("{block_key}.audio_attn1"),
            nvfp4_cache,
        )?;
        let audio_attn2 = LtxAttention::new(
            audio_dim,
            config.audio_num_attention_heads,
            config.audio_num_attention_heads,
            config.audio_attention_head_dim,
            0.0,
            config.attention_bias,
            Some(config.audio_cross_attention_dim),
            config.attention_out_bias,
            &config.qk_norm,
            config.rope_type,
            config.apply_gated_attention,
            vb.pp("audio_attn2"),
            lora_registry,
            &format!("{block_key}.audio_attn2"),
            nvfp4_cache,
        )?;
        let audio_ff = FeedForward::new(
            audio_dim,
            vb.pp("audio_ff"),
            lora_registry,
            &format!("{block_key}.audio_ff"),
            nvfp4_cache,
            config.audio_ff_bias,
        )?;
        let audio_scale_shift_table = vb.get(
            (if config.cross_attention_adaln { 9 } else { 6 }, audio_dim),
            "audio_scale_shift_table",
        )?;
        let audio_prompt_scale_shift_table = if config.cross_attention_adaln {
            Some(vb.get((2, audio_dim), "audio_prompt_scale_shift_table")?)
        } else {
            None
        };

        Ok(Self {
            audio_attn1,
            audio_attn2,
            audio_ff,
            audio_scale_shift_table,
            norm_eps: config.norm_eps,
            cross_attention_adaln: config.cross_attention_adaln,
            audio_prompt_scale_shift_table,
        })
    }

    fn ada_triplet(
        &self,
        timestep: &Tensor,
        start_index: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        Ok((
            ada_component(&self.audio_scale_shift_table, timestep, start_index)?,
            ada_component(&self.audio_scale_shift_table, timestep, start_index + 1)?,
            ada_component(&self.audio_scale_shift_table, timestep, start_index + 2)?,
        ))
    }

    fn apply_text_cross_attention(
        &self,
        x: &Tensor,
        context: &Tensor,
        timestep: &Tensor,
        prompt_timestep: Option<&Tensor>,
        context_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        if self.cross_attention_adaln {
            let prompt_scale_shift_table = self
                .audio_prompt_scale_shift_table
                .as_ref()
                .ok_or_else(|| {
                    candle_core::Error::msg(
                        "cross-attention AdaLN requires prompt scale-shift weights",
                    )
                })?;
            let prompt_timestep = prompt_timestep.ok_or_else(|| {
                candle_core::Error::msg("cross-attention AdaLN requires prompt timestep embeddings")
            })?;
            let (shift_q, scale_q, gate) = self.ada_triplet(timestep, 6)?;
            let attn_input =
                modulate_tokens(&rms_norm_tensor(x, self.norm_eps)?, &scale_q, &shift_q)?;
            let shift_kv = ada_component(prompt_scale_shift_table, prompt_timestep, 0)?;
            let scale_kv = ada_component(prompt_scale_shift_table, prompt_timestep, 1)?;
            let context = modulate_tokens(context, &scale_kv, &shift_kv)?;
            return gate_tokens(
                &self.audio_attn2.forward(
                    &attn_input,
                    Some(&context),
                    context_mask,
                    None,
                    None,
                    None,
                    false,
                )?,
                &gate,
            );
        }
        self.audio_attn2.forward(
            &rms_norm_tensor(x, self.norm_eps)?,
            Some(context),
            context_mask,
            None,
            None,
            None,
            false,
        )
    }

    pub(crate) fn forward(
        &self,
        index: usize,
        audio: &LtxPreparedModality,
        perturbations: &BatchedPerturbationConfig,
    ) -> Result<Tensor> {
        let (a_shift_msa, a_scale_msa, a_gate_msa) = self.ada_triplet(&audio.timesteps, 0)?;
        let a_self_input = modulate_tokens(
            &rms_norm_tensor(&audio.x, self.norm_eps)?,
            &a_scale_msa,
            &a_shift_msa,
        )?;
        let all_audio_self_perturbed =
            perturbations.all_in_batch(PerturbationType::SkipAudioSelfAttention, index);
        let audio_self_mask = if all_audio_self_perturbed
            || !perturbations.any_in_batch(PerturbationType::SkipAudioSelfAttention, index)
        {
            None
        } else {
            Some(perturbations.mask_like(
                PerturbationType::SkipAudioSelfAttention,
                index,
                &a_self_input,
            )?)
        };
        let a_self = gate_tokens(
            &self.audio_attn1.forward(
                &a_self_input,
                None,
                audio.self_attention_mask.as_ref(),
                Some((&audio.rope.0, &audio.rope.1)),
                None,
                audio_self_mask.as_ref(),
                all_audio_self_perturbed,
            )?,
            &a_gate_msa,
        )?;
        log_detail_tensor(index, "audio_input", &audio.x)?;
        log_detail_tensor(index, "audio_timesteps", &audio.timesteps)?;
        log_detail_tensor(index, "audio_embedded_timestep", &audio.embedded_timestep)?;
        log_detail_tensor(index, "audio_context", &audio.context)?;
        log_detail_tensor(index, "audio_self_input", &a_self_input)?;
        log_detail_tensor(index, "audio_self_out", &a_self)?;
        let mut ax = audio.x.broadcast_add(&a_self)?;
        log_detail_tensor(index, "audio_after_self", &ax)?;

        let a_text_cross = self.apply_text_cross_attention(
            &ax,
            &audio.context,
            &audio.timesteps,
            audio.prompt_timestep.as_ref(),
            audio.context_mask.as_ref(),
        )?;
        log_detail_tensor(index, "audio_text_cross_out", &a_text_cross)?;
        ax = ax.broadcast_add(&a_text_cross)?;
        log_detail_tensor(index, "audio_after_text_cross", &ax)?;

        let (a_shift_mlp, a_scale_mlp, a_gate_mlp) = self.ada_triplet(&audio.timesteps, 3)?;
        let a_ff_input = modulate_tokens(
            &rms_norm_tensor(&ax, self.norm_eps)?,
            &a_scale_mlp,
            &a_shift_mlp,
        )?;
        let a_ff = gate_tokens(&self.audio_ff.forward(&a_ff_input)?, &a_gate_mlp)?;
        log_detail_tensor(index, "audio_ff_input", &a_ff_input)?;
        log_detail_tensor(index, "audio_ff_out", &a_ff)?;
        ax = ax.broadcast_add(&a_ff)?;
        log_detail_tensor(index, "audio_after_ff", &ax)?;
        Ok(ax)
    }
}

/// Audio-only LTX-2 denoiser. Loads exactly the `audio_*` half of a combined
/// AV checkpoint; every video weight (and the audio↔video cross-attention
/// pair) is left on disk.
pub struct Ltx2AudioTransformerModel {
    audio_patchify_proj: nn::Linear,
    audio_adaln_single: AdaLayerNormSingle,
    audio_prompt_adaln_single: Option<AdaLayerNormSingle>,
    audio_caption_projection: Option<PixArtAlphaTextProjection>,
    audio_scale_shift_table: Tensor,
    audio_norm_out: LayerNormNoParams,
    audio_proj_out: LtxLinear,
    audio_rope: Ltx2VideoRotaryPosEmbed,
    transformer_blocks: Vec<LtxAudioTransformerBlock>,
    config: Ltx2VideoTransformer3DModelConfig,
}

impl Ltx2AudioTransformerModel {
    pub fn new(
        config: &Ltx2VideoTransformer3DModelConfig,
        vb: VarBuilder,
        lora_registry: Option<Arc<Ltx2LoraRegistry>>,
    ) -> Result<Self> {
        let audio_dim = config.audio_num_attention_heads * config.audio_attention_head_dim;
        let nvfp4_cache = Nvfp4LinearCache::default();
        let adaln_embedding_coefficient = if config.cross_attention_adaln { 9 } else { 6 };

        let mut transformer_blocks = Vec::with_capacity(config.num_layers);
        for layer_idx in 0..config.num_layers {
            transformer_blocks.push(LtxAudioTransformerBlock::new(
                config,
                vb.pp("transformer_blocks").pp(layer_idx.to_string()),
                lora_registry.as_deref(),
                &format!("diffusion_model.transformer_blocks.{layer_idx}"),
                Some(&nvfp4_cache),
            )?);
            if ltx2_load_debug_enabled() {
                eprintln!(
                    "[ltx2-load] eager_audio_block={}/{}",
                    layer_idx + 1,
                    config.num_layers
                );
            }
        }

        Ok(Self {
            audio_patchify_proj: nn::linear(
                config.audio_in_channels,
                audio_dim,
                vb.pp("audio_patchify_proj"),
            )?,
            audio_adaln_single: AdaLayerNormSingle::new_with_coefficient(
                audio_dim,
                adaln_embedding_coefficient,
                vb.pp("audio_adaln_single"),
            )?,
            audio_prompt_adaln_single: if config.cross_attention_adaln {
                Some(AdaLayerNormSingle::new_with_coefficient(
                    audio_dim,
                    2,
                    vb.pp("audio_prompt_adaln_single"),
                )?)
            } else {
                None
            },
            audio_caption_projection: if config.caption_projection_in_transformer {
                Some(PixArtAlphaTextProjection::new_with_out_features(
                    config.caption_channels,
                    audio_dim,
                    audio_dim,
                    vb.pp("audio_caption_projection"),
                )?)
            } else {
                None
            },
            audio_scale_shift_table: vb.get((2, audio_dim), "audio_scale_shift_table")?,
            audio_norm_out: LayerNormNoParams::new(config.norm_eps),
            audio_proj_out: LtxLinear::load_with_nvfp4_cache(
                audio_dim,
                config.audio_out_channels,
                true,
                vb.pp("audio_proj_out"),
                lora_adapters_for(lora_registry.as_deref(), "diffusion_model.audio_proj_out"),
                Some(&nvfp4_cache),
                Some("diffusion_model.audio_proj_out"),
            )?,
            audio_rope: Ltx2VideoRotaryPosEmbed::new(
                audio_dim,
                config.positional_embedding_theta,
                config.audio_positional_embedding_max_pos.clone(),
                config.use_middle_indices_grid,
                config.audio_num_attention_heads,
                config.rope_type,
                config.double_precision_rope,
            ),
            transformer_blocks,
            config: config.clone(),
        })
    }

    fn compute_dtype(&self) -> DType {
        match self.audio_patchify_proj.weight().dtype() {
            DType::F8E4M3 => DType::BF16,
            other => other,
        }
    }

    pub(crate) fn prepare_static_inputs(
        &self,
        audio_encoder_hidden_states: &Tensor,
        audio_encoder_attention_mask: Option<&Tensor>,
        audio_self_attention_mask: Option<&Tensor>,
        audio_positions: &Tensor,
    ) -> Result<LtxPreparedModalityStatic> {
        let compute_dtype = self.compute_dtype();
        let context = audio_encoder_hidden_states.to_dtype(compute_dtype)?;
        let context = match self.audio_caption_projection.as_ref() {
            Some(projection) => projection.forward(&context)?,
            None => context,
        };
        let rope =
            self.audio_rope
                .forward_for_dtype(context.device(), compute_dtype, audio_positions)?;
        Ok(LtxPreparedModalityStatic {
            context_mask: Ltx2AvTransformer3DModel::prepare_context_mask(
                audio_encoder_attention_mask,
                compute_dtype,
            )?,
            self_attention_mask: Ltx2AvTransformer3DModel::prepare_self_attention_mask(
                audio_self_attention_mask,
                compute_dtype,
            )?,
            context,
            rope,
            // No video branch exists, so there is nothing to cross-attend to.
            cross_rope: None,
        })
    }

    #[allow(dead_code)]
    pub fn forward_with_static_inputs(
        &self,
        audio_hidden_states: &Tensor,
        audio_sigma: &Tensor,
        audio_timestep: &Tensor,
        static_inputs: &LtxPreparedModalityStatic,
        perturbations: Option<&BatchedPerturbationConfig>,
    ) -> Result<Tensor> {
        self.forward_with_static_inputs_and_progress(
            audio_hidden_states,
            audio_sigma,
            audio_timestep,
            static_inputs,
            perturbations,
            "Evaluating audio transformer",
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_with_static_inputs_and_progress(
        &self,
        audio_hidden_states: &Tensor,
        audio_sigma: &Tensor,
        audio_timestep: &Tensor,
        static_inputs: &LtxPreparedModalityStatic,
        perturbations: Option<&BatchedPerturbationConfig>,
        progress_name: &str,
        progress: Option<&crate::progress::ProgressCallback>,
    ) -> Result<Tensor> {
        let compute_dtype = self.compute_dtype();
        let audio_sigma = audio_sigma.to_dtype(compute_dtype)?.affine(1000.0, 0.0)?;
        let audio_timestep = audio_timestep
            .to_dtype(compute_dtype)?
            .affine(1000.0, 0.0)?;

        let x = self
            .audio_patchify_proj
            .forward(&audio_hidden_states.to_dtype(compute_dtype)?)?;
        let batch = x.dim(0)?;
        let (timesteps, embedded_timestep) = self
            .audio_adaln_single
            .forward(&audio_timestep.flatten_all()?)?;
        let timesteps = Ltx2AvTransformer3DModel::reshape_adaln_output(&timesteps, batch)?;
        let embedded_timestep =
            Ltx2AvTransformer3DModel::reshape_adaln_output(&embedded_timestep, batch)?;
        let prompt_timestep = match self.audio_prompt_adaln_single.as_ref() {
            Some(prompt_adaln_single) => {
                let (prompt_timestep, _) =
                    prompt_adaln_single.forward(&audio_sigma.flatten_all()?)?;
                Some(Ltx2AvTransformer3DModel::reshape_adaln_output(
                    &prompt_timestep,
                    batch,
                )?)
            }
            None => None,
        };

        let mut audio = LtxPreparedModality {
            x,
            context: static_inputs.context.clone(),
            context_mask: static_inputs.context_mask.clone(),
            self_attention_mask: static_inputs.self_attention_mask.clone(),
            timesteps,
            embedded_timestep,
            rope: static_inputs.rope.clone(),
            cross_rope: None,
            cross_scale_shift_timestep: None,
            cross_gate_timestep: None,
            prompt_timestep,
        };

        let perturbations = perturbations
            .cloned()
            .unwrap_or_else(|| BatchedPerturbationConfig::empty(batch));

        for (index, block) in self.transformer_blocks.iter().enumerate() {
            if ltx2_block_debug_enabled() {
                eprintln!("[ltx2-block-debug] enter audio block={index}");
            }
            audio.x = block.forward(index, &audio, &perturbations)?;
            if let Some(progress) = progress {
                progress(crate::progress::ProgressEvent::StageProgress {
                    name: progress_name.to_string(),
                    current: index + 1,
                    total: self.transformer_blocks.len(),
                });
            }
            if ltx2_block_debug_enabled() {
                let (a_mean, a_abs_mean, a_abs_max) = tensor_debug_stats(&audio.x)?;
                eprintln!(
                    "[ltx2-block-debug] block={index} audio(mean={a_mean:.6}, abs_mean={a_abs_mean:.6}, abs_max={a_abs_max:.6})"
                );
            }
        }

        Ltx2AvTransformer3DModel::process_output(
            &self.audio_scale_shift_table,
            &self.audio_norm_out,
            &self.audio_proj_out,
            &audio.x,
            &audio.embedded_timestep,
        )
    }

    #[allow(dead_code)]
    pub fn config(&self) -> &Ltx2VideoTransformer3DModelConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;

    use super::{Ltx2AudioTransformerModel, LtxAudioTransformerBlock};
    use crate::ltx2::guidance::{
        BatchedPerturbationConfig, Perturbation, PerturbationConfig, PerturbationType,
    };
    use crate::ltx2::model::video_transformer::tests::{
        assert_tensors_close, av_transformer_tensors, av_transformer_var_builder_with_options,
        tiny_av_config,
    };
    use crate::ltx2::model::video_transformer::{
        Ltx2AvTransformer3DModel, Ltx2VideoRotaryPosEmbed, Ltx2VideoTransformer3DModelConfig,
        LtxAvTransformerBlock, LtxPreparedModality,
    };

    const TOKENS: usize = 6;
    const BATCH: usize = 1;

    fn patterned(dims: &[usize], offset: usize) -> Tensor {
        let len: usize = dims.iter().product();
        let values: Vec<f32> = (0..len)
            .map(|index| ((index + offset) as f32 * 0.037).sin())
            .collect();
        Tensor::from_vec(values, dims, &Device::Cpu).unwrap()
    }

    /// `[batch, position_dims, tokens, 2]` — the `(start, end)` pixel-coord
    /// layout `get_pixel_coords` emits and the RoPE embed consumes.
    fn positions(position_dims: usize, tokens: usize) -> Tensor {
        let mut values = Vec::with_capacity(position_dims * tokens * 2);
        for _ in 0..position_dims {
            for token in 0..tokens {
                values.push(token as f32 * 0.25);
                values.push((token + 1) as f32 * 0.25);
            }
        }
        Tensor::from_vec(values, (BATCH, position_dims, tokens, 2), &Device::Cpu).unwrap()
    }

    fn audio_rope(config: &Ltx2VideoTransformer3DModelConfig) -> (Tensor, Tensor) {
        let audio_dim = config.audio_num_attention_heads * config.audio_attention_head_dim;
        Ltx2VideoRotaryPosEmbed::new(
            audio_dim,
            config.positional_embedding_theta,
            config.audio_positional_embedding_max_pos.clone(),
            config.use_middle_indices_grid,
            config.audio_num_attention_heads,
            config.rope_type,
            config.double_precision_rope,
        )
        .forward_for_dtype(&Device::Cpu, DType::F32, &positions(1, TOKENS))
        .unwrap()
    }

    /// Build the audio half of a prepared modality exactly as
    /// `Ltx2AvTransformer3DModel::forward_with_static_inputs` would, so both
    /// blocks see byte-identical inputs.
    fn prepared_audio(config: &Ltx2VideoTransformer3DModelConfig) -> LtxPreparedModality {
        let audio_dim = config.audio_num_attention_heads * config.audio_attention_head_dim;
        let modulation_rows = if config.cross_attention_adaln { 9 } else { 6 };
        LtxPreparedModality {
            x: patterned(&[BATCH, TOKENS, audio_dim], 0),
            context: patterned(&[BATCH, 4, config.audio_cross_attention_dim], 11),
            context_mask: None,
            self_attention_mask: None,
            timesteps: patterned(&[BATCH, 1, modulation_rows * audio_dim], 23),
            embedded_timestep: patterned(&[BATCH, 1, 2 * audio_dim], 31),
            rope: audio_rope(config),
            cross_rope: None,
            cross_scale_shift_timestep: None,
            cross_gate_timestep: None,
            prompt_timestep: config
                .cross_attention_adaln
                .then(|| patterned(&[BATCH, 1, 2 * audio_dim], 53)),
        }
    }

    fn blocks_under_test(
        config: &Ltx2VideoTransformer3DModelConfig,
        vb: &VarBuilder<'static>,
    ) -> (LtxAvTransformerBlock, LtxAudioTransformerBlock) {
        let block_vb = vb.pp("transformer_blocks").pp("0");
        let av = LtxAvTransformerBlock::new(
            config,
            block_vb.clone(),
            None,
            "diffusion_model.transformer_blocks.0",
            None,
        )
        .unwrap();
        let audio_only = LtxAudioTransformerBlock::new(
            config,
            block_vb,
            None,
            "diffusion_model.transformer_blocks.0",
            None,
        )
        .unwrap();
        (av, audio_only)
    }

    /// The guard that keeps the audio-only port from drifting away from the
    /// shipped AV block. `LtxAvTransformerBlock::forward(idx, None, Some(a),
    /// …)` already runs audio-only correctly — it only bails when *both*
    /// modalities are absent — so it is the reference implementation, and the
    /// audio-only block exists purely to avoid instantiating the video
    /// weights. If the two ever disagree, this fails before a T2A render can
    /// quietly produce different audio from the joint pipeline.
    #[test]
    fn audio_only_block_matches_av_block_audio_branch() {
        for cross_attention_adaln in [false, true] {
            let config = Ltx2VideoTransformer3DModelConfig {
                cross_attention_adaln,
                ..tiny_av_config()
            };
            let vb = av_transformer_var_builder_with_options(config.clone(), false);
            let (av, audio_only) = blocks_under_test(&config, &vb);
            let audio = prepared_audio(&config);
            let perturbations = BatchedPerturbationConfig::empty(BATCH);

            let (video_out, av_audio_out) =
                av.forward(0, None, Some(&audio), &perturbations).unwrap();
            assert!(
                video_out.is_none(),
                "the AV block must emit no video output when the video modality is absent"
            );
            let av_audio_out = av_audio_out.expect("AV block should emit audio output");
            let audio_out = audio_only.forward(0, &audio, &perturbations).unwrap();

            assert_eq!(audio_out.dims(), av_audio_out.dims());
            assert_tensors_close(&audio_out, &av_audio_out, 1e-5);
        }
    }

    /// STG perturbs the audio self-attention of specific blocks. Both
    /// implementations must honour the same perturbation batch, or T2A's
    /// spatiotemporal-guidance pass would silently become a second
    /// conditional pass.
    #[test]
    fn audio_only_block_matches_av_block_under_stg_perturbation() {
        let config = tiny_av_config();
        let vb = av_transformer_var_builder_with_options(config.clone(), false);
        let (av, audio_only) = blocks_under_test(&config, &vb);
        let audio = prepared_audio(&config);
        let perturbations =
            BatchedPerturbationConfig::new(vec![PerturbationConfig::new(vec![Perturbation::new(
                PerturbationType::SkipAudioSelfAttention,
                Some(vec![0]),
            )])]);

        let (_, av_audio_out) = av.forward(0, None, Some(&audio), &perturbations).unwrap();
        let av_audio_out = av_audio_out.expect("AV block should emit audio output");
        let audio_out = audio_only.forward(0, &audio, &perturbations).unwrap();
        assert_tensors_close(&audio_out, &av_audio_out, 1e-5);
    }

    /// Whether a checkpoint key belongs to the audio-only model.
    ///
    /// Mirrors upstream's `LTXV_AUDIO_ONLY_MODEL_COMFY_RENAMING_MAP`
    /// (`ltx-core/model/transformer/model_configurator.py:207`), plus
    /// `audio_caption_projection`: mold builds that projection from the
    /// checkpoint whenever `caption_projection_in_transformer` is set, so
    /// unlike upstream's map it must be readable.
    fn is_audio_only_key(key: &str) -> bool {
        const AUDIO_MARKERS: [&str; 9] = [
            "audio_attn1",
            "audio_attn2",
            "audio_ff",
            "audio_patchify",
            "audio_proj_out",
            "audio_adaln_single",
            "audio_prompt",
            "audio_scale_shift_table",
            "audio_caption_projection",
        ];
        // `audio_to_video_attn` and `video_to_audio_attn` are cross-modal, not
        // audio-branch, and must not survive the filter.
        if key.contains("audio_to_video") || key.contains("video_to_audio") {
            return false;
        }
        AUDIO_MARKERS.iter().any(|marker| key.contains(marker))
    }

    /// The load-time half of the reason this module exists: a T2A run must not
    /// instantiate the video transformer. On a real 19B checkpoint the
    /// difference is ~3.5 GB eagerly resident against ~19 GB streamed, so
    /// "did it accidentally build the AV model" is exactly the failure that
    /// would show up as a 20 GB peak instead of a fast render.
    ///
    /// Feeding a var-builder that contains only the `audio_*` keys makes that
    /// mechanical: the audio-only model must load, and the AV model must fail
    /// because its video weights are absent.
    #[test]
    fn audio_only_model_loads_from_audio_weights_alone() {
        let config = tiny_av_config();
        let all_tensors = av_transformer_tensors(config.clone(), false);
        let audio_tensors: HashMap<String, Tensor> = all_tensors
            .iter()
            .filter(|(key, _)| is_audio_only_key(key))
            .map(|(key, tensor)| (key.clone(), tensor.clone()))
            .collect();
        assert!(
            audio_tensors.len() < all_tensors.len(),
            "the filter must actually drop the video weights"
        );

        let audio_vb = VarBuilder::from_tensors(audio_tensors.clone(), DType::F32, &Device::Cpu);
        Ltx2AudioTransformerModel::new(&config, audio_vb.clone(), None)
            .expect("audio-only model must load from the audio weights alone");

        assert!(
            Ltx2AvTransformer3DModel::new(&config, audio_vb, None).is_err(),
            "the AV model must fail on an audio-only weight map — if it loads, the filter \
             is letting video weights through and this test proves nothing"
        );
    }

    /// End-to-end wrapper shape: the audio-only model turns audio latents into
    /// per-token velocities in `audio_out_channels`, which is what the
    /// sampler and the audio VAE expect.
    #[test]
    fn audio_only_model_emits_audio_channel_velocities() {
        let config = tiny_av_config();
        let vb = av_transformer_var_builder_with_options(config.clone(), false);
        let audio_only = Ltx2AudioTransformerModel::new(&config, vb, None).unwrap();

        let audio_latents = patterned(&[BATCH, TOKENS, config.audio_in_channels], 3);
        let audio_context = patterned(&[BATCH, 4, config.caption_channels], 13);
        let sigma = Tensor::full(0.5f32, (BATCH,), &Device::Cpu).unwrap();
        let timestep = Tensor::full(0.5f32, (BATCH, 1), &Device::Cpu).unwrap();

        let static_inputs = audio_only
            .prepare_static_inputs(&audio_context, None, None, &positions(1, TOKENS))
            .unwrap();
        let out = audio_only
            .forward_with_static_inputs(&audio_latents, &sigma, &timestep, &static_inputs, None)
            .unwrap();

        assert_eq!(out.dims(), &[BATCH, TOKENS, config.audio_out_channels]);
        assert!(
            out.flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .iter()
                .all(|value| value.is_finite()),
            "audio velocities must be finite"
        );
    }
}
