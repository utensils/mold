//! SD3 Triple Encoder: CLIP-L + CLIP-G + T5-XXL
//!
//! SD3 uses three text encoders simultaneously:
//! - CLIP-L (768-dim): penultimate hidden states + pooled output
//! - CLIP-G (1280-dim): penultimate hidden states + pooled output (with text_projection)
//! - T5-XXL (4096-dim): full text embeddings
//!
//! The outputs are combined into:
//! - `context`: [clip_l_penult || clip_g_penult padded to 4096, t5_embeddings] along seq dim
//! - `y`: [clip_l_pooled, clip_g_pooled] concatenated (2048-dim vector)
//!
//! CLIP-L uses `Config::sdxl()`, CLIP-G uses `Config::sdxl2()`.
//! Both use `forward_until_encoder_layer(&tokens, usize::MAX, -2)` for penultimate hidden states.

use anyhow::Result;
use candle_core::{DType, IndexOp, Module, Tensor, D};
use candle_nn::VarBuilder;
use candle_transformers::models::stable_diffusion::clip::{
    self, ClipTextTransformer, Config as ClipConfig,
};
use std::path::PathBuf;
use tokenizers::Tokenizer;

use super::t5::T5Encoder;

/// Loaded CLIP encoder with tokenizer for SD3's triple encoding.
struct ClipWithTokenizer {
    model: Option<ClipTextTransformer>,
    config: ClipConfig,
    tokenizer: Tokenizer,
    max_position_embeddings: usize,
    device: candle_core::Device,
}

impl ClipWithTokenizer {
    fn load(
        encoder_path: &PathBuf,
        tokenizer_path: &PathBuf,
        config: ClipConfig,
        max_position_embeddings: usize,
        device: &candle_core::Device,
        dtype: DType,
    ) -> Result<Self> {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(std::slice::from_ref(encoder_path), dtype, device)?
        };
        let model = ClipTextTransformer::new(vb, &config)?;
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("failed to load CLIP tokenizer: {e}"))?;

        Ok(Self {
            model: Some(model),
            config,
            tokenizer,
            max_position_embeddings,
            device: device.clone(),
        })
    }

    /// Encode text to (penultimate_hidden_states, pooled_output).
    fn encode_text_to_embedding(&self, prompt: &str) -> Result<(Tensor, Tensor)> {
        let clip = self
            .model
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("CLIP model unavailable (weights dropped)"))?;

        let pad_id = match &self.config.pad_with {
            Some(padding) => *self
                .tokenizer
                .get_vocab(true)
                .get(padding.as_str())
                .ok_or_else(|| anyhow::anyhow!("Failed to tokenize CLIP padding"))?,
            None => *self
                .tokenizer
                .get_vocab(true)
                .get("<|endoftext|>")
                .ok_or_else(|| anyhow::anyhow!("Failed to tokenize CLIP end-of-text"))?,
        };

        let mut tokens = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow::anyhow!("CLIP tokenization failed: {e}"))?
            .get_ids()
            .to_vec();

        let eos_position = tokens.len() - 1;

        while tokens.len() < self.max_position_embeddings {
            tokens.push(pad_id);
        }
        let tokens = Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;
        let (_text_embeddings, text_embeddings_penultimate) =
            clip.forward_until_encoder_layer(&tokens, usize::MAX, -2)?;
        // Get pooled output from the last layer at EOS position
        let text_embeddings_pooled = {
            let (last_hidden, _) = clip.forward_until_encoder_layer(&tokens, usize::MAX, 0)?;
            last_hidden.i((0, eos_position, ..))?
        };

        Ok((text_embeddings_penultimate, text_embeddings_pooled))
    }

    fn drop_weights(&mut self) {
        self.model = None;
    }

    fn reload(&mut self, encoder_path: &PathBuf, dtype: DType) -> Result<()> {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                std::slice::from_ref(encoder_path),
                dtype,
                &self.device,
            )?
        };
        self.model = Some(ClipTextTransformer::new(vb, &self.config)?);
        Ok(())
    }
}

/// SD3 Triple Encoder combining CLIP-L, CLIP-G, and T5-XXL.
///
/// Produces the (context, y) tensors needed by the MMDiT transformer:
/// - `context`: text embeddings for cross-attention
/// - `y`: vector conditioning from pooled CLIP outputs
pub(crate) struct SD3TripleEncoder {
    clip_l: ClipWithTokenizer,
    clip_g: ClipWithTokenizer,
    clip_g_text_projection: candle_nn::Linear,
    t5: T5Encoder,
    /// Path to CLIP-L weights (for reload)
    clip_l_path: PathBuf,
    /// Path to CLIP-G weights (for reload)
    clip_g_path: PathBuf,
    /// Path to T5 weights (for reload)
    t5_path: PathBuf,
    /// Whether encoders are on GPU
    pub on_gpu: bool,
}

impl SD3TripleEncoder {
    /// Load all three encoders from separate weight files.
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        clip_l_path: &PathBuf,
        clip_l_tokenizer_path: &PathBuf,
        clip_g_path: &PathBuf,
        clip_g_tokenizer_path: &PathBuf,
        t5_path: &PathBuf,
        t5_tokenizer_path: &PathBuf,
        device: &candle_core::Device,
        dtype: DType,
    ) -> Result<Self> {
        let max_position_embeddings = 77usize;

        // CLIP-L uses SDXL config (768-dim)
        let clip_l = ClipWithTokenizer::load(
            clip_l_path,
            clip_l_tokenizer_path,
            clip::Config::sdxl(),
            max_position_embeddings,
            device,
            dtype,
        )?;

        // Load CLIP-G text_projection from the CLIP-G weights
        let clip_g_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(std::slice::from_ref(clip_g_path), dtype, device)?
        };
        let text_projection =
            candle_nn::linear_no_bias(1280, 1280, clip_g_vb.pp("text_projection"))?;

        // CLIP-G uses SDXL2 config (1280-dim)
        let clip_g = ClipWithTokenizer::load(
            clip_g_path,
            clip_g_tokenizer_path,
            clip::Config::sdxl2(),
            max_position_embeddings,
            device,
            dtype,
        )?;

        let on_gpu = device.is_cuda() || device.is_metal();

        // T5 encoder
        let t5 = T5Encoder::load(t5_path, t5_tokenizer_path, device, dtype)?;

        Ok(Self {
            clip_l,
            clip_g,
            clip_g_text_projection: text_projection,
            t5,
            clip_l_path: clip_l_path.clone(),
            clip_g_path: clip_g_path.clone(),
            t5_path: t5_path.clone(),
            on_gpu,
        })
    }

    /// Encode a prompt into (context, y) tensors for the MMDiT.
    ///
    /// - `context`: [clip_embeddings_padded_to_4096 || t5_embeddings] along sequence dim
    /// - `y`: [clip_l_pooled || clip_g_pooled_projected] as 2048-dim vector
    pub fn encode(
        &mut self,
        prompt: &str,
        target_device: &candle_core::Device,
        target_dtype: DType,
    ) -> Result<(Tensor, Tensor)> {
        // CLIP-L encoding: penultimate hidden states + pooled
        let (clip_l_embeddings, clip_l_pooled) = self.clip_l.encode_text_to_embedding(prompt)?;

        // CLIP-G encoding: penultimate hidden states + pooled
        let (clip_g_embeddings, clip_g_pooled) = self.clip_g.encode_text_to_embedding(prompt)?;

        // Apply text_projection to CLIP-G pooled output
        let clip_g_pooled_projected = self
            .clip_g_text_projection
            .forward(&clip_g_pooled.unsqueeze(0)?)?
            .squeeze(0)?;

        // y = [clip_l_pooled(768), clip_g_pooled_projected(1280)] = 2048-dim
        let y = Tensor::cat(&[&clip_l_pooled, &clip_g_pooled_projected], 0)?.unsqueeze(0)?;

        // Concatenate CLIP hidden states along feature dim: [77, 768] + [77, 1280] = [77, 2048]
        // Then pad to 4096 with zeros for MMDiT context_embed_size
        let clip_embeddings_concat = Tensor::cat(
            &[&clip_l_embeddings, &clip_g_embeddings],
            D::Minus1,
        )?
        .pad_with_zeros(D::Minus1, 0, 2048)?;

        // T5 encoding
        let t5_embeddings = self
            .t5
            .encode(prompt, target_device, target_dtype)?
            .to_dtype(DType::F16)?;

        // context = [clip_concat(77, 4096), t5_embeddings(256, 4096)] along sequence dim
        let context = Tensor::cat(&[&clip_embeddings_concat, &t5_embeddings], D::Minus2)?;

        // Move to target device/dtype
        let context = context.to_device(target_device)?.to_dtype(target_dtype)?;
        let y = y.to_device(target_device)?.to_dtype(target_dtype)?;

        Ok((context, y))
    }

    /// Drop all encoder weights to free VRAM.
    pub fn drop_weights(&mut self) {
        self.clip_l.drop_weights();
        self.clip_g.drop_weights();
        self.t5.drop_weights();
    }

    /// Reload all encoder weights (e.g. for next generation after being dropped).
    pub fn reload(&mut self, dtype: DType) -> Result<()> {
        self.clip_l.reload(&self.clip_l_path, dtype)?;
        self.clip_g.reload(&self.clip_g_path, dtype)?;
        self.t5.reload(&self.t5_path, dtype)?;
        Ok(())
    }

    /// Check if encoder weights are currently loaded.
    pub fn is_loaded(&self) -> bool {
        self.clip_l.model.is_some() && self.clip_g.model.is_some() && self.t5.model.is_some()
    }
}
