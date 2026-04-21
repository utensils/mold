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
    #[allow(clippy::too_many_arguments)]
    fn load(
        encoder_path: &PathBuf,
        tokenizer_path: &PathBuf,
        config: ClipConfig,
        max_position_embeddings: usize,
        device: &candle_core::Device,
        dtype: DType,
        component: &str,
        progress: &crate::progress::ProgressReporter,
    ) -> Result<Self> {
        let vb = crate::weight_loader::load_safetensors_with_progress(
            std::slice::from_ref(encoder_path),
            dtype,
            device,
            component,
            progress,
        )?;
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

        let raw_tokens = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow::anyhow!("CLIP tokenization failed: {e}"))?
            .get_ids()
            .to_vec();

        let (tokens, eos_position) =
            prepare_clip_tokens(raw_tokens, self.max_position_embeddings, pad_id);

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

    fn reload(
        &mut self,
        encoder_path: &PathBuf,
        dtype: DType,
        component: &str,
        progress: &crate::progress::ProgressReporter,
    ) -> Result<()> {
        let vb = crate::weight_loader::load_safetensors_with_progress(
            std::slice::from_ref(encoder_path),
            dtype,
            &self.device,
            component,
            progress,
        )?;
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
        progress: &crate::progress::ProgressReporter,
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
            "SD3 CLIP-L",
            progress,
        )?;

        // Load CLIP-G text_projection from the CLIP-G weights
        let clip_g_vb = crate::weight_loader::load_safetensors_with_progress(
            std::slice::from_ref(clip_g_path),
            dtype,
            device,
            "SD3 CLIP-G projection",
            progress,
        )?;
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
            "SD3 CLIP-G",
            progress,
        )?;

        let on_gpu = crate::device::is_gpu(device);

        // T5 encoder
        let t5 = T5Encoder::load(t5_path, t5_tokenizer_path, device, dtype, progress)?;

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
    pub fn reload(
        &mut self,
        dtype: DType,
        progress: &crate::progress::ProgressReporter,
    ) -> Result<()> {
        self.clip_l
            .reload(&self.clip_l_path, dtype, "SD3 CLIP-L", progress)?;
        self.clip_g
            .reload(&self.clip_g_path, dtype, "SD3 CLIP-G", progress)?;
        self.t5.reload(&self.t5_path, dtype, progress)?;
        Ok(())
    }

    /// Check if encoder weights are currently loaded.
    pub fn is_loaded(&self) -> bool {
        self.clip_l.model.is_some() && self.clip_g.model.is_some() && self.t5.model.is_some()
    }
}

/// Prepare a CLIP token sequence for the fixed position-embedding window.
///
/// CLIP's position-embedding table holds exactly `max_len` entries, so a token
/// tensor longer than that fails inside candle's `broadcast_add` when the
/// position embeddings are applied. This helper:
///
/// - Truncates overlong sequences to `max_len`, copying the trailing token
///   (the tokenizer's EOS, assuming `add_special_tokens=true`) into the last
///   slot so the pooled-output path still reads an EOS-position hidden state.
/// - Pads short sequences up to `max_len` with `pad_id`.
/// - Returns the final `tokens` vector and the `eos_position` index the caller
///   uses to slice the pooled output.
fn prepare_clip_tokens(mut raw_tokens: Vec<u32>, max_len: usize, pad_id: u32) -> (Vec<u32>, usize) {
    let original_len = raw_tokens.len();

    if original_len > max_len {
        let eos_id = *raw_tokens
            .last()
            .expect("original_len > max_len implies non-empty");
        raw_tokens.truncate(max_len);
        if let Some(last) = raw_tokens.last_mut() {
            *last = eos_id;
        }
        tracing::debug!(
            "SD3 CLIP prompt exceeded {} tokens ({} raw); truncated with EOS preserved",
            max_len,
            original_len,
        );
    }

    let eos_position = raw_tokens.len().saturating_sub(1);

    while raw_tokens.len() < max_len {
        raw_tokens.push(pad_id);
    }

    (raw_tokens, eos_position)
}

#[cfg(test)]
mod tests {
    use super::prepare_clip_tokens;

    const MAX_LEN: usize = 77;
    const PAD_ID: u32 = 0;
    const EOS_ID: u32 = 49407;

    #[test]
    fn pads_short_prompt_to_max_len() {
        let raw = vec![49406, 10, 20, 30, EOS_ID]; // 5 tokens, last is EOS
        let (tokens, eos) = prepare_clip_tokens(raw, MAX_LEN, PAD_ID);
        assert_eq!(tokens.len(), MAX_LEN, "must pad up to max_len");
        assert_eq!(eos, 4, "eos_position tracks the raw EOS slot");
        assert_eq!(tokens[4], EOS_ID, "EOS preserved at original position");
        assert_eq!(tokens[5], PAD_ID, "pads follow the real tokens");
        assert_eq!(*tokens.last().unwrap(), PAD_ID);
    }

    #[test]
    fn leaves_exact_length_untouched() {
        let mut raw: Vec<u32> = (1..MAX_LEN as u32).collect();
        raw.push(EOS_ID);
        assert_eq!(raw.len(), MAX_LEN);
        let (tokens, eos) = prepare_clip_tokens(raw.clone(), MAX_LEN, PAD_ID);
        assert_eq!(tokens.len(), MAX_LEN);
        assert_eq!(eos, MAX_LEN - 1);
        assert_eq!(tokens, raw);
    }

    #[test]
    fn truncates_overlong_prompt_preserving_eos() {
        // 132-token sequence — matches the shapes in the original bug report
        // ([1, 132, 768] vs [1, 77, 768]).
        let mut raw: Vec<u32> = (1..=131).collect();
        raw.push(EOS_ID);
        assert_eq!(raw.len(), 132);

        let (tokens, eos) = prepare_clip_tokens(raw, MAX_LEN, PAD_ID);

        assert_eq!(tokens.len(), MAX_LEN, "overlong sequence must be truncated");
        assert_eq!(eos, MAX_LEN - 1, "eos_position must land on the last slot");
        assert_eq!(
            tokens[MAX_LEN - 1],
            EOS_ID,
            "EOS must be preserved in the final slot so pooled output reads EOS hidden state",
        );
    }

    #[test]
    fn handles_empty_input() {
        // Degenerate case: tokenizer somehow returns no ids. Shouldn't panic.
        let (tokens, eos) = prepare_clip_tokens(Vec::new(), MAX_LEN, PAD_ID);
        assert_eq!(tokens.len(), MAX_LEN);
        assert_eq!(eos, 0);
        assert!(tokens.iter().all(|t| *t == PAD_ID));
    }
}
