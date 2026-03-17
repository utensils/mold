//! Qwen2.5-VL text encoder adapter for Qwen-Image-2512.
//!
//! Wraps candle's `qwen2::Model` to provide text encoding for the Qwen-Image pipeline.
//! Only the text path of the Qwen2.5-VL model is used (vision components are ignored).
//!
//! Key differences from Qwen3 (Z-Image) encoder:
//! - hidden_size=3584 (vs 2560)
//! - 28 layers (vs 36)
//! - 28 attention heads, 4 KV heads (vs 32/8)
//! - Returns second-to-last layer hidden states (like Qwen3 encoder)
//! - Uses `qwen2::Model` from candle (compatible architecture)

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use std::path::PathBuf;

/// Qwen2.5 text encoder configuration for Qwen-Image-2512.
///
/// Matches the `Qwen2_5_VLForConditionalGeneration` config from HuggingFace,
/// using only the text model parameters.
pub(crate) struct Qwen2TextEncoderConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
}

impl Default for Qwen2TextEncoderConfig {
    fn default() -> Self {
        Self::qwen_image()
    }
}

impl Qwen2TextEncoderConfig {
    /// Configuration for Qwen-Image-2512 text encoder (Qwen2.5-VL 7B text path).
    pub fn qwen_image() -> Self {
        Self {
            vocab_size: 152064,
            hidden_size: 3584,
            intermediate_size: 18944,
            num_hidden_layers: 28,
            num_attention_heads: 28,
            num_key_value_heads: 4,
            max_position_embeddings: 128000,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
        }
    }

    /// Convert to candle's `qwen2::Config`.
    fn to_candle_config(&self) -> candle_transformers::models::qwen2::Config {
        candle_transformers::models::qwen2::Config {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            max_position_embeddings: self.max_position_embeddings,
            // Qwen2 model defaults — not used for text encoding but needed for struct
            sliding_window: self.max_position_embeddings,
            max_window_layers: 0,
            tie_word_embeddings: false,
            rope_theta: self.rope_theta,
            rms_norm_eps: self.rms_norm_eps,
            use_sliding_window: false,
            hidden_act: candle_nn::Activation::Silu,
        }
    }
}

/// Qwen2.5 text encoder model for Qwen-Image.
///
/// Uses candle's `qwen2::Model` internally and returns the second-to-last
/// layer hidden states (matching the diffusers QwenImagePipeline behavior).
pub(crate) struct Qwen2TextEncoder {
    pub model: Option<candle_transformers::models::qwen2::Model>,
    pub tokenizer: tokenizers::Tokenizer,
    pub device: Device,
    pub on_gpu: bool,
    /// Paths needed for reload after drop.
    encoder_paths: Vec<PathBuf>,
    dtype: DType,
    config: Qwen2TextEncoderConfig,
}

impl Qwen2TextEncoder {
    /// Load a BF16 Qwen2.5 text encoder from safetensors shards.
    pub fn load_bf16(
        encoder_paths: &[PathBuf],
        tokenizer_path: &PathBuf,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let config = Qwen2TextEncoderConfig::qwen_image();
        let candle_cfg = config.to_candle_config();
        let path_strs: Vec<&str> = encoder_paths
            .iter()
            .map(|p| p.to_str().expect("non-UTF8 path"))
            .collect();
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&path_strs, dtype, device)? };
        let model = candle_transformers::models::qwen2::Model::new(&candle_cfg, vb)?;

        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("failed to load Qwen2.5 tokenizer: {e}"))?;
        let on_gpu = device.is_cuda() || device.is_metal();

        Ok(Self {
            model: Some(model),
            tokenizer,
            device: device.clone(),
            on_gpu,
            encoder_paths: encoder_paths.to_vec(),
            dtype,
            config,
        })
    }

    /// Encode a text prompt into Qwen2.5 embeddings.
    ///
    /// Tokenizes the prompt, runs the forward pass through the Qwen2 model,
    /// and returns the final hidden states. The model naturally returns all
    /// layer outputs; we use the model's built-in forward which applies the
    /// final norm. The hidden_size=3584 matches joint_attention_dim directly.
    ///
    /// Returns (embeddings, token_count) where embeddings has shape (1, seq_len, 3584).
    pub fn encode(
        &mut self,
        prompt: &str,
        target_device: &Device,
        target_dtype: DType,
    ) -> Result<(Tensor, usize)> {
        let model = self
            .model
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("Qwen2.5 model unavailable (weights dropped)"))?;

        // Clear any cached KV state from previous forward passes
        model.clear_kv_cache();

        let tokens = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow::anyhow!("Qwen2.5 tokenization failed: {e}"))?
            .get_ids()
            .to_vec();

        let token_count = tokens.len();
        let input_ids = Tensor::from_vec(tokens, (1, token_count), &self.device)?;

        // Forward pass: returns (B, seq_len, hidden_size) after final norm
        // seqlen_offset=0, no attention mask (full attention for encoding)
        let emb = model.forward(&input_ids, 0, None)?;
        let emb = emb.to_device(target_device)?.to_dtype(target_dtype)?;
        Ok((emb, token_count))
    }

    /// Drop model weights to free memory (e.g. GPU VRAM after encoding).
    pub fn drop_weights(&mut self) {
        self.model = None;
    }

    /// Reload model weights from disk (e.g. for the next generation after being dropped).
    pub fn reload(&mut self) -> Result<()> {
        let candle_cfg = self.config.to_candle_config();
        let path_strs: Vec<&str> = self
            .encoder_paths
            .iter()
            .map(|p| p.to_str().expect("non-UTF8 path"))
            .collect();
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&path_strs, self.dtype, &self.device)? };
        self.model = Some(candle_transformers::models::qwen2::Model::new(
            &candle_cfg,
            vb,
        )?);
        Ok(())
    }
}
