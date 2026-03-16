//! Qwen3 text encoder wrapper for Z-Image.
//!
//! Wraps either the BF16 `ZImageTextEncoder` (from candle) or the quantized
//! `GgufQwen3Encoder` (from this crate), providing a unified load/encode/drop/reload
//! interface that mirrors `T5Encoder`.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::z_image::{TextEncoderConfig, ZImageTextEncoder};
use std::path::{Path, PathBuf};

use super::qwen3_gguf::GgufQwen3Encoder;

/// BF16 (safetensors) or quantized (GGUF) Qwen3 text encoder.
pub(crate) enum Qwen3Model {
    BF16(ZImageTextEncoder),
    Quantized(GgufQwen3Encoder),
}

impl Qwen3Model {
    pub fn forward(&mut self, input_ids: &Tensor) -> Result<Tensor> {
        match self {
            Self::BF16(m) => Ok(m.forward(input_ids)?),
            Self::Quantized(m) => m.forward(input_ids),
        }
    }
}

/// Reusable Qwen3 text encoder wrapper.
///
/// Holds the model weights (optionally — `None` when dropped to free VRAM),
/// the tokenizer, and device placement info.
pub(crate) struct Qwen3Encoder {
    pub model: Option<Qwen3Model>,
    pub tokenizer: tokenizers::Tokenizer,
    pub device: Device,
    pub on_gpu: bool,
    pub is_quantized: bool,
    /// Paths needed for reload.
    encoder_paths: Vec<PathBuf>,
    dtype: DType,
}

/// Format a user prompt for the Qwen3 chat template used by Z-Image.
fn format_prompt_for_qwen3(prompt: &str) -> String {
    format!(
        "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        prompt
    )
}

impl Qwen3Encoder {
    /// Load a BF16 Qwen3 encoder from safetensors shards.
    pub fn load_bf16(
        encoder_paths: &[PathBuf],
        tokenizer_path: &PathBuf,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let te_cfg = TextEncoderConfig::z_image();
        let path_strs: Vec<&str> = encoder_paths
            .iter()
            .map(|p| p.to_str().expect("non-UTF8 path"))
            .collect();
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&path_strs, dtype, device)? };
        let model = Qwen3Model::BF16(ZImageTextEncoder::new(&te_cfg, vb)?);

        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("failed to load Qwen3 tokenizer: {e}"))?;
        let on_gpu = device.is_cuda() || device.is_metal();

        Ok(Self {
            model: Some(model),
            tokenizer,
            device: device.clone(),
            on_gpu,
            is_quantized: false,
            encoder_paths: encoder_paths.to_vec(),
            dtype,
        })
    }

    /// Load a quantized Qwen3 encoder from a GGUF file.
    pub fn load_gguf(gguf_path: &Path, tokenizer_path: &PathBuf, device: &Device) -> Result<Self> {
        let model = Qwen3Model::Quantized(GgufQwen3Encoder::load(gguf_path, device)?);
        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("failed to load Qwen3 tokenizer: {e}"))?;
        let on_gpu = device.is_cuda() || device.is_metal();

        Ok(Self {
            model: Some(model),
            tokenizer,
            device: device.clone(),
            on_gpu,
            is_quantized: true,
            encoder_paths: vec![gguf_path.to_path_buf()],
            dtype: DType::F32, // GGUF dequantizes to F32
        })
    }

    /// Encode a text prompt into Qwen3 embeddings.
    /// Applies the Qwen3 chat template, tokenizes, runs the forward pass,
    /// and moves the result to `target_device` with `target_dtype`.
    /// Returns (embeddings, token_count).
    pub fn encode(
        &mut self,
        prompt: &str,
        target_device: &Device,
        target_dtype: DType,
    ) -> Result<(Tensor, usize)> {
        let model = self
            .model
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("Qwen3 model unavailable (weights dropped)"))?;

        let formatted = format_prompt_for_qwen3(prompt);
        let tokens = self
            .tokenizer
            .encode(formatted.as_str(), true)
            .map_err(|e| anyhow::anyhow!("Qwen3 tokenization failed: {e}"))?
            .get_ids()
            .to_vec();

        let token_count = tokens.len();
        let input_ids = Tensor::from_vec(tokens, (1, token_count), &self.device)?;

        let emb = model.forward(&input_ids)?;
        let emb = emb.to_device(target_device)?.to_dtype(target_dtype)?;
        Ok((emb, token_count))
    }

    /// Drop model weights to free memory (e.g. GPU VRAM after encoding).
    pub fn drop_weights(&mut self) {
        self.model = None;
    }

    /// Reload model weights (e.g. for the next generation after being dropped).
    pub fn reload(&mut self) -> Result<()> {
        if self.is_quantized {
            self.model = Some(Qwen3Model::Quantized(GgufQwen3Encoder::load(
                &self.encoder_paths[0],
                &self.device,
            )?));
        } else {
            let te_cfg = TextEncoderConfig::z_image();
            let path_strs: Vec<&str> = self
                .encoder_paths
                .iter()
                .map(|p| p.to_str().expect("non-UTF8 path"))
                .collect();
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(&path_strs, self.dtype, &self.device)?
            };
            self.model = Some(Qwen3Model::BF16(ZImageTextEncoder::new(&te_cfg, vb)?));
        }
        Ok(())
    }
}
