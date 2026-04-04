//! Qwen3 text encoder wrapper.
//!
//! Wraps either the native BF16 `Bf16Qwen3Encoder` (with multi-layer extraction)
//! or the quantized `GgufQwen3Encoder`, providing a unified load/encode/drop/reload
//! interface that mirrors `T5Encoder`.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use std::path::{Path, PathBuf};

use super::qwen3_bf16::{Bf16Qwen3Encoder, Qwen3BF16Config};
use super::qwen3_gguf::GgufQwen3Encoder;

/// BF16 (safetensors) or quantized (GGUF) Qwen3 text encoder.
pub(crate) enum Qwen3Model {
    BF16(Bf16Qwen3Encoder),
    Quantized(GgufQwen3Encoder),
}

impl Qwen3Model {
    pub fn forward(&mut self, input_ids: &Tensor) -> Result<Tensor> {
        match self {
            Self::BF16(m) => m.forward(input_ids),
            Self::Quantized(m) => m.forward(input_ids),
        }
    }

    /// Run forward pass and collect hidden states from specific layer indices.
    /// Returns (B, seq_len, num_layers * hidden_size).
    /// Used by Flux.2 Klein which stacks layers 9, 18, 27 to get 7680-dim embeddings.
    pub fn forward_with_layers(
        &mut self,
        input_ids: &Tensor,
        layer_indices: &[usize],
    ) -> Result<Tensor> {
        match self {
            Self::BF16(m) => m.forward_with_layers(input_ids, layer_indices),
            Self::Quantized(m) => m.forward_with_layers(input_ids, layer_indices),
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
    /// BF16 architecture config (Qwen3-4B vs 8B). Used for BF16 reload.
    bf16_config: Qwen3BF16Config,
}

/// Format a user prompt for the Qwen3 chat template used by Z-Image.
fn format_prompt_for_qwen3(prompt: &str) -> String {
    format!(
        "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        prompt
    )
}

/// Format a user prompt for the Qwen3 chat template used by Flux.2 Klein.
///
/// Flux.2 Klein uses `enable_thinking=False` which adds an explicit empty thinking
/// block (`<think>\n\n</think>\n\n`) after the assistant prefix. This signals the
/// model to skip thinking mode and produce text encoding directly.
fn format_prompt_for_flux2(prompt: &str) -> String {
    format!("{}<think>\n\n</think>\n\n", format_prompt_for_qwen3(prompt))
}

impl Qwen3Encoder {
    /// Load a BF16 Qwen3 encoder from safetensors shards.
    ///
    /// The `bf16_config` selects the architecture variant: `Qwen3BF16Config::qwen3_4b()`
    /// for Klein-4B / Z-Image, or `Qwen3BF16Config::qwen3_8b()` for Klein-9B.
    pub fn load_bf16(
        encoder_paths: &[PathBuf],
        tokenizer_path: &PathBuf,
        device: &Device,
        dtype: DType,
        bf16_config: &Qwen3BF16Config,
        progress: &crate::progress::ProgressReporter,
    ) -> Result<Self> {
        let vb = crate::weight_loader::load_safetensors_with_progress(
            encoder_paths,
            dtype,
            device,
            "Qwen3 encoder",
            progress,
        )?;
        let model = Qwen3Model::BF16(Bf16Qwen3Encoder::load(bf16_config, vb)?);

        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("failed to load Qwen3 tokenizer: {e}"))?;
        let on_gpu = crate::device::is_gpu(device);

        Ok(Self {
            model: Some(model),
            tokenizer,
            device: device.clone(),
            on_gpu,
            is_quantized: false,
            encoder_paths: encoder_paths.to_vec(),
            dtype,
            bf16_config: *bf16_config,
        })
    }

    /// Load a quantized Qwen3 encoder from a GGUF file.
    ///
    /// The `bf16_config` is stored for potential BF16 fallback reload but is not
    /// used during GGUF loading (GGUF reads dimensions from file metadata).
    pub fn load_gguf(
        gguf_path: &Path,
        tokenizer_path: &PathBuf,
        device: &Device,
        bf16_config: &Qwen3BF16Config,
    ) -> Result<Self> {
        let model = Qwen3Model::Quantized(GgufQwen3Encoder::load(gguf_path, device)?);
        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("failed to load Qwen3 tokenizer: {e}"))?;
        let on_gpu = crate::device::is_gpu(device);

        Ok(Self {
            model: Some(model),
            tokenizer,
            device: device.clone(),
            on_gpu,
            is_quantized: true,
            encoder_paths: vec![gguf_path.to_path_buf()],
            dtype: DType::F32, // GGUF dequantizes to F32
            bf16_config: *bf16_config,
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

    /// Encode a text prompt, extracting hidden states from specific layers.
    /// Returns (stacked_embeddings, token_count) where stacked_embeddings has
    /// shape (B, seq_len, num_layers * hidden_size).
    ///
    /// Uses the Flux.2 Klein chat template (with empty thinking block) since
    /// only Flux.2 Klein calls this method.
    pub fn encode_with_layers(
        &mut self,
        prompt: &str,
        target_device: &Device,
        target_dtype: DType,
        layer_indices: &[usize],
    ) -> Result<(Tensor, usize)> {
        let model = self
            .model
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("Qwen3 model unavailable (weights dropped)"))?;

        let formatted = format_prompt_for_flux2(prompt);
        let tokens = self
            .tokenizer
            .encode(formatted.as_str(), true)
            .map_err(|e| anyhow::anyhow!("Qwen3 tokenization failed: {e}"))?
            .get_ids()
            .to_vec();

        let token_count = tokens.len();
        let input_ids = Tensor::from_vec(tokens, (1, token_count), &self.device)?;

        let emb = model.forward_with_layers(&input_ids, layer_indices)?;
        let emb = emb.to_device(target_device)?.to_dtype(target_dtype)?;
        Ok((emb, token_count))
    }

    /// Drop model weights to free memory (e.g. GPU VRAM after encoding).
    pub fn drop_weights(&mut self) {
        self.model = None;
    }

    /// Reload model weights (e.g. for the next generation after being dropped).
    pub fn reload(&mut self, progress: &crate::progress::ProgressReporter) -> Result<()> {
        if self.is_quantized {
            self.model = Some(Qwen3Model::Quantized(GgufQwen3Encoder::load(
                &self.encoder_paths[0],
                &self.device,
            )?));
        } else {
            let vb = crate::weight_loader::load_safetensors_with_progress(
                &self.encoder_paths,
                self.dtype,
                &self.device,
                "Qwen3 encoder",
                progress,
            )?;
            self.model = Some(Qwen3Model::BF16(Bf16Qwen3Encoder::load(
                &self.bf16_config,
                vb,
            )?));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn z_image_chat_template() {
        let result = format_prompt_for_qwen3("a cat");
        assert!(result.starts_with("<|im_start|>user\n"));
        assert!(result.contains("a cat"));
        assert!(result.ends_with("<|im_start|>assistant\n"));
        assert!(!result.contains("<think>"));
    }

    #[test]
    fn flux2_chat_template_includes_thinking() {
        let result = format_prompt_for_flux2("a sunset");
        assert!(result.starts_with("<|im_start|>user\n"));
        assert!(result.contains("a sunset"));
        assert!(result.contains("<|im_start|>assistant\n"));
        assert!(result.contains("<think>\n\n</think>\n\n"));
        assert!(result.ends_with("<think>\n\n</think>\n\n"));
    }

    #[test]
    fn templates_differ_only_in_thinking_block() {
        let z = format_prompt_for_qwen3("test");
        let f = format_prompt_for_flux2("test");
        // Flux.2 template = Z-Image template + thinking block
        assert_eq!(f, format!("{z}<think>\n\n</think>\n\n"));
    }

    #[test]
    fn test_qwen3_template_empty_prompt() {
        let result = format_prompt_for_qwen3("");
        assert_eq!(
            result,
            "<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n"
        );
        // Flux.2 variant should also handle empty prompt
        let flux_result = format_prompt_for_flux2("");
        assert!(flux_result.contains("<|im_end|>"));
        assert!(flux_result.ends_with("<think>\n\n</think>\n\n"));
    }

    #[test]
    fn test_flux2_template_preserves_special_chars() {
        let prompt = "a <robot> in {brackets} & symbols <>";
        let result = format_prompt_for_flux2(prompt);
        // Special characters must pass through unescaped
        assert!(result.contains("<robot>"));
        assert!(result.contains("{brackets}"));
        assert!(result.contains("& symbols <>"));
        // The template markers must still be intact around the prompt
        assert!(result.starts_with("<|im_start|>user\n"));
        assert!(result.contains("<|im_end|>"));
    }

    #[test]
    fn test_templates_exact_structure() {
        let prompt = "hello";
        let qwen3 = format_prompt_for_qwen3(prompt);
        // Verify exact character-level structure
        assert_eq!(
            qwen3,
            "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"
        );

        let flux2 = format_prompt_for_flux2(prompt);
        assert_eq!(
            flux2,
            "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }
}
