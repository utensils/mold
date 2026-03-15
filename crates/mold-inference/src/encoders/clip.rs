use anyhow::Result;
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::clip;
use std::path::PathBuf;
use tokenizers::Tokenizer;

/// CLIP-L text config (hardcoded — this model variant is fixed for FLUX).
/// SDXL would use a different config for CLIP-G.
pub fn config() -> clip::text_model::ClipTextConfig {
    clip::text_model::ClipTextConfig {
        vocab_size: 49408,
        projection_dim: 768,
        activation: clip::text_model::Activation::QuickGelu,
        intermediate_size: 3072,
        embed_dim: 768,
        max_position_embeddings: 77,
        pad_with: None,
        num_hidden_layers: 12,
        num_attention_heads: 12,
    }
}

/// Reusable CLIP text encoder wrapper.
///
/// Holds the model weights (optionally — `None` when dropped to free VRAM),
/// the tokenizer, and device placement info.
pub(crate) struct ClipEncoder {
    pub model: Option<clip::text_model::ClipTextTransformer>,
    pub tokenizer: Tokenizer,
    pub device: Device,
    pub on_gpu: bool,
}

impl ClipEncoder {
    /// Load CLIP encoder weights and tokenizer.
    pub fn load(
        encoder_path: &PathBuf,
        tokenizer_path: &PathBuf,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(std::slice::from_ref(encoder_path), dtype, device)?
        };
        let model = clip::text_model::ClipTextTransformer::new(vb.pp("text_model"), &config())?;
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("failed to load CLIP tokenizer: {e}"))?;
        let on_gpu = device.is_cuda() || device.is_metal();

        Ok(Self {
            model: Some(model),
            tokenizer,
            device: device.clone(),
            on_gpu,
        })
    }

    /// Encode a text prompt into CLIP embeddings (truncated to 77 tokens).
    /// The output tensor is moved to `target_device` with `target_dtype`.
    pub fn encode(
        &mut self,
        prompt: &str,
        target_device: &Device,
        target_dtype: DType,
    ) -> Result<Tensor> {
        let clip = self
            .model
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("CLIP model unavailable"))?;

        let mut tokens = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow::anyhow!("CLIP tokenization failed: {e}"))?
            .get_ids()
            .to_vec();
        // CLIP hard limit: 77 tokens (including BOS/EOS)
        tokens.truncate(77);

        let input_ids = Tensor::new(&tokens[..], &self.device)?.unsqueeze(0)?;
        let emb = clip.forward(&input_ids)?;
        // Ensure on target device with correct dtype
        Ok(emb.to_device(target_device)?.to_dtype(target_dtype)?)
    }

    /// Drop model weights to free memory (e.g. GPU VRAM after encoding).
    pub fn drop_weights(&mut self) {
        self.model = None;
    }

    /// Reload model weights (e.g. for the next generation after being dropped).
    pub fn reload(&mut self, encoder_path: &PathBuf, dtype: DType) -> Result<()> {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                std::slice::from_ref(encoder_path),
                dtype,
                &self.device,
            )?
        };
        self.model = Some(clip::text_model::ClipTextTransformer::new(
            vb.pp("text_model"),
            &config(),
        )?);
        Ok(())
    }
}
