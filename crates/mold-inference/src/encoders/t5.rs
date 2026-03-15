use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::t5;
use std::path::PathBuf;
use tokenizers::Tokenizer;

/// T5-XXL config (hardcoded — this model variant is fixed for FLUX).
pub fn config() -> t5::Config {
    t5::Config {
        vocab_size: 32128,
        d_model: 4096,
        d_kv: 64,
        d_ff: 10240,
        num_heads: 64,
        num_layers: 24,
        relative_attention_num_buckets: 32,
        relative_attention_max_distance: 128,
        dropout_rate: 0.1,
        layer_norm_epsilon: 1e-6,
        initializer_factor: 1.0,
        feed_forward_proj: t5::ActivationWithOptionalGating {
            gated: true,
            activation: candle_nn::Activation::NewGelu,
        },
        tie_word_embeddings: false,
        use_cache: true,
        pad_token_id: 0,
        eos_token_id: 1,
        decoder_start_token_id: Some(0),
        is_decoder: false,
        is_encoder_decoder: true,
        num_decoder_layers: Some(24),
    }
}

/// Reusable T5 text encoder wrapper.
///
/// Holds the model weights (optionally — `None` when dropped to free VRAM),
/// the tokenizer, and device placement info.
pub(crate) struct T5Encoder {
    pub model: Option<t5::T5EncoderModel>,
    pub tokenizer: Tokenizer,
    pub device: Device,
    pub on_gpu: bool,
}

impl T5Encoder {
    /// Load T5 encoder weights and tokenizer.
    pub fn load(
        encoder_path: &PathBuf,
        tokenizer_path: &PathBuf,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(std::slice::from_ref(encoder_path), dtype, device)?
        };
        let model = t5::T5EncoderModel::load(vb, &config())?;
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("failed to load T5 tokenizer: {e}"))?;
        let on_gpu = device.is_cuda() || device.is_metal();

        Ok(Self {
            model: Some(model),
            tokenizer,
            device: device.clone(),
            on_gpu,
        })
    }

    /// Encode a text prompt into T5 embeddings, padded to 256 tokens.
    /// The output tensor is moved to `target_device` with `target_dtype`.
    pub fn encode(
        &mut self,
        prompt: &str,
        target_device: &Device,
        target_dtype: DType,
    ) -> Result<Tensor> {
        let t5 = self
            .model
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("T5 model unavailable"))?;

        let mut tokens = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow::anyhow!("T5 tokenization failed: {e}"))?
            .get_ids()
            .to_vec();
        tokens.resize(256, 0);

        let input_ids = Tensor::new(&tokens[..], &self.device)?.unsqueeze(0)?;
        let emb = t5.forward(&input_ids)?;
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
        self.model = Some(t5::T5EncoderModel::load(vb, &config())?);
        Ok(())
    }
}
