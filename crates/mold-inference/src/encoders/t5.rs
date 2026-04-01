use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_transformers::models::t5;
use std::path::PathBuf;
use std::sync::Arc;
use tokenizers::Tokenizer;

use super::t5_gguf::GgufT5Encoder;

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

/// FP16 (safetensors) or quantized (GGUF) T5 encoder.
pub(crate) enum T5Model {
    FP16(t5::T5EncoderModel),
    Quantized(GgufT5Encoder),
}

impl T5Model {
    pub fn forward(&mut self, input_ids: &Tensor) -> Result<Tensor> {
        match self {
            Self::FP16(m) => Ok(m.forward(input_ids)?),
            Self::Quantized(m) => m.forward(input_ids),
        }
    }
}

/// Reusable T5 text encoder wrapper.
///
/// Holds the model weights (optionally — `None` when dropped to free VRAM),
/// the tokenizer, and device placement info. Supports both FP16 safetensors
/// and GGUF quantized T5 models.
pub(crate) struct T5Encoder {
    pub model: Option<T5Model>,
    pub tokenizer: Arc<Tokenizer>,
    pub device: Device,
    pub on_gpu: bool,
    /// Whether this encoder uses a quantized GGUF model.
    pub is_quantized: bool,
}

impl T5Encoder {
    /// Load T5 encoder weights and tokenizer.
    /// Auto-detects `.gguf` extension to choose quantized vs FP16 loading.
    pub fn load(
        encoder_path: &PathBuf,
        tokenizer_path: &PathBuf,
        device: &Device,
        dtype: DType,
        progress: &crate::progress::ProgressReporter,
    ) -> Result<Self> {
        Self::load_with_tokenizer(encoder_path, tokenizer_path, device, dtype, progress, None)
    }

    /// Load T5 encoder weights, reusing a cached tokenizer if provided.
    pub fn load_with_tokenizer(
        encoder_path: &PathBuf,
        tokenizer_path: &PathBuf,
        device: &Device,
        dtype: DType,
        progress: &crate::progress::ProgressReporter,
        cached_tokenizer: Option<Arc<Tokenizer>>,
    ) -> Result<Self> {
        let is_quantized = encoder_path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.eq_ignore_ascii_case("gguf"))
            .unwrap_or(false);

        let model = if is_quantized {
            T5Model::Quantized(GgufT5Encoder::load(encoder_path, device)?)
        } else {
            let vb = crate::weight_loader::load_safetensors_with_progress(
                std::slice::from_ref(encoder_path),
                dtype,
                device,
                "T5 encoder",
                progress,
            )?;
            T5Model::FP16(t5::T5EncoderModel::load(vb, &config())?)
        };

        let tokenizer = match cached_tokenizer {
            Some(tok) => tok,
            None => Arc::new(
                Tokenizer::from_file(tokenizer_path)
                    .map_err(|e| anyhow::anyhow!("failed to load T5 tokenizer: {e}"))?,
            ),
        };
        let on_gpu = crate::device::is_gpu(device);

        Ok(Self {
            model: Some(model),
            tokenizer,
            device: device.clone(),
            on_gpu,
            is_quantized,
        })
    }

    /// Get a reference-counted handle to this encoder's tokenizer (for caching in SharedPool).
    pub fn tokenizer_arc(&self) -> Arc<Tokenizer> {
        self.tokenizer.clone()
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
    pub fn reload(
        &mut self,
        encoder_path: &PathBuf,
        dtype: DType,
        progress: &crate::progress::ProgressReporter,
    ) -> Result<()> {
        if self.is_quantized {
            self.model = Some(T5Model::Quantized(GgufT5Encoder::load(
                encoder_path,
                &self.device,
            )?));
        } else {
            let vb = crate::weight_loader::load_safetensors_with_progress(
                std::slice::from_ref(encoder_path),
                dtype,
                &self.device,
                "T5 encoder",
                progress,
            )?;
            self.model = Some(T5Model::FP16(t5::T5EncoderModel::load(vb, &config())?));
        }
        Ok(())
    }
}
