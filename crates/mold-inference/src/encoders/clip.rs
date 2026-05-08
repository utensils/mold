use anyhow::Result;
use candle_core::{DType, Device, Module, Tensor};
use candle_transformers::models::clip;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokenizers::Tokenizer;

use super::park;

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
///
/// Supports park-on-CPU when `MOLD_KEEP_TE_RAM=1`: see [`Self::park_to_cpu`].
pub(crate) struct ClipEncoder {
    pub model: Option<clip::text_model::ClipTextTransformer>,
    pub tokenizer: Arc<Tokenizer>,
    pub device: Device,
    pub on_gpu: bool,
    /// Encoder weights path — needed to populate `parked_tensors` on first
    /// park and to drive the `reload()` fallback.
    encoder_path: PathBuf,
    /// Parameters parked on CPU host RAM, ready for fast unpark.
    parked_tensors: Option<HashMap<String, Tensor>>,
}

impl ClipEncoder {
    /// Load CLIP encoder weights and tokenizer.
    #[allow(dead_code)]
    pub fn load(
        encoder_path: &PathBuf,
        tokenizer_path: &PathBuf,
        device: &Device,
        dtype: DType,
        progress: &crate::progress::ProgressReporter,
    ) -> Result<Self> {
        Self::load_with_tokenizer(encoder_path, tokenizer_path, device, dtype, progress, None)
    }

    /// Load CLIP encoder weights, reusing a cached tokenizer if provided.
    pub fn load_with_tokenizer(
        encoder_path: &PathBuf,
        tokenizer_path: &PathBuf,
        device: &Device,
        dtype: DType,
        progress: &crate::progress::ProgressReporter,
        cached_tokenizer: Option<Arc<Tokenizer>>,
    ) -> Result<Self> {
        let vb = crate::weight_loader::load_safetensors_with_progress(
            std::slice::from_ref(encoder_path),
            dtype,
            device,
            "CLIP-L",
            progress,
        )?;
        let model = clip::text_model::ClipTextTransformer::new(vb.pp("text_model"), &config())?;
        let tokenizer = match cached_tokenizer {
            Some(tok) => tok,
            None => Arc::new(
                Tokenizer::from_file(tokenizer_path)
                    .map_err(|e| anyhow::anyhow!("failed to load CLIP tokenizer: {e}"))?,
            ),
        };
        let on_gpu = crate::device::is_gpu(device);

        Ok(Self {
            model: Some(model),
            tokenizer,
            device: device.clone(),
            on_gpu,
            encoder_path: encoder_path.clone(),
            parked_tensors: None,
        })
    }

    /// Get a reference-counted handle to this encoder's tokenizer (for caching in SharedPool).
    pub fn tokenizer_arc(&self) -> Arc<Tokenizer> {
        self.tokenizer.clone()
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
        self.parked_tensors = None;
    }

    /// Reload model weights (e.g. for the next generation after being dropped).
    pub fn reload(
        &mut self,
        encoder_path: &PathBuf,
        dtype: DType,
        progress: &crate::progress::ProgressReporter,
    ) -> Result<()> {
        let vb = crate::weight_loader::load_safetensors_with_progress(
            std::slice::from_ref(encoder_path),
            dtype,
            &self.device,
            "CLIP-L",
            progress,
        )?;
        self.model = Some(clip::text_model::ClipTextTransformer::new(
            vb.pp("text_model"),
            &config(),
        )?);
        Ok(())
    }

    /// Park encoder parameters into a CPU-resident HashMap of named tensors.
    ///
    /// The first call after a `reload()` reads the safetensors fresh from
    /// disk into CPU RAM (so the on-disk file is paged in once, not avoided);
    /// subsequent park/unpark cycles reuse the existing CPU tensors and
    /// avoid disk I/O. The GPU model is dropped after the CPU map is
    /// populated. Subsequent `unpark_to_gpu()` calls are CPU→GPU tensor
    /// copies (~100-300 ms typical). CLIP-L is small (~246 MB) so the
    /// CPU footprint is negligible compared to T5/Qwen3.
    ///
    /// No-op when already parked.
    pub fn park_to_cpu(&mut self) -> Result<()> {
        if self.is_parked() {
            self.model = None;
            return Ok(());
        }
        let parked = park::load_tensors_to_cpu(std::slice::from_ref(&self.encoder_path))?;
        self.parked_tensors = Some(parked);
        self.model = None;
        Ok(())
    }

    /// Restore parameters from CPU back to the encoder's primary device.
    ///
    /// No-op when the model is already loaded.
    pub fn unpark_to_gpu(
        &mut self,
        dtype: DType,
        progress: &crate::progress::ProgressReporter,
    ) -> Result<()> {
        if self.model.is_some() {
            return Ok(());
        }
        if let Some(parked) = self.parked_tensors.as_ref() {
            let vb = park::varbuilder_from_parked(parked, dtype, &self.device);
            self.model = Some(clip::text_model::ClipTextTransformer::new(
                vb.pp("text_model"),
                &config(),
            )?);
            return Ok(());
        }
        let path = self.encoder_path.clone();
        self.reload(&path, dtype, progress)
    }

    /// Whether this encoder is currently parked (CPU-resident, GPU-free).
    pub fn is_parked(&self) -> bool {
        self.model.is_none() && self.parked_tensors.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a CPU-only CLIP encoder skeleton for state-machine tests.
    /// `model` is `None` because constructing a real `ClipTextTransformer`
    /// requires the full HuggingFace clip-vit-large-patch14 weight tree.
    fn make_test_encoder() -> ClipEncoder {
        let dummy_path = std::env::temp_dir().join("nonexistent-clip-tokenizer.json");
        let tokenizer = Arc::new(tokenizers::Tokenizer::new(
            tokenizers::models::wordpiece::WordPiece::default(),
        ));
        ClipEncoder {
            model: None,
            tokenizer,
            device: Device::Cpu,
            on_gpu: false,
            encoder_path: dummy_path,
            parked_tensors: None,
        }
    }

    #[test]
    fn test_is_parked_state_machine() {
        let mut e = make_test_encoder();
        assert!(!e.is_parked());

        // Park-state simulation
        e.parked_tensors = Some(HashMap::new());
        assert!(e.is_parked());

        // Drop should clear both
        e.drop_weights();
        assert!(!e.is_parked());
        assert!(e.parked_tensors.is_none());
    }

    #[test]
    fn test_park_when_already_parked_is_noop() {
        let mut e = make_test_encoder();
        let mut map = HashMap::new();
        map.insert(
            "canary".to_string(),
            Tensor::zeros((1,), DType::F32, &Device::Cpu).unwrap(),
        );
        e.parked_tensors = Some(map);
        e.model = None;
        assert!(e.is_parked());

        // Re-park is noop on the parked map (no disk read)
        e.park_to_cpu().expect("re-park is noop");
        assert!(e.is_parked());
        assert!(
            e.parked_tensors.as_ref().unwrap().contains_key("canary"),
            "re-park preserved the existing parked map"
        );
    }
}
