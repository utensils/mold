use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_transformers::models::t5;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokenizers::Tokenizer;

use super::park;
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
///
/// Two retention modes between requests:
/// - `drop_weights()` / `reload()` — frees host RAM too; reload reads from
///   disk again. Default for backward compatibility.
/// - `park_to_cpu()` / `unpark_to_gpu()` — keeps weights resident in host
///   RAM (~9 GB for T5-XXL fp16) so the next request only pays a CPU→GPU
///   copy. Opt-in via `MOLD_KEEP_TE_RAM=1`. FP16-only — GGUF falls through
///   to drop/reload because `QTensor` storage is device-tied.
pub(crate) struct T5Encoder {
    pub model: Option<T5Model>,
    pub tokenizer: Arc<Tokenizer>,
    pub device: Device,
    pub on_gpu: bool,
    /// Whether this encoder uses a quantized GGUF model.
    pub is_quantized: bool,
    /// Path to the encoder weights — needed for park (FP16: re-read into CPU
    /// HashMap; GGUF: handed to `reload()` on unpark fallback).
    encoder_path: PathBuf,
    /// FP16-only: parameters parked on CPU, ready for fast unpark. `None`
    /// when not parked. Skipped entirely for the GGUF path.
    parked_tensors: Option<HashMap<String, Tensor>>,
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
            encoder_path: encoder_path.clone(),
            parked_tensors: None,
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
        // Don't keep parked tensors when explicitly dropping — caller wants
        // RAM freed, not retained.
        self.parked_tensors = None;
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

    /// Move parameters from GPU to CPU host RAM, keeping them ready for a
    /// fast unpark on the next request.
    ///
    /// FP16 path: lazily populates `parked_tensors` on the first call (one-
    /// time disk read into CPU), then drops the GPU model. Subsequent
    /// `park_to_cpu()` after an unpark is a no-op apart from dropping the
    /// GPU copy — the CPU HashMap is already populated.
    ///
    /// GGUF path: falls back to `drop_weights()`. The next `unpark_to_gpu()`
    /// detects the missing parked map and routes to `reload()`. This keeps
    /// the API uniform without dragging quantized tensor walking into the
    /// park module.
    ///
    /// No-op if already parked.
    pub fn park_to_cpu(&mut self) -> Result<()> {
        if self.is_parked() {
            // Already parked — nothing to do, but keep the GPU copy gone if
            // somehow it lingered. (Defensive — the state machine shouldn't
            // get here with both populated.)
            self.model = None;
            return Ok(());
        }

        if self.is_quantized {
            // GGUF: drop and let unpark route to reload. Leaves
            // `parked_tensors = None` — that's the signal for the unpark
            // path.
            self.drop_weights();
            return Ok(());
        }

        // FP16 path: pre-load all tensors fresh to CPU (mmap → CPU buffer
        // copy). This is a one-time cost that pays for itself on the next
        // unpark — subsequent park/unpark cycles never touch the disk.
        let parked = park::load_tensors_to_cpu(std::slice::from_ref(&self.encoder_path))?;
        self.parked_tensors = Some(parked);
        // Drop the GPU model — its VRAM is what we wanted to free.
        self.model = None;
        Ok(())
    }

    /// Restore parameters from the CPU-resident HashMap back to the
    /// encoder's primary device.
    ///
    /// FP16 + parked: rebuilds the model via `VarBuilder::from_tensors`.
    /// The H2D copy happens inside the backend's `get()` calls during
    /// model construction. The parked HashMap is retained — a subsequent
    /// `park_to_cpu()` becomes a near-instant "drop the GPU copy."
    ///
    /// GGUF or no parked map: routes to `reload()` (worst-case fallback).
    ///
    /// No-op if the model is already loaded (i.e. not currently parked).
    pub fn unpark_to_gpu(
        &mut self,
        dtype: DType,
        progress: &crate::progress::ProgressReporter,
    ) -> Result<()> {
        if self.model.is_some() {
            // Already on-device.
            return Ok(());
        }

        if let Some(parked) = self.parked_tensors.as_ref() {
            // Fast path: rebuild from CPU tensors → target device.
            let vb = park::varbuilder_from_parked(parked, dtype, &self.device);
            self.model = Some(T5Model::FP16(t5::T5EncoderModel::load(vb, &config())?));
            return Ok(());
        }

        // No parked map (GGUF path or never parked) → full reload from disk.
        let path = self.encoder_path.clone();
        self.reload(&path, dtype, progress)
    }

    /// Whether this encoder is currently parked (CPU-resident, GPU-free).
    ///
    /// True when:
    /// - `model` is `None` (no GPU copy), AND
    /// - `parked_tensors` is `Some` (FP16 path with CPU copy ready).
    ///
    /// Quantized encoders never report `is_parked() == true` — they bounce
    /// through `drop_weights()` on park and `reload()` on unpark.
    pub fn is_parked(&self) -> bool {
        self.model.is_none() && self.parked_tensors.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::{serialize_to_file, Dtype as SafeDtype, TensorView};
    use std::collections::HashMap as StdHashMap;

    /// Minimal CPU-only `T5Encoder` for state-machine tests.
    ///
    /// We can't easily build a real T5 model without the full weights tree
    /// (T5-XXL has 24 layers across thousands of tensors), so we construct
    /// the wrapper directly with a `model: None` and a hand-rolled
    /// `parked_tensors` map. This still exercises:
    /// - `is_parked()` transitions across drop/park/unpark calls
    /// - `park_to_cpu()` no-op semantics when already parked
    /// - `drop_weights()` clearing the parked map (caller wants RAM freed)
    /// - The GGUF fallback path (`is_quantized=true` → no parked map)
    fn make_test_encoder(is_quantized: bool) -> T5Encoder {
        // Dummy tokenizer — never invoked in these tests
        let dummy_tokenizer_path = std::env::temp_dir().join("nonexistent-t5-tokenizer.json");
        // We can't build a real Tokenizer without a real file; cheat by
        // wrapping a default-constructed empty one behind Arc. The tests
        // never call `encode()` so the tokenizer is never exercised.
        let tokenizer = Arc::new(tokenizers::Tokenizer::new(
            tokenizers::models::wordpiece::WordPiece::default(),
        ));
        T5Encoder {
            model: None,
            tokenizer,
            device: Device::Cpu,
            on_gpu: false,
            is_quantized,
            encoder_path: dummy_tokenizer_path,
            parked_tensors: None,
        }
    }

    /// `is_parked()` is the disjunction `model.is_none() &&
    /// parked_tensors.is_some()`. Walk the state machine: empty → parked
    /// → loaded → parked → dropped.
    #[test]
    fn test_is_parked_state_machine() {
        let mut e = make_test_encoder(false);
        // Initial: not loaded, not parked → not parked.
        assert!(!e.is_parked(), "fresh encoder with no map is not parked");

        // Pretend we parked: parked_tensors = Some, model = None
        e.parked_tensors = Some(HashMap::new());
        assert!(e.is_parked(), "model=None + parked=Some → parked");

        // Pretend we unparked: parked_tensors keeps its value, model = Some
        // (we can't construct a real T5Model in tests, so use the
        // is_some() check directly).
        // Skipping the model = Some path here because constructing T5Model
        // requires real weights — but the `is_parked()` definition
        // guarantees it returns false when model is Some.

        // Now drop: should clear parked_tensors too
        e.drop_weights();
        assert!(!e.is_parked(), "drop_weights clears parked map");
        assert!(e.parked_tensors.is_none(), "parked map cleared on drop");
    }

    /// Quantized (GGUF) encoders never report parked: park falls through
    /// to drop_weights and unpark falls through to reload.
    #[test]
    fn test_park_when_quantized_falls_through_to_drop() {
        let mut e = make_test_encoder(true);
        // Pre-condition: a "loaded" GGUF encoder would have model = Some,
        // but for this state-only test we just confirm the park path
        // doesn't try to populate parked_tensors when is_quantized=true.
        e.parked_tensors = None;
        // park_to_cpu would normally need a real path — but for the
        // is_quantized branch, we hit drop_weights() before any disk I/O.
        e.park_to_cpu()
            .expect("quantized park is just drop_weights");
        assert!(!e.is_parked(), "quantized encoder is never parked");
        assert!(e.parked_tensors.is_none(), "quantized never holds CPU map");
    }

    /// `park_to_cpu()` when already parked is a no-op apart from making
    /// sure the GPU model copy is gone.
    #[test]
    fn test_park_when_already_parked_is_noop() {
        let mut e = make_test_encoder(false);

        // Manually set parked state
        let mut map = HashMap::new();
        map.insert(
            "marker".to_string(),
            Tensor::zeros((1,), DType::F32, &Device::Cpu).unwrap(),
        );
        e.parked_tensors = Some(map);
        e.model = None;
        assert!(e.is_parked());

        // Re-parking should not blow away the existing map (no second disk
        // read). The marker tensor is the canary.
        e.park_to_cpu().expect("re-park is noop");
        assert!(e.is_parked());
        assert!(
            e.parked_tensors.as_ref().unwrap().contains_key("marker"),
            "re-park preserved the existing parked map (no redundant disk read)"
        );
    }

    /// End-to-end park→unpark round-trip on a tiny standalone safetensors
    /// file that mirrors the tensor-extraction-and-rebuild flow real
    /// encoders use, without standing up a full T5. This is the closest we
    /// can get to integration testing in unit-test scope.
    #[test]
    fn test_park_unpark_roundtrip_via_helpers() {
        use candle_nn::{Linear, Module, VarBuilder};

        // Build a synthetic safetensors file (matches T5Encoder's loading
        // contract: `VarBuilder::from_mmaped_safetensors`).
        let path = std::env::temp_dir().join(format!(
            "mold-t5-roundtrip-{}-{}.safetensors",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let w_bytes: Vec<u8> = [0.1f32, 0.2, 0.3, 0.4]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let mut tensors: StdHashMap<String, TensorView> = StdHashMap::new();
        tensors.insert(
            "shared.weight".to_string(),
            TensorView::new(SafeDtype::F32, vec![2, 2], &w_bytes).unwrap(),
        );
        serialize_to_file(&tensors, &None, &path).unwrap();

        // First: load original VarBuilder → Linear, capture an output.
        let vb_orig = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&path], DType::F32, &Device::Cpu).unwrap()
        };
        let lin_orig = Linear::new(vb_orig.get((2, 2), "shared.weight").unwrap(), None);
        let x = Tensor::from_slice(&[1.0f32, 2.0], (1, 2), &Device::Cpu).unwrap();
        let y_orig: Vec<f32> = lin_orig
            .forward(&x)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        // Park: load to CPU map (this is what T5Encoder::park_to_cpu does)
        let parked = park::load_tensors_to_cpu(std::slice::from_ref(&path)).unwrap();

        // Unpark: rebuild VarBuilder from parked map → Linear
        let vb_new = park::varbuilder_from_parked(&parked, DType::F32, &Device::Cpu);
        let lin_new = Linear::new(vb_new.get((2, 2), "shared.weight").unwrap(), None);
        let y_new: Vec<f32> = lin_new
            .forward(&x)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        // Bit-identical: same weights, same dtype, same device.
        assert_eq!(
            y_orig, y_new,
            "park→unpark must reproduce the original output exactly"
        );

        // Park again from the same map (re-park = noop): map is still usable
        let parked_again = parked.clone();
        let vb_third = park::varbuilder_from_parked(&parked_again, DType::F32, &Device::Cpu);
        let _ = vb_third.get((2, 2), "shared.weight").unwrap();

        let _ = std::fs::remove_file(&path);
    }
}
