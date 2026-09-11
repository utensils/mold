//! Qwen3 text encoder wrapper.
//!
//! Wraps either the native BF16 `Bf16Qwen3Encoder` (with multi-layer extraction)
//! or the quantized `GgufQwen3Encoder`, providing a unified load/encode/drop/reload
//! interface that mirrors `T5Encoder`.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokenizers::Tokenizer;

use super::park;
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
    ///
    /// `attention` names the real (non-pad) positions. `None` keeps the bare
    /// causal mask; `Some` additionally excludes the padded keys, which is
    /// what BFL passes the Qwen3 language model (`text_encoder.py:408-416`).
    pub fn forward_with_layers(
        &mut self,
        input_ids: &Tensor,
        layer_indices: &[usize],
        attention: Option<&[bool]>,
    ) -> Result<Tensor> {
        match self {
            Self::BF16(m) => m.forward_with_layers(input_ids, layer_indices, attention),
            Self::Quantized(m) => m.forward_with_layers(input_ids, layer_indices, attention),
        }
    }
}

/// Reusable Qwen3 text encoder wrapper.
///
/// Holds the model weights (optionally — `None` when dropped to free VRAM),
/// the tokenizer, and device placement info.
///
/// Supports park-on-CPU when `MOLD_KEEP_TE_RAM=1`: see [`Self::park_to_cpu`].
/// BF16 path uses the HashMap-of-CPU-tensors plumbing; GGUF path falls
/// through to drop/reload (QTensor storage is device-tied).
pub(crate) struct Qwen3Encoder {
    pub model: Option<Qwen3Model>,
    pub tokenizer: Arc<Tokenizer>,
    pub device: Device,
    pub on_gpu: bool,
    pub is_quantized: bool,
    /// Paths needed for reload.
    encoder_paths: Vec<PathBuf>,
    dtype: DType,
    /// BF16 architecture config (Qwen3-4B vs 8B). Used for BF16 reload.
    bf16_config: Qwen3BF16Config,
    /// BF16-only: parameters parked on CPU host RAM, ready for fast unpark.
    /// `None` when not parked or when running the GGUF path.
    parked_tensors: Option<HashMap<String, Tensor>>,
    /// GGUF-only: the quantized checkpoint parked on CPU host RAM.
    ///
    /// A separate slot rather than a shared one because the two are different
    /// kinds of tensor and rebuild through different constructors — but the
    /// PROPERTY is the same, and since #1044 gave Qwen-Image's Qwen2 encoder
    /// the same treatment there is no longer any reason for GGUF to be the
    /// carve-out that re-reads from disk.
    parked_gguf: Option<super::qwen3_gguf::GgufParked>,
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

/// The fixed width FLUX.2 [klein] conditioning is padded to.
///
/// BFL `flux2/src/flux2/text_encoder.py:28` is `MAX_LENGTH = 512`, and
/// `:397-403` tokenizes with `padding="max_length", truncation=True,
/// max_length=512`. diffusers repeats it as `max_sequence_length: int = 512`
/// with the same tokenizer call (`pipeline_flux2_klein.py:214,236-238`), and
/// ComfyUI's `Qwen3Tokenizer` reaches it from the other side with
/// `min_length=512` (`comfy/text_encoders/flux.py:143-151`), with
/// `model_base.py:1087-1096` padding any shorter embedding up to 512 before
/// the DiT sees it.
///
/// All 512 rows reach the transformer UNMASKED — BFL's `:418-419` returns the
/// whole stacked tensor and never trims by the attention mask. The mask is the
/// language model's business only; the DiT attends to the pad rows as ordinary
/// context, which is why they must be computed the way upstream computes them
/// rather than dropped.
pub(crate) const FLUX2_KLEIN_MAX_LENGTH: usize = 512;

/// Qwen3's pad token, used only when the tokenizer will not name its own.
///
/// ComfyUI hard-codes this exact id for both Klein tokenizers
/// (`comfy/text_encoders/flux.py:143,148`, `pad_token=151643` —
/// `<|endoftext|>`). [`resolve_pad_token_id`] prefers whatever the loaded
/// tokenizer declares and falls back here, and a unit test pins the two
/// together so a checkpoint that ships a different vocabulary is caught rather
/// than silently padded with someone else's token.
pub(crate) const QWEN3_PAD_TOKEN_ID: u32 = 151_643;

/// The token id to right-pad a Klein prompt with.
///
/// Asks the tokenizer first — its configured padding parameters, then the
/// `<|endoftext|>` special token — and only then falls back to
/// [`QWEN3_PAD_TOKEN_ID`].
fn resolve_pad_token_id(tokenizer: &Tokenizer) -> u32 {
    if let Some(padding) = tokenizer.get_padding() {
        return padding.pad_id;
    }
    tokenizer
        .token_to_id("<|endoftext|>")
        .unwrap_or(QWEN3_PAD_TOKEN_ID)
}

/// Truncate to `max_length` and right-pad with `pad_id`, reporting which
/// positions are real.
///
/// Right-padding is upstream's: `Qwen2TokenizerFast` pads on the right, and
/// ComfyUI's `SDTokenizer::pad_tokens` appends (`pad_left` is false for both
/// Klein tokenizers, `comfy/sd1_clip.py:565-570`). It is also what makes the
/// language model's real rows independent of the pads — a causal mask already
/// hides every later position from an earlier query — so the attention mask
/// exists to give the PAD rows upstream's values, not to protect the prompt.
fn pad_to_max_length(
    mut tokens: Vec<u32>,
    max_length: usize,
    pad_id: u32,
) -> (Vec<u32>, Vec<bool>) {
    tokens.truncate(max_length);
    let real = tokens.len();
    tokens.resize(max_length, pad_id);
    let attention = (0..max_length).map(|index| index < real).collect();
    (tokens, attention)
}

/// A combined causal + key-padding additive attention mask, `(1, 1, L, L)`.
///
/// `0` where a query may read a key, `-inf` otherwise: the key must be at or
/// before the query (causal) AND must be a real token (padding). This mirrors
/// what HuggingFace builds from `attention_mask` for the Qwen3 forward BFL
/// calls at `text_encoder.py:408-416`, and is the same construction
/// `encoders/mistral3.rs:483-499` already uses for FLUX.2 [dev]'s Mistral3
/// encoder — FLUX.2's two tiers now pad and mask identically.
pub(crate) fn causal_padding_mask(
    attention: &[bool],
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let len = attention.len();
    let values = (0..len)
        .flat_map(|query| {
            (0..len).map(move |key| {
                if key <= query && attention[key] {
                    0.0f32
                } else {
                    f32::NEG_INFINITY
                }
            })
        })
        .collect::<Vec<_>>();
    Tensor::from_vec(values, (1, 1, len, len), device)?
        .to_dtype(dtype)
        .map_err(Into::into)
}

impl Qwen3Encoder {
    /// Load a BF16 Qwen3 encoder from safetensors shards.
    ///
    /// The `bf16_config` selects the architecture variant: `Qwen3BF16Config::qwen3_4b()`
    /// for Klein-4B / Z-Image, or `Qwen3BF16Config::qwen3_8b()` for Klein-9B.
    #[allow(dead_code)]
    pub fn load_bf16(
        encoder_paths: &[PathBuf],
        tokenizer_path: &PathBuf,
        device: &Device,
        dtype: DType,
        bf16_config: &Qwen3BF16Config,
        progress: &crate::progress::ProgressReporter,
    ) -> Result<Self> {
        Self::load_bf16_with_tokenizer(
            encoder_paths,
            tokenizer_path,
            None,
            device,
            dtype,
            bf16_config,
            progress,
        )
    }

    /// Load a BF16 Qwen3 encoder using a preloaded tokenizer when available.
    #[allow(clippy::too_many_arguments)]
    pub fn load_bf16_with_tokenizer(
        encoder_paths: &[PathBuf],
        tokenizer_path: &PathBuf,
        tokenizer: Option<Arc<Tokenizer>>,
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

        let tokenizer = tokenizer.map(Ok).unwrap_or_else(|| {
            Tokenizer::from_file(tokenizer_path)
                .map(Arc::new)
                .map_err(|e| anyhow::anyhow!("failed to load Qwen3 tokenizer: {e}"))
        })?;
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
            parked_tensors: None,
            parked_gguf: None,
        })
    }

    /// Load a quantized Qwen3 encoder from a GGUF file.
    ///
    /// The `bf16_config` is stored for potential BF16 fallback reload but is not
    /// used during GGUF loading (GGUF reads dimensions from file metadata).
    #[allow(dead_code)]
    pub fn load_gguf(
        gguf_path: &Path,
        tokenizer_path: &PathBuf,
        device: &Device,
        bf16_config: &Qwen3BF16Config,
    ) -> Result<Self> {
        Self::load_gguf_with_tokenizer(gguf_path, tokenizer_path, None, device, bf16_config)
    }

    /// Load a quantized Qwen3 encoder using a preloaded tokenizer when available.
    pub fn load_gguf_with_tokenizer(
        gguf_path: &Path,
        tokenizer_path: &PathBuf,
        tokenizer: Option<Arc<Tokenizer>>,
        device: &Device,
        bf16_config: &Qwen3BF16Config,
    ) -> Result<Self> {
        let model = Qwen3Model::Quantized(GgufQwen3Encoder::load(gguf_path, device)?);
        let tokenizer = tokenizer.map(Ok).unwrap_or_else(|| {
            Tokenizer::from_file(tokenizer_path)
                .map(Arc::new)
                .map_err(|e| anyhow::anyhow!("failed to load Qwen3 tokenizer: {e}"))
        })?;
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
            parked_tensors: None,
            parked_gguf: None,
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
    ///
    /// `max_length` is the fixed conditioning width. `Some(n)` truncates to `n`
    /// tokens, right-pads to `n` with the tokenizer's pad token, masks the pad
    /// KEYS out of the language model's attention, and returns all `n` rows —
    /// BFL's `Qwen3Embedder::forward` (`flux2/text_encoder.py:397-419`) and
    /// diffusers' `_get_qwen3_prompt_embeds`
    /// (`pipeline_flux2_klein.py:236-256`) exactly. `None` keeps the prompt's
    /// natural length.
    ///
    /// The parameter exists because the fixed width is FLUX.2 [klein]'s, not
    /// Qwen3's. Z-Image runs the same encoder and does NOT hand padded rows to
    /// its transformer: ComfyUI's Z-Image tokenizer sets `min_length=1`
    /// (`comfy/text_encoders/z_image.py:8`) and diffusers, though it pads for
    /// the language model, trims straight back with
    /// `prompt_embeds[i][prompt_masks[i]]`
    /// (`pipelines/z_image/pipeline_z_image.py:229-250`). Z-Image reaches this
    /// encoder through [`Self::encode`], which is untouched.
    pub fn encode_with_layers(
        &mut self,
        prompt: &str,
        target_device: &Device,
        target_dtype: DType,
        layer_indices: &[usize],
        max_length: Option<usize>,
    ) -> Result<(Tensor, usize)> {
        let pad_id = resolve_pad_token_id(&self.tokenizer);
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

        let (tokens, attention) = match max_length {
            Some(max_length) => {
                let (tokens, attention) = pad_to_max_length(tokens, max_length, pad_id);
                (tokens, Some(attention))
            }
            None => (tokens, None),
        };

        let token_count = tokens.len();
        let input_ids = Tensor::from_vec(tokens, (1, token_count), &self.device)?;

        let emb = model.forward_with_layers(&input_ids, layer_indices, attention.as_deref())?;
        let emb = emb.to_device(target_device)?.to_dtype(target_dtype)?;
        Ok((emb, token_count))
    }

    /// Drop model weights to free memory (e.g. GPU VRAM after encoding).
    pub fn drop_weights(&mut self) {
        self.model = None;
        self.parked_tensors = None;
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

    /// Park encoder parameters into a CPU-resident HashMap of named tensors.
    ///
    /// The first call after a `reload()` reads the safetensors fresh from
    /// disk into CPU RAM (so the on-disk file is paged in once, not avoided);
    /// subsequent park/unpark cycles reuse the existing CPU tensors and
    /// avoid disk I/O. The GPU model is dropped after the CPU map is
    /// populated. Subsequent `unpark_to_gpu()` calls are CPU→GPU tensor
    /// copies (~100-300 ms typical).
    ///
    /// BF16 path: load all shards to CPU once, drop GPU. GGUF path: falls
    /// through to `drop_weights()` — `unpark_to_gpu()` will route to
    /// `reload()`.
    /// No-op when already parked.
    pub fn park_to_cpu(&mut self) -> Result<()> {
        if self.is_parked() {
            self.model = None;
            return Ok(());
        }
        if self.is_quantized {
            // GGUF parks too. `wan::block_offload::qtensor_to_device`
            // serializes a quantized tensor through its own bytes and rebuilds
            // it on the target, so the round trip is byte-exact — the
            // "device-tied QTensors cannot survive a CPU round-trip" this
            // replaces was a scoping decision from before #1044 proved
            // otherwise for Qwen-Image's Qwen2 encoder.
            let Some(Qwen3Model::Quantized(model)) = self.model.as_ref() else {
                self.drop_weights();
                return Ok(());
            };
            self.parked_gguf = Some(model.park_to_cpu()?);
            self.model = None;
            return Ok(());
        }
        let parked = park::load_tensors_to_cpu(&self.encoder_paths)?;
        self.parked_tensors = Some(parked);
        self.model = None;
        Ok(())
    }

    /// Restore parameters from CPU back to the encoder's primary device.
    /// Falls back to `reload()` for the GGUF path or when no parked map
    /// is present. No-op if the model is already loaded.
    pub fn unpark_to_gpu(&mut self, progress: &crate::progress::ProgressReporter) -> Result<()> {
        if self.model.is_some() {
            return Ok(());
        }
        if let Some(parked) = self.parked_gguf.as_ref() {
            self.model = Some(Qwen3Model::Quantized(GgufQwen3Encoder::from_parked(
                parked,
                &self.device,
            )?));
            return Ok(());
        }
        if let Some(parked) = self.parked_tensors.as_ref() {
            let vb = park::varbuilder_from_parked(parked, self.dtype, &self.device);
            self.model = Some(Qwen3Model::BF16(Bf16Qwen3Encoder::load(
                &self.bf16_config,
                vb,
            )?));
            return Ok(());
        }
        self.reload(progress)
    }

    /// The checkpoint files this encoder was actually built from.
    ///
    /// The residency decision needs THESE, not the manifest's BF16 shards: a
    /// host that resolved a Q8 GGUF variant would otherwise be asked whether
    /// it can park bytes ten times the size of what it is holding.
    pub fn encoder_paths(&self) -> &[PathBuf] {
        &self.encoder_paths
    }

    /// Host bytes this encoder's park is holding, or zero.
    ///
    /// Read back into the residency decision as `already_parked_bytes`:
    /// `MemAvailable` already excludes these, so a warm engine that did not
    /// credit them would be asked whether it could park a SECOND copy and
    /// would release the one it has.
    pub fn parked_bytes(&self) -> u64 {
        let dense: u64 = self
            .parked_tensors
            .iter()
            .flat_map(|map| map.values())
            .map(|tensor| (tensor.elem_count() * tensor.dtype().size_in_bytes()) as u64)
            .sum();
        let quantized: u64 = self
            .parked_gguf
            .iter()
            .flat_map(|(tensors, _)| tensors.values())
            .map(|tensor| tensor.storage_size_in_bytes() as u64)
            .sum();
        dense.saturating_add(quantized)
    }

    /// Whether this encoder is currently parked (CPU-resident, GPU-free), on
    /// either the BF16 or the GGUF path.
    pub fn is_parked(&self) -> bool {
        self.model.is_none() && (self.parked_tensors.is_some() || self.parked_gguf.is_some())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::IndexOp;

    /// FLUX.2 [klein] conditions on a FIXED 512 rows, whatever the prompt says.
    ///
    /// BFL `flux2/text_encoder.py:28` (`MAX_LENGTH = 512`) and `:397-403`
    /// (`padding="max_length", truncation=True`) are the contract; diffusers
    /// repeats it at `pipeline_flux2_klein.py:214,236-238` and ComfyUI reaches
    /// it from the other side with `min_length=512, pad_token=151643`
    /// (`comfy/text_encoders/flux.py:143-151`). Padding is on the RIGHT —
    /// ComfyUI's `pad_tokens` appends unless `pad_left` (`comfy/sd1_clip.py
    /// :565-570`), which neither Klein tokenizer sets.
    ///
    /// The mask is `encoders/mistral3.rs:483-499`'s shape, which FLUX.2 [dev]
    /// has used since it shipped: real keys attend causally, pad keys are
    /// excluded for every query.
    #[test]
    fn klein_prompt_is_padded_to_512_and_masked() {
        let (tokens, attention) = pad_to_max_length(vec![7, 8, 9], 512, QWEN3_PAD_TOKEN_ID);
        assert_eq!(tokens.len(), 512, "the conditioning width is fixed");
        assert_eq!(&tokens[..3], &[7, 8, 9]);
        assert!(
            tokens[3..].iter().all(|&id| id == QWEN3_PAD_TOKEN_ID),
            "pads go on the RIGHT, and they are the tokenizer's pad token"
        );
        assert_eq!(attention.len(), 512);
        assert_eq!(attention.iter().filter(|real| **real).count(), 3);
        assert!(attention[..3].iter().all(|real| *real));
        assert!(attention[3..].iter().all(|real| !*real));

        // A prompt past the budget is TRUNCATED, never grown.
        let (long, long_attention) = pad_to_max_length(vec![1u32; 900], 512, QWEN3_PAD_TOKEN_ID);
        assert_eq!(long.len(), 512);
        assert!(long_attention.iter().all(|real| *real));

        // The mask, read on a short length so the table is legible. Positions
        // 0-2 are real, 3-5 are pads.
        let attention = [true, true, true, false, false, false];
        let mask = causal_padding_mask(&attention, DType::F32, &Device::Cpu).unwrap();
        assert_eq!(mask.dims(), &[1, 1, 6, 6]);
        let rows = mask.i((0, 0)).unwrap().to_vec2::<f32>().unwrap();
        // A real query reads earlier real keys and nothing later.
        assert_eq!(rows[1][0], 0.0);
        assert_eq!(rows[1][1], 0.0);
        assert_eq!(rows[1][2], f32::NEG_INFINITY);
        // A pad query reads the real prefix but never another pad.
        assert_eq!(rows[4][0], 0.0);
        assert_eq!(rows[4][2], 0.0);
        assert_eq!(
            rows[4][3],
            f32::NEG_INFINITY,
            "a pad key must be excluded even from a later query"
        );
        assert_eq!(rows[4][4], f32::NEG_INFINITY, "including itself");
        assert_eq!(rows[4][5], f32::NEG_INFINITY);
        // And no real query anywhere can see a pad key.
        for (query, row) in rows.iter().enumerate().take(3) {
            for (key, value) in row.iter().enumerate().skip(3) {
                assert_eq!(
                    *value,
                    f32::NEG_INFINITY,
                    "real query {query} must not read pad key {key}"
                );
            }
        }
    }

    /// ComfyUI hard-codes `pad_token=151643` for both Klein tokenizers
    /// (`comfy/text_encoders/flux.py:143,148`). [`resolve_pad_token_id`] asks
    /// the loaded tokenizer first, so this pins the FALLBACK it drops back to
    /// when the tokenizer declares nothing.
    #[test]
    fn the_fallback_pad_token_is_comfyuis_hardcoded_id() {
        assert_eq!(QWEN3_PAD_TOKEN_ID, 151_643);
        assert_eq!(FLUX2_KLEIN_MAX_LENGTH, 512);
    }

    /// Appending masked pads must not move the real rows.
    ///
    /// There is no tiny Qwen3 fixture to run — the architecture constants are
    /// the checkpoint's (36 layers, a 151,936-row embedding table), so a
    /// "small" one is still gigabytes. What the padding change owns is the
    /// MASK, and this runs the one operation the mask feeds: a single-head
    /// scaled-dot-product attention over random keys and values, once over a
    /// 3-token prompt with a plain causal mask and once over the same prompt
    /// padded to 8 with [`causal_padding_mask`]. The first three output rows
    /// must be bit-identical — which is the whole reason the pads are masked
    /// rather than merely appended.
    #[test]
    fn padded_klein_encode_matches_unpadded_on_real_rows() {
        let device = Device::Cpu;
        let real = 3usize;
        let padded = 8usize;
        let width = 4usize;

        // One sequence of `padded` rows; the first `real` are the prompt and
        // the rest are what the pad token would have embedded to.
        let q = Tensor::randn(0f32, 1., (padded, width), &device).unwrap();
        let k = Tensor::randn(0f32, 1., (padded, width), &device).unwrap();
        let v = Tensor::randn(0f32, 1., (padded, width), &device).unwrap();

        let attend = |q: &Tensor, k: &Tensor, v: &Tensor, mask: &Tensor| -> Vec<Vec<f32>> {
            let scores = q.matmul(&k.t().unwrap()).unwrap();
            let scores = (scores + mask).unwrap();
            let weights = candle_nn::ops::softmax_last_dim(&scores).unwrap();
            weights.matmul(v).unwrap().to_vec2::<f32>().unwrap()
        };

        let unpadded_mask = causal_padding_mask(&vec![true; real], DType::F32, &device)
            .unwrap()
            .i((0, 0))
            .unwrap();
        let unpadded = attend(
            &q.narrow(0, 0, real).unwrap(),
            &k.narrow(0, 0, real).unwrap(),
            &v.narrow(0, 0, real).unwrap(),
            &unpadded_mask,
        );

        let attention = (0..padded).map(|index| index < real).collect::<Vec<_>>();
        let padded_mask = causal_padding_mask(&attention, DType::F32, &device)
            .unwrap()
            .i((0, 0))
            .unwrap();
        let with_pads = attend(&q, &k, &v, &padded_mask);

        assert_eq!(
            unpadded,
            with_pads[..real].to_vec(),
            "masked pads must leave the prompt's hidden states untouched"
        );
    }

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
