use anyhow::Result;
use candle_core::{DType, Device, Module, Tensor};
use candle_transformers::models::clip;
use std::path::PathBuf;
use std::sync::Arc;
use tokenizers::Tokenizer;

/// Total length of a single CLIP token window: BOS + 75 content + EOS = 77.
pub(crate) const CLIP_CHUNK_LEN: usize = 77;

/// Maximum content tokens per chunk (excludes BOS/EOS).
pub(crate) const CLIP_MAX_TOKENS_PER_CHUNK: usize = CLIP_CHUNK_LEN - 2;

/// CLIP convention: pad with EOS (end-of-text) — pad_id == eos_id.
/// `<|startoftext|>` (BOS) is id 49406, `<|endoftext|>` (EOS) is id 49407.
pub(crate) const CLIP_BOS_TOKEN: &str = "<|startoftext|>";
pub(crate) const CLIP_EOS_TOKEN: &str = "<|endoftext|>";

/// Returns `true` when the user opts into ComfyUI-style chunked encoding for
/// long prompts (`MOLD_LONG_PROMPTS=1`). Default off — preserves the current
/// hard-truncate-at-77 behavior.
///
/// When enabled, prompts longer than 75 content tokens are split into 75-token
/// windows; each window is encoded independently with its own `[BOS, ..., EOS]`
/// framing, and the per-window pooled outputs are averaged into the single
/// 768-dim conditioning vector that FLUX's `vector_in` stem expects.
pub(crate) fn long_prompts_enabled() -> bool {
    std::env::var("MOLD_LONG_PROMPTS")
        .map(|v| v == "1")
        .unwrap_or(false)
}

/// Resolve the BOS/EOS token ids from the CLIP tokenizer's vocab. Falls back
/// to the documented constants (49406/49407) if the special tokens are absent
/// — both real CLIP variants ship with these strings, so the fallback only
/// triggers in malformed test fixtures.
fn clip_special_ids(tokenizer: &Tokenizer) -> (u32, u32) {
    let bos = tokenizer.token_to_id(CLIP_BOS_TOKEN).unwrap_or(49406);
    let eos = tokenizer.token_to_id(CLIP_EOS_TOKEN).unwrap_or(49407);
    (bos, eos)
}

/// Strip a leading BOS or trailing EOS that the tokenizer added itself, so we
/// can re-frame the raw content tokens into per-chunk windows. Some tokenizer
/// configurations add specials even with `add_special_tokens = false`; this
/// keeps the chunking math correct in both cases.
fn strip_specials(mut ids: Vec<u32>, bos_id: u32, eos_id: u32) -> Vec<u32> {
    if ids.first() == Some(&bos_id) {
        ids.remove(0);
    }
    if ids.last() == Some(&eos_id) {
        ids.pop();
    }
    ids
}

/// Pure chunking core: builds the `[BOS, content..., EOS, pad...]` token
/// vectors for a CLIP encoder, with each chunk padded to `CLIP_CHUNK_LEN`.
///
/// - Empty input yields a single `[BOS, EOS, EOS-pad x 75]` chunk so callers
///   never have to special-case "no prompt".
/// - The pad id is the EOS id (CLIP convention; matches both ComfyUI and the
///   reference `transformers` implementation).
pub(crate) fn chunk_token_ids(
    raw_ids: &[u32],
    max_per_chunk: usize,
    bos_id: u32,
    eos_id: u32,
) -> Vec<Vec<u32>> {
    let pad_id = eos_id;
    if raw_ids.is_empty() {
        let mut chunk = Vec::with_capacity(CLIP_CHUNK_LEN);
        chunk.push(bos_id);
        chunk.push(eos_id);
        chunk.resize(CLIP_CHUNK_LEN, pad_id);
        return vec![chunk];
    }

    raw_ids
        .chunks(max_per_chunk)
        .map(|window| {
            let mut chunk = Vec::with_capacity(CLIP_CHUNK_LEN);
            chunk.push(bos_id);
            chunk.extend_from_slice(window);
            chunk.push(eos_id);
            // Pad up to fixed window length with EOS (== pad).
            chunk.resize(CLIP_CHUNK_LEN, pad_id);
            chunk
        })
        .collect()
}

/// Tokenize `prompt` into CLIP-compatible chunked tensors, ready for
/// `ClipTextTransformer::forward`. Each tensor has shape `[1, 77]`.
pub(crate) fn tokenize_chunks(
    prompt: &str,
    tokenizer: &Tokenizer,
    device: &Device,
) -> Result<Vec<Tensor>> {
    let (bos_id, eos_id) = clip_special_ids(tokenizer);

    // `add_special_tokens = false` to get just the content tokens; some
    // tokenizers ignore this hint, so we defensively strip BOS/EOS afterwards.
    let raw = tokenizer
        .encode(prompt, false)
        .map_err(|e| anyhow::anyhow!("CLIP tokenization failed: {e}"))?
        .get_ids()
        .to_vec();
    let raw = strip_specials(raw, bos_id, eos_id);

    let chunks = chunk_token_ids(&raw, CLIP_MAX_TOKENS_PER_CHUNK, bos_id, eos_id);
    chunks
        .into_iter()
        .map(|ids| {
            Tensor::new(ids.as_slice(), device)
                .and_then(|t| t.unsqueeze(0))
                .map_err(Into::into)
        })
        .collect()
}

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
    pub tokenizer: Arc<Tokenizer>,
    pub device: Device,
    pub on_gpu: bool,
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
        })
    }

    /// Get a reference-counted handle to this encoder's tokenizer (for caching in SharedPool).
    pub fn tokenizer_arc(&self) -> Arc<Tokenizer> {
        self.tokenizer.clone()
    }

    /// Encode a text prompt into a single 768-dim pooled CLIP embedding.
    ///
    /// Default behavior (truncation): the prompt is tokenized once, truncated
    /// to 77 tokens (CLIP's positional limit), and the model returns a single
    /// pooled vector at the EOS position. Tokens past 77 are silently dropped.
    ///
    /// With `MOLD_LONG_PROMPTS=1`: the prompt is split into ComfyUI-style
    /// 75-token chunks (`[BOS] + 75 content + [EOS]` = 77 each); each chunk is
    /// encoded independently and the per-chunk pooled outputs are averaged
    /// into a single 768-dim vector. Output shape stays `[1, 768]` so FLUX's
    /// `vector_in` stem (the only consumer of `ClipEncoder`) keeps working.
    ///
    /// FLUX's CLIP path consumes a pooled vector, not a sequence — so chunking
    /// here cannot grow the conditioning sequence the way ComfyUI does for
    /// SDXL/SD3 cross-attention. The win is that CLIP now "sees" all of the
    /// user's prompt content (averaged) instead of just the first 75 tokens.
    pub fn encode(
        &mut self,
        prompt: &str,
        target_device: &Device,
        target_dtype: DType,
    ) -> Result<Tensor> {
        let emb = if long_prompts_enabled() {
            self.encode_chunked(prompt)?
        } else {
            self.encode_truncated(prompt)?
        };
        Ok(emb.to_device(target_device)?.to_dtype(target_dtype)?)
    }

    /// Legacy path: tokenize once, truncate at 77, run a single forward pass.
    /// Returns the pooled `[1, 768]` embedding on the encoder's own device.
    fn encode_truncated(&self, prompt: &str) -> Result<Tensor> {
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
        tokens.truncate(CLIP_CHUNK_LEN);

        let input_ids = Tensor::new(&tokens[..], &self.device)?.unsqueeze(0)?;
        Ok(clip.forward(&input_ids)?)
    }

    /// Long-prompt path: split into 75-token chunks, encode each, average the
    /// per-chunk pooled outputs. Returns a `[1, 768]` tensor on the encoder's
    /// own device.
    fn encode_chunked(&self, prompt: &str) -> Result<Tensor> {
        let clip = self
            .model
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("CLIP model unavailable"))?;

        let chunks = tokenize_chunks(prompt, &self.tokenizer, &self.device)?;
        debug_assert!(!chunks.is_empty(), "tokenize_chunks always emits ≥1 chunk");

        let mut pooled = Vec::with_capacity(chunks.len());
        for chunk in &chunks {
            // candle's `clip::text_model::ClipTextTransformer::forward` returns
            // pooled `[1, 768]` (hidden state at the argmax/EOS position).
            pooled.push(clip.forward(chunk)?);
        }

        if pooled.len() == 1 {
            return Ok(pooled.into_iter().next().expect("len==1"));
        }

        // Average pooled vectors across chunks: [N, 768] → [1, 768].
        let stacked = Tensor::cat(&pooled, 0)?;
        let mean = stacked.mean_keepdim(0)?;
        // `mean_keepdim(0)` of `[N, 768]` returns `[1, 768]` — exactly the
        // shape the truncation path produces.
        Ok(mean)
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
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Real CLIP-L tokenizer ids (from openai/clip-vit-large-patch14).
    const BOS_ID: u32 = 49406;
    const EOS_ID: u32 = 49407;

    // ---------------------------------------------------------------------
    // chunk_token_ids: pure logic, no tokenizer needed
    // ---------------------------------------------------------------------

    #[test]
    fn tokenize_chunks_short_prompt_one_chunk() {
        // 10-token "prompt" → 1 chunk of length 77 (BOS + 10 + EOS + 65 pad).
        let raw: Vec<u32> = (1..=10).collect();
        let chunks = chunk_token_ids(&raw, CLIP_MAX_TOKENS_PER_CHUNK, BOS_ID, EOS_ID);
        assert_eq!(chunks.len(), 1, "≤75 content tokens fit in one chunk");
        let chunk = &chunks[0];
        assert_eq!(chunk.len(), CLIP_CHUNK_LEN);
        assert_eq!(chunk[0], BOS_ID);
        assert_eq!(&chunk[1..=10], raw.as_slice());
        assert_eq!(chunk[11], EOS_ID);
        // Remaining slots are pad (== EOS).
        assert!(chunk[12..].iter().all(|&t| t == EOS_ID));
    }

    #[test]
    fn tokenize_chunks_two_chunks() {
        // 100 content tokens → 2 chunks (75 in chunk 1, 25 in chunk 2).
        let raw: Vec<u32> = (1..=100).collect();
        let chunks = chunk_token_ids(&raw, CLIP_MAX_TOKENS_PER_CHUNK, BOS_ID, EOS_ID);
        assert_eq!(chunks.len(), 2, "100 tokens straddle the 75-token boundary");

        // Chunk 1: BOS + tokens 1..=75 + EOS, fully packed.
        assert_eq!(chunks[0][0], BOS_ID);
        assert_eq!(&chunks[0][1..=75], &raw[..75]);
        assert_eq!(chunks[0][76], EOS_ID);

        // Chunk 2: BOS + tokens 76..=100 + EOS + pad.
        assert_eq!(chunks[1][0], BOS_ID);
        assert_eq!(&chunks[1][1..=25], &raw[75..]);
        assert_eq!(chunks[1][26], EOS_ID);
        assert!(chunks[1][27..].iter().all(|&t| t == EOS_ID));
    }

    #[test]
    fn tokenize_chunks_exact_75_boundary() {
        // Exactly 75 content tokens → still 1 chunk (75 + BOS + EOS = 77).
        // The next chunk would only contain BOS+EOS+padding — `chunks(75)`
        // does not emit empty trailing slices, so this is a single window.
        let raw: Vec<u32> = (1..=75).collect();
        let chunks = chunk_token_ids(&raw, CLIP_MAX_TOKENS_PER_CHUNK, BOS_ID, EOS_ID);
        assert_eq!(
            chunks.len(),
            1,
            "exactly 75 content tokens fit in one chunk"
        );
        assert_eq!(chunks[0].len(), CLIP_CHUNK_LEN);
        assert_eq!(chunks[0][0], BOS_ID);
        assert_eq!(chunks[0][76], EOS_ID, "EOS lands in the last slot");
    }

    #[test]
    fn tokenize_chunks_empty_prompt() {
        // Empty input → 1 chunk: [BOS, EOS, EOS x 75]. Caller never has to
        // special-case "no prompt".
        let raw: Vec<u32> = Vec::new();
        let chunks = chunk_token_ids(&raw, CLIP_MAX_TOKENS_PER_CHUNK, BOS_ID, EOS_ID);
        assert_eq!(chunks.len(), 1);
        let chunk = &chunks[0];
        assert_eq!(chunk.len(), CLIP_CHUNK_LEN);
        assert_eq!(chunk[0], BOS_ID);
        assert_eq!(chunk[1], EOS_ID);
        assert!(
            chunk[2..].iter().all(|&t| t == EOS_ID),
            "remaining slots are EOS-padded",
        );
    }

    #[test]
    fn tokenize_chunks_padding_uses_eos() {
        // CLIP convention: pad_id == eos_id. Verify by passing a distinct EOS
        // and confirming it shows up as both the trailing-EOS marker and the
        // pad fill.
        const CUSTOM_EOS: u32 = 12345;
        let raw: Vec<u32> = vec![10, 20, 30];
        let chunks = chunk_token_ids(&raw, CLIP_MAX_TOKENS_PER_CHUNK, BOS_ID, CUSTOM_EOS);
        assert_eq!(chunks.len(), 1);
        let chunk = &chunks[0];
        assert_eq!(chunk[0], BOS_ID);
        assert_eq!(chunk[1..=3], [10, 20, 30]);
        assert_eq!(chunk[4], CUSTOM_EOS, "EOS marker after content");
        assert!(
            chunk[5..].iter().all(|&t| t == CUSTOM_EOS),
            "pad_id == eos_id: padding fills with EOS",
        );
    }

    #[test]
    fn tokenize_chunks_three_chunks_at_150_tokens() {
        // 150 tokens → 2 chunks (75 + 75) with last fully packed.
        let raw: Vec<u32> = (1..=150).collect();
        let chunks = chunk_token_ids(&raw, CLIP_MAX_TOKENS_PER_CHUNK, BOS_ID, EOS_ID);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[1][0], BOS_ID);
        assert_eq!(&chunks[1][1..=75], &raw[75..]);
        assert_eq!(chunks[1][76], EOS_ID);

        // 151 tokens → 3 chunks.
        let raw: Vec<u32> = (1..=151).collect();
        let chunks = chunk_token_ids(&raw, CLIP_MAX_TOKENS_PER_CHUNK, BOS_ID, EOS_ID);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[2][1], 151, "last chunk holds the trailing token");
        assert_eq!(chunks[2][2], EOS_ID, "EOS immediately after the lone token");
    }

    // ---------------------------------------------------------------------
    // strip_specials: defensive cleanup of tokenizer-added BOS/EOS
    // ---------------------------------------------------------------------

    #[test]
    fn strip_specials_removes_leading_bos_and_trailing_eos() {
        let stripped = strip_specials(vec![BOS_ID, 10, 20, EOS_ID], BOS_ID, EOS_ID);
        assert_eq!(stripped, vec![10, 20]);
    }

    #[test]
    fn strip_specials_leaves_clean_input_alone() {
        let stripped = strip_specials(vec![10, 20, 30], BOS_ID, EOS_ID);
        assert_eq!(stripped, vec![10, 20, 30]);
    }

    #[test]
    fn strip_specials_handles_only_specials() {
        let stripped = strip_specials(vec![BOS_ID, EOS_ID], BOS_ID, EOS_ID);
        assert_eq!(stripped, vec![] as Vec<u32>);
    }

    // ---------------------------------------------------------------------
    // long_prompts_enabled: env var contract
    // ---------------------------------------------------------------------
    //
    // These tests serialize via a static mutex because Rust's test runner
    // runs the suite in parallel and `MOLD_LONG_PROMPTS` is process-global.

    fn env_lock() -> std::sync::MutexGuard<'static, ()> {
        use std::sync::{Mutex, OnceLock};
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(|p| p.into_inner())
    }

    #[test]
    fn long_prompts_enabled_env_default_off() {
        let _guard = env_lock();
        // SAFETY: serialized with env_lock() to avoid racing other env tests.
        unsafe { std::env::remove_var("MOLD_LONG_PROMPTS") };
        assert!(!long_prompts_enabled(), "default must be off");
    }

    #[test]
    fn long_prompts_enabled_env_set_to_1() {
        let _guard = env_lock();
        // SAFETY: serialized with env_lock() to avoid racing other env tests.
        unsafe { std::env::set_var("MOLD_LONG_PROMPTS", "1") };
        let on = long_prompts_enabled();
        unsafe { std::env::remove_var("MOLD_LONG_PROMPTS") };
        assert!(on, "MOLD_LONG_PROMPTS=1 must enable chunking");
    }

    #[test]
    fn long_prompts_enabled_env_other_value_off() {
        let _guard = env_lock();
        // Only the literal "1" enables the feature; "true", "yes", "0" stay off.
        unsafe { std::env::set_var("MOLD_LONG_PROMPTS", "true") };
        let on = long_prompts_enabled();
        unsafe { std::env::remove_var("MOLD_LONG_PROMPTS") };
        assert!(!on, "only the literal '1' should enable chunking");
    }

    // ---------------------------------------------------------------------
    // tokenize_chunks (integration with a real tokenizer)
    //
    // These tests use the local CLIP-L tokenizer if present. We probe the
    // standard mold model cache path; absent that, the test is skipped via
    // a soft-pass with an explanatory message.
    // ---------------------------------------------------------------------

    fn try_load_clip_tokenizer() -> Option<Tokenizer> {
        let candidates = [
            std::env::var("MOLD_TEST_CLIP_TOKENIZER").ok(),
            std::env::var("HOME")
                .ok()
                .map(|h| format!("{h}/.mold/models/shared/clip-vit-large-patch14/tokenizer.json")),
        ];
        for path in candidates.into_iter().flatten() {
            if std::path::Path::new(&path).exists() {
                if let Ok(tok) = Tokenizer::from_file(&path) {
                    return Some(tok);
                }
            }
        }
        None
    }

    #[test]
    fn tokenize_chunks_short_prompt_with_real_tokenizer() {
        let Some(tokenizer) = try_load_clip_tokenizer() else {
            eprintln!(
                "skipping: no CLIP tokenizer fixture (set MOLD_TEST_CLIP_TOKENIZER \
                 or place tokenizer.json under ~/.mold/models/shared/clip-vit-large-patch14/)",
            );
            return;
        };
        let chunks = tokenize_chunks("a cat", &tokenizer, &Device::Cpu)
            .expect("real tokenizer must accept a short prompt");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].dims(), &[1, CLIP_CHUNK_LEN]);
    }

    #[test]
    fn tokenize_chunks_long_prompt_with_real_tokenizer_grows_chunks() {
        let Some(tokenizer) = try_load_clip_tokenizer() else {
            eprintln!(
                "skipping: no CLIP tokenizer fixture (set MOLD_TEST_CLIP_TOKENIZER \
                 or place tokenizer.json under ~/.mold/models/shared/clip-vit-large-patch14/)",
            );
            return;
        };
        // Build a prompt long enough to guarantee at least 2 chunks. CLIP-L
        // tokenizes a single short word as ≥1 token, so 200 repetitions of
        // "alpha" definitely overflows 75 content tokens.
        let prompt = "alpha ".repeat(200);
        let short =
            tokenize_chunks("a cat", &tokenizer, &Device::Cpu).expect("short prompt tokenizes");
        let long = tokenize_chunks(prompt.trim(), &tokenizer, &Device::Cpu)
            .expect("long prompt tokenizes");
        assert!(
            long.len() > short.len(),
            "long prompt produces more chunks than a short one ({} vs {})",
            long.len(),
            short.len(),
        );
        // Every chunk has the canonical [1, 77] shape.
        for chunk in &long {
            assert_eq!(chunk.dims(), &[1, CLIP_CHUNK_LEN]);
        }
    }
}
