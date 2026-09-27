//! Native Qwen Image 2.1 implementation.
//!
//! Qwen Image 2.1 is not wire-compatible with the older Qwen-Image 2512
//! engine.  In particular it has a Qwen3-VL text encoder, a 64-channel /16
//! VAE, and a 32-block causal-condition transformer.  Keep its implementation
//! isolated here rather than using a model-name branch in `qwen_image`, where
//! doing so would make the two checkpoint layouts look interchangeable.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};

use crate::encoders::qwen3::{resolve_pad_token_id, Qwen3Encoder};

pub(crate) mod attention;
pub(crate) mod banded_conv;
pub(crate) mod conditioning;
pub(crate) mod exec_path;
pub(crate) mod layout;
pub(crate) mod linear;
pub(crate) mod lora;
pub(crate) mod pipeline;
pub(crate) mod reference;
pub(crate) mod scheduler;
pub mod text_encoder_residency;
#[cfg(test)]
mod tier_renders;
pub(crate) mod transformer;
pub(crate) mod vae;
pub(crate) mod vae_encoder;

#[cfg(test)]
mod parity_tests;

pub use pipeline::QwenImage21Engine;

/// Only this family's Metal denoiser uses the qualified BF16 path. Encoding
/// and decoding retain `gpu_dtype`; other devices retain their existing policy.
pub(crate) fn transformer_dtype(device: &Device) -> DType {
    if device.is_metal() {
        metal_transformer_dtype(crate::runtime_env::value("MOLD_QWEN_IMAGE21_DTYPE").as_deref())
    } else {
        crate::engine::gpu_dtype(device)
    }
}

/// Shared precision parser for Metal loading and frozen execution identity.
pub fn metal_transformer_dtype(value: Option<&str>) -> DType {
    match value.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
        None | Some("" | "auto" | "bf16") => DType::BF16,
        Some("f32" | "fp32") => DType::F32,
        Some(other) => {
            tracing::warn!(
                value = other,
                "MOLD_QWEN_IMAGE21_DTYPE must be auto/bf16/f32; using BF16"
            );
            DType::BF16
        }
    }
}

/// The `MOLD_QWEN_IMAGE21_QMATMUL` decision for a raw value — the engine's own
/// parser, exported so mold-server's execution identity canonicalizes the
/// spelling to the arm that will actually run.
pub fn qmatmul_env_enabled(value: Option<&str>) -> bool {
    linear::parse_qwen_image21_qmatmul(value)
}

/// v0.32's per-branch retention bound, and still the whole rule for
/// text-to-image: a prompt of at most this many rows retains its prefix and a
/// longer one recomputes it, exactly as v0.32 did, so no text-to-image byte
/// moves on any card.
pub(crate) const LEGACY_PREFIX_CACHE_TOKENS: usize = 512;

/// The retained prefix of every branch of a REFERENCE-conditioned request may
/// take at most this much together under the request-only rule
/// ([`PrefixCacheBudget::RequestOnly`]: the legacy path, Metal and CPU). One
/// 1024² reference with CFG in BF16 (~4.6 GB) retains; three do not. The CUDA
/// fast path replaces this constant with the card's headroom.
pub(crate) const PREFIX_CACHE_REFERENCE_BUDGET_BYTES: u64 = 6 << 30;

/// Allocator margin the fast path keeps free beyond the denoise workspace and
/// the retained cache, so retention never takes the last gigabyte a
/// fragmented pool needs (the `still_transformer_residency` margin).
pub(crate) const PREFIX_CACHE_MARGIN_BYTES: u64 = 1 << 30;

/// How much memory the prefix-cache decision may consider.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefixCacheBudget {
    /// The request-only rule: v0.32's 512-row rule for text-to-image and
    /// [`PREFIX_CACHE_REFERENCE_BUDGET_BYTES`] for references. The same
    /// request retains identically on every card. This is the legacy path
    /// (`MOLD_ATTN=math`, byte-identical to v0.32), Metal and CPU.
    RequestOnly,
    /// The CUDA fast path: retain every branch whenever the summed cache fits
    /// these bytes — what the card has left for it ([`prefix_cache_headroom`]).
    /// Upstream always caches (`use_kv_cache=True`,
    /// `pipeline_qwenimage21.py:528`, extracted at step 0 and reused,
    /// `:750-764`) and simply runs out of memory where it cannot; mold
    /// recomputes instead, so on this path retention — and with it the exact
    /// pixels of a very long prefix (`:585-589`) — depends on available memory.
    Headroom(u64),
}

/// Bytes a retained prefix cache may take: `free_bytes` less every weight
/// still to be counted against it, the denoise workspace and
/// [`PREFIX_CACHE_MARGIN_BYTES`]. The engine passes the free memory it
/// samples at denoise (weights already resident, so `resident_bytes` is 0);
/// admission and the text-encoder residency plan pass the usable card and the
/// transformer plus VAE — the same fit question.
pub fn prefix_cache_headroom(free_bytes: u64, resident_bytes: u64, workspace_bytes: u64) -> u64 {
    free_bytes
        .saturating_sub(resident_bytes)
        .saturating_sub(workspace_bytes)
        .saturating_sub(PREFIX_CACHE_MARGIN_BYTES)
}

/// Whether this process's renders take the fast path's memory-following cache
/// rule: a CUDA (non-Metal) build whose `MOLD_ATTN` does not select the legacy
/// path. Admission asks this; the engine asks its own transformer's resolved
/// [`exec_path::Qwen21ExecPath`] and device.
pub fn prefix_cache_follows_memory() -> bool {
    cfg!(feature = "cuda")
        && !cfg!(feature = "metal")
        && !exec_path::Qwen21ExecPath::resolve_for(
            exec_path::ExecDevice::Cuda,
            crate::attention::requested_backend(),
        )
        .is_legacy()
}

/// Both CFG branches at the legacy retained length, in the widest supported
/// working dtype (F32 on Metal/CPU). Added outside the activation area estimate.
pub(crate) fn prefix_cache_budget_bytes(batch: u32) -> u64 {
    2 * prefix_cache_bytes(LEGACY_PREFIX_CACHE_TOKENS, batch as usize, 4)
}

/// Bytes one branch's retained prefix K/V takes: key and value, every layer,
/// every head, per prefix token (`transformer_qwenimage21.py:56-85` stores
/// post-RoPE `(B, prefix, heads, head_dim)` per layer).
pub(crate) fn prefix_cache_bytes(prefix_tokens: usize, batch: usize, dtype_bytes: usize) -> u64 {
    let cfg = transformer::QwenImage21TransformerConfig::official();
    2 * cfg.num_layers as u64
        * (cfg.num_attention_heads * cfg.attention_head_dim) as u64
        * prefix_tokens as u64
        * dtype_bytes as u64
        * batch.max(1) as u64
}

/// `MOLD_QWEN_IMAGE21_KV_CACHE`: whether a request retains its prefix K/V.
///
/// Upstream always caches (`use_kv_cache=True`, `pipeline_qwenimage21.py:528`)
/// and warns that cached and uncached renders are not bit-identical in BF16
/// (`:585-589`), so the answer MOVES PIXELS and is engine-shaping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefixCacheMode {
    /// [`PrefixCachePolicy::resolve`]'s default rule: the request-only rule
    /// on the legacy path, the card's headroom on the CUDA fast path.
    Auto,
    /// Always retain.
    On,
    /// Always recompute the prefix every step.
    Off,
}

/// The one parser for `MOLD_QWEN_IMAGE21_KV_CACHE`, shared by the engine and
/// mold-server's execution-equivalence classifier. Unset, empty, `auto` and
/// any unrecognized spelling are `Auto`.
pub fn parse_prefix_cache_mode(value: Option<&str>) -> PrefixCacheMode {
    match value.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
        Some("on" | "1" | "true" | "yes") => PrefixCacheMode::On,
        Some("off" | "0" | "false" | "no") => PrefixCacheMode::Off,
        None | Some("" | "auto") => PrefixCacheMode::Auto,
        Some(other) => {
            tracing::warn!(
                value = other,
                "MOLD_QWEN_IMAGE21_KV_CACHE must be auto/on/off; using auto"
            );
            PrefixCacheMode::Auto
        }
    }
}

/// The process's `MOLD_QWEN_IMAGE21_KV_CACHE`.
pub fn prefix_cache_mode_from_env() -> PrefixCacheMode {
    parse_prefix_cache_mode(crate::runtime_env::value("MOLD_QWEN_IMAGE21_KV_CACHE").as_deref())
}

/// What one conditioning branch does with its prefix K/V.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PrefixCacheDecision {
    /// Prefill once, then run only the target block on later steps.
    Retain,
    /// Recompute the whole joint sequence every step.
    Recompute,
}

/// The one decision for prefix K/V retention, read by the engine and by
/// admission ([`crate::device::qwen_image21_prefix_cache_bytes`]).
pub(crate) struct PrefixCachePolicy;

impl PrefixCachePolicy {
    /// Decide for every branch of one request.
    ///
    /// `branch_prefix_tokens` holds each branch's prefix length (text plus
    /// condition-image tokens) and `has_condition` whether the request carries
    /// reference images. Under `Auto`:
    ///
    /// - [`PrefixCacheBudget::Headroom`] (CUDA fast path): EVERY branch
    ///   retains iff their summed cache fits the headroom, else every branch
    ///   recomputes (upstream always caches; a card that cannot hold it
    ///   recomputes rather than running out of memory);
    /// - [`PrefixCacheBudget::RequestOnly`]: text-to-image keeps v0.32's
    ///   per-branch 512-token rule, and a reference-conditioned request
    ///   retains every branch iff their summed cache fits
    ///   [`PREFIX_CACHE_REFERENCE_BUDGET_BYTES`].
    ///
    /// `On` / `Off` override both.
    pub(crate) fn resolve(
        branch_prefix_tokens: &[usize],
        has_condition: bool,
        batch: usize,
        dtype_bytes: usize,
        mode: PrefixCacheMode,
        budget: PrefixCacheBudget,
    ) -> Vec<PrefixCacheDecision> {
        let all = |decision| vec![decision; branch_prefix_tokens.len()];
        let summed = prefix_cache_total_bytes(branch_prefix_tokens, batch, dtype_bytes);
        let retain_if = |fits: bool| {
            all(if fits {
                PrefixCacheDecision::Retain
            } else {
                PrefixCacheDecision::Recompute
            })
        };
        match (mode, budget) {
            (PrefixCacheMode::On, _) => all(PrefixCacheDecision::Retain),
            (PrefixCacheMode::Off, _) => all(PrefixCacheDecision::Recompute),
            (PrefixCacheMode::Auto, PrefixCacheBudget::Headroom(bytes)) => {
                retain_if(summed <= bytes)
            }
            (PrefixCacheMode::Auto, PrefixCacheBudget::RequestOnly) if has_condition => {
                retain_if(summed <= PREFIX_CACHE_REFERENCE_BUDGET_BYTES)
            }
            (PrefixCacheMode::Auto, PrefixCacheBudget::RequestOnly) => branch_prefix_tokens
                .iter()
                .map(|&tokens| {
                    if tokens <= LEGACY_PREFIX_CACHE_TOKENS {
                        PrefixCacheDecision::Retain
                    } else {
                        PrefixCacheDecision::Recompute
                    }
                })
                .collect(),
        }
    }

    /// [`Self::resolve`] under the process's override.
    pub(crate) fn resolve_from_env(
        branch_prefix_tokens: &[usize],
        has_condition: bool,
        batch: usize,
        dtype: DType,
        budget: PrefixCacheBudget,
    ) -> Vec<PrefixCacheDecision> {
        Self::resolve(
            branch_prefix_tokens,
            has_condition,
            batch,
            dtype.size_in_bytes(),
            prefix_cache_mode_from_env(),
            budget,
        )
    }
}

/// Every branch's retained cache together.
pub(crate) fn prefix_cache_total_bytes(
    branch_prefix_tokens: &[usize],
    batch: usize,
    dtype_bytes: usize,
) -> u64 {
    branch_prefix_tokens
        .iter()
        .map(|&tokens| prefix_cache_bytes(tokens, batch, dtype_bytes))
        .sum()
}

/// The request warning a render carries when the automatic rule made it
/// recompute its prefix every step (never for an explicit `off`).
pub(crate) fn prefix_cache_recompute_warning(
    cache_bytes: u64,
    budget: PrefixCacheBudget,
) -> String {
    let gib = |bytes: u64| bytes as f64 / (1u64 << 30) as f64;
    match budget {
        PrefixCacheBudget::Headroom(headroom) => format!(
            "Qwen Image 2.1 recomputed its {:.1} GiB prefix cache every step because it did not fit the {:.1} GiB this card had left (slower, and pixels can differ slightly from a cached render); MOLD_QWEN_IMAGE21_KV_CACHE=on forces the cache.",
            gib(cache_bytes),
            gib(headroom)
        ),
        PrefixCacheBudget::RequestOnly => format!(
            "Qwen Image 2.1 recomputed its {:.1} GiB prefix cache every step (over the request-only retention budget of the math/legacy path); MOLD_QWEN_IMAGE21_KV_CACHE=on forces the cache.",
            gib(cache_bytes)
        ),
    }
}

/// The fixed system message from the upstream `QwenImage21Pipeline`.
pub(crate) const QWEN_IMAGE_21_SYSTEM_PROMPT: &str = "Comprehend and analyze the provided prompt.";

/// Qwen Image 2.1 consumes unpatched 64-channel VAE latents.
pub(crate) const QWEN_IMAGE_21_LATENT_CHANNELS: usize = 64;
/// One VAE latent cell spans a 16x16 pixel tile.
pub(crate) const QWEN_IMAGE_21_VAE_SCALE_FACTOR: usize = 16;
/// The reference rounds output dimensions down to a pair of latent cells.
/// This makes final canvases multiples of 32 pixels.
pub(crate) const QWEN_IMAGE_21_CANVAS_ALIGNMENT: usize = QWEN_IMAGE_21_VAE_SCALE_FACTOR * 2;

/// The raw prompt form passed to Qwen3-VL's processor for text-to-image.
///
/// This must remain a direct template string.  The upstream pipeline calls
/// its processor with this string instead of `apply_chat_template`, and the
/// two paths deliberately tokenize differently for a number of chat
/// templates.
pub(crate) fn t2i_prompt_template(prompt: &str) -> String {
    let prompt = if prompt.is_empty() { " " } else { prompt };
    format!(
        "<|im_start|>system\n{QWEN_IMAGE_21_SYSTEM_PROMPT}<|im_end|>\n\
         <|im_start|>user\n{prompt}<|im_end|>\n\
         <|im_start|>assistant\n"
    )
}

/// The system-role prefix whose token count is removed after encoding.
///
/// It is intentionally derived through the loaded tokenizer at runtime rather
/// than kept as a magic token count.  This is the same invariant as the
/// reference's `processor.apply_chat_template(system_message, tokenize=True)`.
pub(crate) fn system_message_prefix() -> String {
    format!("<|im_start|>system\n{QWEN_IMAGE_21_SYSTEM_PROMPT}<|im_end|>\n")
}

/// Native text conditioning returned by [`encode_t2i_prompts`].
///
/// `valid_tokens` is deliberately held as booleans rather than a Tensor: the
/// transformer needs a per-row block-causal key mask, and keeping it in this
/// shape avoids a device-to-host sync merely to discover which prompt rows
/// were left-padded.  `image_slots` stays all-false for text-to-image; the
/// type makes the later image-conditioned path explicit instead of smuggling
/// that state through a convention.
pub(crate) struct QwenImage21TextConditioning {
    pub embeddings: Tensor,
    pub valid_tokens: Vec<Vec<bool>>,
    pub image_slots: Vec<Vec<bool>>,
}

impl QwenImage21TextConditioning {
    pub(crate) fn batch_size(&self) -> usize {
        self.valid_tokens.len()
    }

    pub(crate) fn sequence_length(&self) -> usize {
        self.valid_tokens.first().map_or(0, Vec::len)
    }

    pub(crate) fn to_device_dtype(&self, device: &Device, dtype: DType) -> Result<Self> {
        Ok(Self {
            embeddings: self.embeddings.to_device(device)?.to_dtype(dtype)?,
            valid_tokens: self.valid_tokens.clone(),
            image_slots: self.image_slots.clone(),
        })
    }
}

/// Construct Qwen3-VL's left-padded input batch.
///
/// The processor specifically selects `padding_side="left"`, so the pad
/// positions affect the language model's rotary positions even though they are
/// discarded immediately after the encoder.  Do not replace this with the
/// right-padding helper used by Flux.2 Klein.
fn left_pad_tokens(
    token_batches: &[Vec<u32>],
    pad_id: u32,
) -> Result<(Vec<u32>, Vec<Vec<bool>>, usize)> {
    anyhow::ensure!(
        !token_batches.is_empty(),
        "Qwen Image 2.1 needs at least one prompt"
    );
    let width = token_batches.iter().map(Vec::len).max().unwrap_or(0);
    anyhow::ensure!(
        width > 0,
        "Qwen Image 2.1 prompt tokenization produced no tokens"
    );

    let mut ids = Vec::with_capacity(token_batches.len() * width);
    let mut attention = Vec::with_capacity(token_batches.len());
    for tokens in token_batches {
        anyhow::ensure!(
            !tokens.is_empty(),
            "Qwen Image 2.1 prompt tokenization produced an empty prompt row"
        );
        let pad = width - tokens.len();
        ids.extend(std::iter::repeat_n(pad_id, pad));
        ids.extend_from_slice(tokens);
        let mut row = vec![false; pad];
        row.extend(std::iter::repeat_n(true, tokens.len()));
        attention.push(row);
    }
    Ok((ids, attention, width))
}

/// Removes left padding and the system-role hidden states, then right-pads
/// the retained prompt conditioning rows for the image transformer.
fn trim_and_right_pad_conditioning(
    hidden_states: &Tensor,
    source_attention: &[Vec<bool>],
    drop_idx: usize,
) -> Result<(Tensor, Vec<Vec<bool>>)> {
    let (batch, source_len, hidden) = hidden_states.dims3()?;
    anyhow::ensure!(
        source_attention.len() == batch,
        "Qwen Image 2.1 conditioning batch mismatch: expected {batch}, got {}",
        source_attention.len()
    );

    let mut rows = Vec::with_capacity(batch);
    let mut retained_lengths = Vec::with_capacity(batch);
    for (index, attention) in source_attention.iter().enumerate() {
        anyhow::ensure!(
            attention.len() == source_len,
            "Qwen Image 2.1 conditioning mask row {index} has {}, expected {source_len}",
            attention.len()
        );
        let left_pad = attention.iter().take_while(|value| !**value).count();
        anyhow::ensure!(
            attention[left_pad..].iter().all(|value| *value),
            "Qwen Image 2.1 conditioning mask row {index} must be left-padded"
        );
        let real_len = source_len - left_pad;
        anyhow::ensure!(
            real_len > drop_idx,
            "Qwen Image 2.1 system prefix ({drop_idx} tokens) consumes prompt row {index} ({real_len} tokens)"
        );
        rows.push(hidden_states.narrow(0, index, 1)?.narrow(
            1,
            left_pad + drop_idx,
            real_len - drop_idx,
        )?);
        retained_lengths.push(real_len - drop_idx);
    }

    let target_len = retained_lengths.iter().copied().max().unwrap_or(0);
    let mut padded_rows = Vec::with_capacity(batch);
    let mut valid_tokens = Vec::with_capacity(batch);
    for (row, retained) in rows.iter().zip(&retained_lengths) {
        let mut pieces = vec![row.clone()];
        if *retained < target_len {
            pieces.push(Tensor::zeros(
                (1, target_len - retained, hidden),
                hidden_states.dtype(),
                hidden_states.device(),
            )?);
        }
        let piece_refs: Vec<&Tensor> = pieces.iter().collect();
        padded_rows.push(Tensor::cat(&piece_refs, 1)?);

        let mut row_mask = vec![true; *retained];
        row_mask.extend(std::iter::repeat_n(false, target_len - retained));
        valid_tokens.push(row_mask);
    }
    let row_refs: Vec<&Tensor> = padded_rows.iter().collect();
    Ok((Tensor::cat(&row_refs, 0)?, valid_tokens))
}

/// Encode text-only Qwen Image 2.1 prompts exactly as the upstream pipeline
/// does before the diffusion transformer:
///
/// 1. render the raw Qwen3-VL template;
/// 2. left-pad the batch and hide pad keys in the language model;
/// 3. capture the final decoder state before Qwen3-VL's final RMSNorm;
/// 4. remove the tokenized system-role prefix; and
/// 5. right-pad the transformer condition rows with zero embeddings.
pub(crate) fn encode_t2i_prompts(
    encoder: &mut Qwen3Encoder,
    prompts: &[String],
) -> Result<QwenImage21TextConditioning> {
    anyhow::ensure!(
        !prompts.is_empty(),
        "Qwen Image 2.1 needs at least one prompt"
    );
    let token_batches = prompts
        .iter()
        .map(|prompt| {
            encoder
                .tokenizer
                .encode(t2i_prompt_template(prompt), true)
                .map(|encoding| encoding.get_ids().to_vec())
                .map_err(|err| anyhow::anyhow!("Qwen Image 2.1 prompt tokenization failed: {err}"))
        })
        .collect::<Result<Vec<_>>>()?;
    let drop_idx = encoder
        .tokenizer
        .encode(system_message_prefix(), true)
        .map_err(|err| anyhow::anyhow!("Qwen Image 2.1 system prompt tokenization failed: {err}"))?
        .len();
    let (ids, source_attention, width) =
        left_pad_tokens(&token_batches, resolve_pad_token_id(&encoder.tokenizer))?;
    let input_ids = Tensor::from_vec(ids, (prompts.len(), width), &encoder.device)?;
    let hidden_states =
        encoder.forward_final_pre_norm_with_attention(&input_ids, Some(&source_attention))?;
    let (embeddings, valid_tokens) =
        trim_and_right_pad_conditioning(&hidden_states, &source_attention, drop_idx)?;
    let image_slots = valid_tokens
        .iter()
        .map(|row| vec![false; row.len()])
        .collect();
    Ok(QwenImage21TextConditioning {
        embeddings,
        valid_tokens,
        image_slots,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn metal_denoiser_precision_has_an_explicit_f32_fallback() {
        for value in [None, Some("auto"), Some("BF16"), Some("")] {
            assert_eq!(metal_transformer_dtype(value), DType::BF16);
        }
        for value in [Some("f32"), Some(" FP32 ")] {
            assert_eq!(metal_transformer_dtype(value), DType::F32);
        }
        assert_eq!(metal_transformer_dtype(Some("unsupported")), DType::BF16);
        assert_eq!(transformer_dtype(&Device::Cpu), DType::F32);
    }

    #[test]
    fn prefix_cache_mode_parser_accepts_the_documented_spellings() {
        for value in [None, Some(""), Some("auto"), Some(" AUTO "), Some("bogus")] {
            assert_eq!(parse_prefix_cache_mode(value), PrefixCacheMode::Auto);
        }
        for value in ["on", "1", "true", " ON "] {
            assert_eq!(parse_prefix_cache_mode(Some(value)), PrefixCacheMode::On);
        }
        for value in ["off", "0", "false", "Off"] {
            assert_eq!(parse_prefix_cache_mode(Some(value)), PrefixCacheMode::Off);
        }
    }

    #[test]
    fn prefix_cache_policy_is_a_property_of_the_request() {
        use PrefixCacheDecision::{Recompute, Retain};
        // 512 KiB per BF16 prefix token per branch.
        assert_eq!(prefix_cache_bytes(1, 1, 2), 512 * 1024);
        let auto = PrefixCacheMode::Auto;
        // Text-to-image keeps v0.32's per-branch 512-token rule exactly.
        assert_eq!(
            PrefixCachePolicy::resolve(
                &[512, 513],
                false,
                1,
                2,
                auto,
                PrefixCacheBudget::RequestOnly
            ),
            vec![Retain, Recompute]
        );
        assert_eq!(
            PrefixCachePolicy::resolve(
                &[27, 30],
                false,
                1,
                4,
                auto,
                PrefixCacheBudget::RequestOnly
            ),
            vec![Retain, Retain]
        );
        // One 1024² reference + prompt ≈ 4.2k prefix tokens ≈ 2.1 GiB per
        // BF16 branch: both CFG branches retain under the 6 GiB budget.
        assert_eq!(
            PrefixCachePolicy::resolve(
                &[4200, 4150],
                true,
                1,
                2,
                auto,
                PrefixCacheBudget::RequestOnly
            ),
            vec![Retain, Retain]
        );
        // The same request at F32 (8.3 GiB) recomputes, and so do three
        // references; the decision is shared so both branches agree.
        assert_eq!(
            PrefixCachePolicy::resolve(
                &[4200, 4150],
                true,
                1,
                4,
                auto,
                PrefixCacheBudget::RequestOnly
            ),
            vec![Recompute, Recompute]
        );
        assert_eq!(
            PrefixCachePolicy::resolve(
                &[12_500, 12_450],
                true,
                1,
                2,
                auto,
                PrefixCacheBudget::RequestOnly
            ),
            vec![Recompute, Recompute]
        );
        // Overrides win in both directions.
        assert_eq!(
            PrefixCachePolicy::resolve(
                &[40_000],
                true,
                1,
                2,
                PrefixCacheMode::On,
                PrefixCacheBudget::RequestOnly
            ),
            vec![Retain]
        );
        assert_eq!(
            PrefixCachePolicy::resolve(
                &[3],
                false,
                1,
                2,
                PrefixCacheMode::Off,
                PrefixCacheBudget::RequestOnly
            ),
            vec![Recompute]
        );
    }

    /// On the CUDA fast path the cache follows the memory the render has
    /// left, as upstream's always-on cache does (`P:528`, `:750-764`): every
    /// branch retains when their summed cache fits the headroom, and all
    /// recompute together when it does not. The legacy rule is untouched.
    #[test]
    fn fast_path_retention_follows_the_memory_headroom() {
        use PrefixCacheDecision::{Recompute, Retain};
        let auto = PrefixCacheMode::Auto;
        // Three 1024² references with CFG: ~16.8k prefix tokens per branch,
        // 8.2 GiB each in BF16 — over the request-only 6 GiB budget.
        let three = [16_800, 16_790];
        let cache: u64 = three
            .iter()
            .map(|&tokens| prefix_cache_bytes(tokens, 1, 2))
            .sum();
        assert_eq!(
            PrefixCachePolicy::resolve(&three, true, 1, 2, auto, PrefixCacheBudget::RequestOnly),
            vec![Recompute, Recompute]
        );
        assert_eq!(
            PrefixCachePolicy::resolve(
                &three,
                true,
                1,
                2,
                auto,
                PrefixCacheBudget::Headroom(cache)
            ),
            vec![Retain, Retain]
        );
        assert_eq!(
            PrefixCachePolicy::resolve(
                &three,
                true,
                1,
                2,
                auto,
                PrefixCacheBudget::Headroom(cache - 1)
            ),
            vec![Recompute, Recompute]
        );
        // A long text-to-image prompt retains on the fast path when it fits,
        // where v0.32's 512-row rule recomputed it.
        assert_eq!(
            PrefixCachePolicy::resolve(
                &[900],
                false,
                1,
                2,
                auto,
                PrefixCacheBudget::Headroom(u64::MAX)
            ),
            vec![Retain]
        );
        // Overrides still win over the headroom in both directions.
        assert_eq!(
            PrefixCachePolicy::resolve(
                &three,
                true,
                1,
                2,
                PrefixCacheMode::On,
                PrefixCacheBudget::Headroom(0)
            ),
            vec![Retain, Retain]
        );
        assert_eq!(
            PrefixCachePolicy::resolve(
                &[3],
                false,
                1,
                2,
                PrefixCacheMode::Off,
                PrefixCacheBudget::Headroom(u64::MAX)
            ),
            vec![Recompute]
        );
    }

    /// The headroom is what the card has left beyond the resident weights,
    /// the denoise workspace and one allocator margin — one formula for the
    /// engine and for admission.
    #[test]
    fn prefix_cache_headroom_subtracts_workspace_and_margin() {
        let gib = 1u64 << 30;
        assert_eq!(
            prefix_cache_headroom(40 * gib, 15 * gib, 4 * gib),
            40 * gib - 15 * gib - 4 * gib - PREFIX_CACHE_MARGIN_BYTES
        );
        assert_eq!(prefix_cache_headroom(10 * gib, 15 * gib, 4 * gib), 0);
    }

    #[test]
    fn t2i_template_is_the_upstream_raw_processor_template() {
        assert_eq!(
            t2i_prompt_template("a capybara"),
            "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n\
             <|im_start|>user\na capybara<|im_end|>\n<|im_start|>assistant\n"
        );
        assert!(t2i_prompt_template("").contains("user\n <|im_end|>"));
    }

    #[test]
    fn left_padding_preserves_each_prompt_position_layout() {
        let (ids, mask, width) =
            left_pad_tokens(&[vec![10, 11], vec![20, 21, 22, 23]], 99).unwrap();
        assert_eq!(width, 4);
        assert_eq!(ids, vec![99, 99, 10, 11, 20, 21, 22, 23]);
        assert_eq!(mask, vec![vec![false, false, true, true], vec![true; 4]]);
    }

    #[test]
    fn trimming_drops_left_padding_and_system_prefix_then_right_pads() {
        let device = Device::Cpu;
        let hidden = Tensor::arange(0f32, 2.0 * 5.0 * 2.0, &device)
            .unwrap()
            .reshape((2, 5, 2))
            .unwrap();
        let attention = vec![vec![false, false, true, true, true], vec![true; 5]];
        let (got, mask) = trim_and_right_pad_conditioning(&hidden, &attention, 1).unwrap();
        assert_eq!(got.dims(), &[2, 4, 2]);
        assert_eq!(mask, vec![vec![true, true, false, false], vec![true; 4]]);
        assert_eq!(
            got.to_vec3::<f32>().unwrap(),
            vec![
                vec![
                    vec![6.0, 7.0],
                    vec![8.0, 9.0],
                    vec![0.0, 0.0],
                    vec![0.0, 0.0]
                ],
                vec![
                    vec![12.0, 13.0],
                    vec![14.0, 15.0],
                    vec![16.0, 17.0],
                    vec![18.0, 19.0]
                ],
            ]
        );
    }
}
