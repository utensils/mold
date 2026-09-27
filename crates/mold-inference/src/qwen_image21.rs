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
pub(crate) mod conditioning;
pub(crate) mod layout;
pub(crate) mod pipeline;
pub(crate) mod scheduler;
pub(crate) mod transformer;
pub(crate) mod vae;
pub(crate) mod vae_encoder;

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

/// v0.32's per-branch retention bound. It is still the rule wherever the
/// card's total memory cannot be measured (Metal, CPU, an unqueryable CUDA
/// device), so those renders keep exactly the cache behaviour they had.
pub(crate) const LEGACY_PREFIX_CACHE_TOKENS: usize = 512;

/// On a card whose total memory is known, the retained prefix of every
/// branch together may take at most `total / PREFIX_CACHE_DEVICE_SHARE`.
pub(crate) const PREFIX_CACHE_DEVICE_SHARE: u64 = 4;

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
    /// Retain while the prefix fits the card's budget (the default).
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

/// What one conditioning branch does with its prefix K/V.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PrefixCacheDecision {
    /// Prefill once, then run only the target block on later steps.
    Retain,
    /// Recompute the whole joint sequence every step.
    Recompute,
}

/// The one decision for prefix K/V retention.
pub(crate) struct PrefixCachePolicy;

impl PrefixCachePolicy {
    /// Decide for every branch of one request.
    ///
    /// `branch_prefix_tokens` holds each branch's prefix length (text plus
    /// condition-image tokens). With a known device total, `Auto` retains
    /// EVERY branch iff their summed cache fits `total /
    /// PREFIX_CACHE_DEVICE_SHARE` — the card's TOTAL, never its free bytes at
    /// this instant, so the server's plan and the engine agree on the same
    /// request (the `flux2_cfg_batching` rule). Without one, each branch keeps
    /// v0.32's 512-token rule.
    pub(crate) fn resolve(
        branch_prefix_tokens: &[usize],
        batch: usize,
        dtype_bytes: usize,
        device_total_bytes: Option<u64>,
        mode: PrefixCacheMode,
    ) -> Vec<PrefixCacheDecision> {
        let all = |decision| vec![decision; branch_prefix_tokens.len()];
        match mode {
            PrefixCacheMode::On => all(PrefixCacheDecision::Retain),
            PrefixCacheMode::Off => all(PrefixCacheDecision::Recompute),
            PrefixCacheMode::Auto => match device_total_bytes {
                Some(total) => {
                    let bytes: u64 = branch_prefix_tokens
                        .iter()
                        .map(|&tokens| prefix_cache_bytes(tokens, batch, dtype_bytes))
                        .sum();
                    all(if bytes <= total / PREFIX_CACHE_DEVICE_SHARE {
                        PrefixCacheDecision::Retain
                    } else {
                        PrefixCacheDecision::Recompute
                    })
                }
                None => branch_prefix_tokens
                    .iter()
                    .map(|&tokens| {
                        if tokens <= LEGACY_PREFIX_CACHE_TOKENS {
                            PrefixCacheDecision::Retain
                        } else {
                            PrefixCacheDecision::Recompute
                        }
                    })
                    .collect(),
            },
        }
    }

    /// Resolve for a request running on `device`, reading the override and
    /// the device's total memory.
    pub(crate) fn resolve_for_device(
        branch_prefix_tokens: &[usize],
        batch: usize,
        dtype: DType,
        device: &Device,
    ) -> Vec<PrefixCacheDecision> {
        let total = match device.location() {
            candle_core::DeviceLocation::Cuda { gpu_id } => crate::device::total_vram_bytes(gpu_id),
            _ => None,
        };
        Self::resolve(
            branch_prefix_tokens,
            batch,
            dtype.size_in_bytes(),
            total,
            parse_prefix_cache_mode(
                crate::runtime_env::value("MOLD_QWEN_IMAGE21_KV_CACHE").as_deref(),
            ),
        )
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
fn system_message_prefix() -> String {
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
    fn prefix_cache_policy_budgets_every_branch_against_total_memory() {
        use PrefixCacheDecision::{Recompute, Retain};
        const GIB: u64 = 1 << 30;
        // 512 KiB per BF16 prefix token per branch.
        assert_eq!(prefix_cache_bytes(1, 1, 2), 512 * 1024);
        // One 1024² reference + prompt ≈ 4.2k prefix tokens ≈ 2.1 GiB/branch:
        // both CFG branches fit a quarter of a 46 GB L40S.
        let l40s = Some(46 * GIB);
        assert_eq!(
            PrefixCachePolicy::resolve(&[4200, 4150], 1, 2, l40s, PrefixCacheMode::Auto),
            vec![Retain, Retain]
        );
        // Ten references (~10.3k tokens) with CFG is 10.1 GiB, inside the
        // 11.5 GiB share; 12k tokens per branch is over it. The decision is
        // shared so both branches take the same path.
        assert_eq!(
            PrefixCachePolicy::resolve(&[10_300, 10_250], 1, 2, l40s, PrefixCacheMode::Auto),
            vec![Retain, Retain]
        );
        assert_eq!(
            PrefixCachePolicy::resolve(&[12_000, 12_000], 1, 2, l40s, PrefixCacheMode::Auto),
            vec![Recompute, Recompute]
        );
        // A 24 GB card: one reference without CFG fits, with CFG it does not
        // at F32 activations.
        let rtx = Some(24 * GIB);
        assert_eq!(
            PrefixCachePolicy::resolve(&[4200], 1, 2, rtx, PrefixCacheMode::Auto),
            vec![Retain]
        );
        assert_eq!(
            PrefixCachePolicy::resolve(&[4200, 4200], 1, 4, rtx, PrefixCacheMode::Auto),
            vec![Recompute, Recompute]
        );
        // Every text-to-image prompt v0.32 cached still caches on any card of
        // 4 GiB or more.
        assert_eq!(
            PrefixCachePolicy::resolve(&[512, 512], 1, 4, Some(4 * GIB), PrefixCacheMode::Auto),
            vec![Retain, Retain]
        );
        // Without a measurable total, each branch keeps v0.32's own rule.
        assert_eq!(
            PrefixCachePolicy::resolve(&[512, 513], 1, 4, None, PrefixCacheMode::Auto),
            vec![Retain, Recompute]
        );
        // Overrides win in both directions.
        assert_eq!(
            PrefixCachePolicy::resolve(&[10_300], 1, 2, None, PrefixCacheMode::On),
            vec![Retain]
        );
        assert_eq!(
            PrefixCachePolicy::resolve(&[3], 1, 2, l40s, PrefixCacheMode::Off),
            vec![Recompute]
        );
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
