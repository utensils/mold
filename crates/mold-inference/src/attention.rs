//! Pluggable attention backend for FLUX-family transformers.
//!
//! Two implementations live behind a single dispatch helper:
//!
//! * `Math`  — the historical hand-rolled `q.matmul(k.t()) * scale → softmax → matmul(v)`.
//!   Materialises the full `B·H·N·N` attention matrix; fine on CPU/Metal,
//!   the dominant VRAM cost on CUDA at FLUX 1024^2.
//! * `Flash` — `candle-flash-attn` (flash-attention v2). Only available with
//!   `--features cuda,flash-attn` and a CUDA tensor in fp16/bf16. Falls through
//!   to `Math` for an ineligible tensor, or with a one-shot warning when the
//!   binary was not compiled with FlashAttention support.
//!
//! Selection is env-driven via `MOLD_ATTN={flash,math}` and cached in a
//! `OnceLock` so we don't re-read the environment on every block.
//!
//! ComfyUI does the same thing in `ldm/modules/attention.py:495-540`.

use candle_core::{DType, Device, Result, Tensor, D};
use std::sync::OnceLock;

/// Selectable attention backend. See module docs for semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionBackend {
    /// Hand-rolled QK^T softmax V — current default everywhere.
    Math,
    /// `candle-flash-attn` (flash-attention v2). CUDA + fp16/bf16 only.
    Flash,
}

/// Process-frozen override for math-attention query chunking. `Auto` retains
/// the CUDA/sequence-length heuristic, `Off` forbids chunking, and `Size`
/// fixes the maximum query rows per chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionChunkPolicy {
    Auto,
    Off,
    Size(usize),
}

/// Tracks whether we've already emitted the "flash requested but unavailable"
/// warning for this process. The dispatcher prints it at most once so a
/// 50-step diffusion run doesn't spam the operator log with the same line.
static FLASH_FALLBACK_WARNED: OnceLock<()> = OnceLock::new();

impl AttentionBackend {
    /// Resolve the backend once, cache forever.
    ///
    /// Precedence:
    /// 1. `MOLD_ATTN` env (`flash` / `math`, case-insensitive).
    /// 2. `flash-attn` cargo feature → default `Flash`.
    /// 3. Otherwise → `Math`.
    pub fn resolve() -> AttentionBackend {
        static CACHED: OnceLock<AttentionBackend> = OnceLock::new();
        *CACHED.get_or_init(|| {
            let backend = parse_backend_env(crate::runtime_env::value("MOLD_ATTN").as_deref());
            tracing::info!(backend = ?backend, "attention backend selected");
            backend
        })
    }
}

/// Pure function used by `resolve()` and unit tests so we can exercise the env
/// parser without poisoning the global `OnceLock`.
fn parse_backend_env(raw: Option<&str>) -> AttentionBackend {
    if let Some(value) = raw {
        match value.trim().to_ascii_lowercase().as_str() {
            "flash" => return AttentionBackend::Flash,
            "math" => return AttentionBackend::Math,
            // `sdpa` was removed in the Tier 1 review followup — it was a
            // no-op alias for `math` with no signal to the user. Anyone
            // still setting it gets the math path with a one-time warning.
            "sdpa" => {
                tracing::warn!(
                    "MOLD_ATTN=sdpa was removed (it was a no-op alias for math); using math"
                );
                return AttentionBackend::Math;
            }
            other if !other.is_empty() => {
                tracing::warn!(
                    "MOLD_ATTN={other} is not one of flash/math; falling back to default"
                );
            }
            _ => {}
        }
    }
    default_backend()
}

pub fn resolved_chunk_policy() -> AttentionChunkPolicy {
    static CACHED: OnceLock<AttentionChunkPolicy> = OnceLock::new();
    *CACHED.get_or_init(|| {
        let raw = crate::runtime_env::value("MOLD_ATTN_CHUNK");
        match raw.as_deref().map(str::trim) {
            Some("0") => AttentionChunkPolicy::Off,
            Some(value) if value.eq_ignore_ascii_case("off") => AttentionChunkPolicy::Off,
            Some(value) => match value.parse::<usize>() {
                Ok(size) if size > 0 => AttentionChunkPolicy::Size(size),
                _ => {
                    tracing::warn!(
                        value,
                        "MOLD_ATTN_CHUNK must be a positive integer, 0, or off; using default"
                    );
                    AttentionChunkPolicy::Auto
                }
            },
            None => AttentionChunkPolicy::Auto,
        }
    })
}

/// Emit the "flash requested but not compiled" warning at most once per
/// process. Returns `true` if this call was the one that fired the warning.
/// Exposed at `pub(crate)` so the unit tests can assert the OnceLock state.
pub(crate) fn warn_flash_fallback_once() -> bool {
    let mut fired = false;
    FLASH_FALLBACK_WARNED.get_or_init(|| {
        tracing::warn!(
            "attention backend 'flash' requested but FlashAttention is not compiled \
             (build with --features cuda,flash-attn); \
             falling back to math"
        );
        fired = true;
    });
    fired
}

/// Whether the flash-fallback warning has fired this process. Test helper.
#[cfg(test)]
pub(crate) fn flash_fallback_warned() -> bool {
    FLASH_FALLBACK_WARNED.get().is_some()
}

#[cfg(feature = "flash-attn")]
fn default_backend() -> AttentionBackend {
    AttentionBackend::Flash
}

#[cfg(not(feature = "flash-attn"))]
fn default_backend() -> AttentionBackend {
    AttentionBackend::Math
}

/// Scaled dot-product attention.
///
/// Input layout: `[batch, n_heads, seq, head_dim]` (BHND), as produced by FLUX's
/// `qkv` projection. Output has the same shape.
///
/// `scale` is the explicit `1 / sqrt(head_dim)` factor — passing it in (rather
/// than recomputing) lets callers reuse a value they already have, and keeps
/// the test surface deterministic.
pub fn attention(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32) -> Result<Tensor> {
    match AttentionBackend::resolve() {
        AttentionBackend::Flash => flash_attention(q, k, v, scale),
        AttentionBackend::Math => math_attention(q, k, v, scale),
    }
}

/// Convenience: derive `scale` from `head_dim` and dispatch.
pub fn attention_default_scale(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let head_dim = q.dim(D::Minus1)?;
    let scale = 1.0 / (head_dim as f64).sqrt();
    attention(q, k, v, scale as f32)
}

/// Tracks whether we've already logged chunked math attention selection.
static CHUNKED_MATH_LOGGED: OnceLock<()> = OnceLock::new();

/// Hand-rolled SDP — the historical FLUX path. Flattens batch+heads into a
/// single leading dim to avoid the 4D `matmul` quirks on some backends.
pub fn math_attention(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32) -> Result<Tensor> {
    math_attention_impl(q, k, v, scale, math_attention_chunk_size(q))
}

fn math_attention_impl(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    chunk_size: Option<usize>,
) -> Result<Tensor> {
    let mut batch_dims = q.dims().to_vec();
    batch_dims.pop();
    batch_dims.pop();
    let q3 = q.flatten_to(batch_dims.len() - 1)?;
    let k3 = k.flatten_to(batch_dims.len() - 1)?;
    let v3 = v.flatten_to(batch_dims.len() - 1)?;
    let attn = if let Some(chunk_size) = chunk_size {
        math_attention_chunked_flat(&q3, &k3, &v3, scale, chunk_size)?
    } else {
        let attn_weights = (q3.matmul(&k3.t()?)? * f64::from(scale))?;
        candle_nn::ops::softmax_last_dim(&attn_weights)?.matmul(&v3)?
    };
    batch_dims.push(attn.dim(D::Minus2)?);
    batch_dims.push(attn.dim(D::Minus1)?);
    attn.reshape(batch_dims)
}

fn math_attention_chunk_size(q: &Tensor) -> Option<usize> {
    let q_len = q.dim(D::Minus2).ok()?;
    match resolved_chunk_policy() {
        AttentionChunkPolicy::Off => return None,
        AttentionChunkPolicy::Size(size) => return (size < q_len).then_some(size),
        AttentionChunkPolicy::Auto => {}
    }

    if matches!(q.device(), Device::Cuda(_)) && q_len > 1024 {
        Some(512)
    } else {
        None
    }
}

fn math_attention_chunked_flat(
    q3: &Tensor,
    k3: &Tensor,
    v3: &Tensor,
    scale: f32,
    chunk_size: usize,
) -> Result<Tensor> {
    let q_len = q3.dim(1)?;
    let k_t = k3.t()?;
    let mut chunks = Vec::with_capacity(q_len.div_ceil(chunk_size));
    let mut start = 0;
    while start < q_len {
        let len = (q_len - start).min(chunk_size);
        let q_chunk = q3.narrow(1, start, len)?;
        let attn_weights = (q_chunk.matmul(&k_t)? * f64::from(scale))?;
        let attn = candle_nn::ops::softmax_last_dim(&attn_weights)?.matmul(v3)?;
        chunks.push(attn);
        start += len;
    }

    CHUNKED_MATH_LOGGED.get_or_init(|| {
        tracing::info!(
            chunk_size,
            q_len,
            "using chunked math attention to reduce peak VRAM"
        );
    });

    let refs: Vec<&Tensor> = chunks.iter().collect();
    Tensor::cat(&refs, 1)
}

/// Flash-attention v2 path.
///
/// When the `flash-attn` feature is on and the tensors are CUDA + fp16/bf16,
/// this calls `candle_flash_attn::flash_attn`. The workspace patch retains
/// Candle's upstream package names, so the FFI crate and Mold share the same
/// `Tensor` type without an out-of-band build cfg.
pub fn flash_attention(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32) -> Result<Tensor> {
    if !flash_is_eligible(q) {
        // CPU/Metal tensors or wrong dtype — fall back without the
        // FFI-gate warning. Hitting the math path on these devices is the
        // expected behavior, not a misconfiguration.
        return math_attention(q, k, v, scale);
    }

    #[cfg(feature = "flash-attn")]
    {
        // FLUX QKV are `[B, H, N, D]`. candle-flash-attn wants `[B, N, H, D]`.
        let q_t = q.transpose(1, 2)?.contiguous()?;
        let k_t = k.transpose(1, 2)?.contiguous()?;
        let v_t = v.transpose(1, 2)?.contiguous()?;
        let out = candle_flash_attn::flash_attn(&q_t, &k_t, &v_t, scale, false)?;
        // Output is `[B, N, H, D]`; restore `[B, H, N, D]` for callers.
        return out.transpose(1, 2)?.contiguous();
    }

    // Tensor was eligible (CUDA + fp16/bf16), but this binary omitted the
    // optional kernel. Warn once before falling through to math.
    #[cfg(not(feature = "flash-attn"))]
    {
        warn_flash_fallback_once();
    }
    math_attention(q, k, v, scale)
}

/// Flash-attention 2 requires CUDA tensors in fp16 or bf16.
fn flash_is_eligible(q: &Tensor) -> bool {
    matches!(q.device(), Device::Cuda(_)) && matches!(q.dtype(), DType::F16 | DType::BF16)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn cpu() -> Device {
        Device::Cpu
    }

    /// Brute-force reference: explicit loops over (b, h, q, k) with f32.
    fn reference_attention(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32) -> Tensor {
        // Always promote to f32 for the reference to keep tolerances tight.
        let q = q.to_dtype(DType::F32).unwrap();
        let k = k.to_dtype(DType::F32).unwrap();
        let v = v.to_dtype(DType::F32).unwrap();
        let weights = q.matmul(&k.t().unwrap()).unwrap();
        let weights = (weights * scale as f64).unwrap();
        let weights = candle_nn::ops::softmax_last_dim(&weights).unwrap();
        weights.matmul(&v).unwrap()
    }

    fn rand_qkv(shape: (usize, usize, usize, usize)) -> (Tensor, Tensor, Tensor) {
        let dev = cpu();
        let q = Tensor::randn(0.0_f32, 1.0_f32, shape, &dev).unwrap();
        let k = Tensor::randn(0.0_f32, 1.0_f32, shape, &dev).unwrap();
        let v = Tensor::randn(0.0_f32, 1.0_f32, shape, &dev).unwrap();
        (q, k, v)
    }

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
        let diff = (a - b).unwrap().abs().unwrap();
        diff.flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    }

    #[test]
    fn test_math_attention_matches_reference() {
        // Toy shape: 2 batch, 4 heads, 16 seq, 32 head_dim.
        let (q, k, v) = rand_qkv((2, 4, 16, 32));
        let scale = 1.0 / (32f32).sqrt();
        let got = math_attention(&q, &k, &v, scale).unwrap();
        let want = reference_attention(&q, &k, &v, scale);
        assert_eq!(got.dims(), &[2, 4, 16, 32]);
        assert!(
            max_abs_diff(&got, &want) < 1e-5,
            "math attention diverged from reference"
        );
    }

    #[test]
    fn test_chunked_math_attention_matches_full_math() {
        let (q, k, v) = rand_qkv((1, 3, 17, 16));
        let scale = 1.0 / (16f32).sqrt();
        let full = math_attention_impl(&q, &k, &v, scale, None).unwrap();
        let chunked = math_attention_impl(&q, &k, &v, scale, Some(5)).unwrap();

        assert_eq!(chunked.dims(), full.dims());
        assert!(
            max_abs_diff(&chunked, &full) < 1e-5,
            "chunked math attention diverged from full math"
        );
    }

    #[test]
    fn test_flash_falls_back_on_cpu() {
        // CPU tensors are not flash-eligible, so flash_attention must fall
        // through to math regardless of the cargo feature. This path does
        // NOT fire the one-shot warning (CPU is the expected fallback
        // surface, not a misconfiguration).
        let (q, k, v) = rand_qkv((1, 2, 8, 16));
        let scale = 1.0 / (16f32).sqrt();
        let math = math_attention(&q, &k, &v, scale).unwrap();
        let flash = flash_attention(&q, &k, &v, scale).unwrap();
        assert!(max_abs_diff(&math, &flash) < 1e-5);
    }

    #[test]
    fn test_attention_default_scale() {
        // Sanity: helper computes 1/sqrt(d) and matches an explicit pass.
        let (q, k, v) = rand_qkv((1, 2, 4, 8));
        let scale = 1.0 / (8f32).sqrt();
        let explicit = math_attention(&q, &k, &v, scale).unwrap();
        let implicit = attention_default_scale(&q, &k, &v).unwrap();
        assert!(max_abs_diff(&explicit, &implicit) < 1e-5);
    }

    #[test]
    fn test_resolve_backend_from_env() {
        // OnceLock-free parser: covers the env contract exhaustively.
        assert_eq!(parse_backend_env(Some("flash")), AttentionBackend::Flash);
        assert_eq!(parse_backend_env(Some("FLASH")), AttentionBackend::Flash);
        assert_eq!(parse_backend_env(Some("math")), AttentionBackend::Math);
        // Unknown values warn and fall back.
        assert_eq!(parse_backend_env(Some("xformers")), default_backend());
        assert_eq!(parse_backend_env(Some("")), default_backend());
        assert_eq!(parse_backend_env(None), default_backend());
    }

    /// `Sdpa` is gone from the public enum (T1.5 review followup): it was a
    /// no-op alias for `Math` whose presence misled users into thinking
    /// they'd selected a real second backend. The parser now warns and
    /// returns `Math` so the same env stays functional, but no
    /// `AttentionBackend::Sdpa` variant exists for callers to match on.
    #[test]
    fn resolve_returns_only_known_backends() {
        assert_eq!(parse_backend_env(Some("sdpa")), AttentionBackend::Math);
        assert_eq!(parse_backend_env(Some("SDPA")), AttentionBackend::Math);
        assert_eq!(parse_backend_env(Some(" sdpa ")), AttentionBackend::Math);
        // Spot-check that the supported set is the documented two:
        for value in ["flash", "math"] {
            let backend = parse_backend_env(Some(value));
            assert!(matches!(
                backend,
                AttentionBackend::Flash | AttentionBackend::Math
            ));
        }
    }

    /// When `MOLD_ATTN=flash` is requested but the kernel was not compiled, the
    /// dispatcher must fire a `tracing::warn!` exactly once per process —
    /// not on every block of every step. We assert the OnceLock state
    /// directly because tracing-test introduces a heavy dep for what is
    /// fundamentally a single-bit observation.
    ///
    /// The CPU test build does not compile the optional CUDA kernel; asserting
    /// the helper directly keeps the one-shot contract independent of hardware.
    #[test]
    fn flash_fallback_warns_once() {
        // First call fires the warning; subsequent calls are no-ops.
        let first = warn_flash_fallback_once();
        let second = warn_flash_fallback_once();
        let third = warn_flash_fallback_once();
        // Either the first call we ever made in this process fired (and
        // subsequent calls did not), or some earlier test in the same
        // process already fired it — in which case none of our calls
        // should have fired. Both are valid outcomes.
        assert!(
            !(second || third),
            "warn_flash_fallback_once must not re-fire after the first call"
        );
        if first {
            // We were the first call in this process — verify the latch
            // is now sticky.
            assert!(
                flash_fallback_warned(),
                "OnceLock state must reflect that the warning fired"
            );
        }
        // Either way, the OnceLock must now be set.
        assert!(
            flash_fallback_warned(),
            "warn_flash_fallback_once must always leave the latch set"
        );
    }

    #[test]
    #[cfg(not(feature = "flash-attn"))]
    fn test_resolve_default_without_feature() {
        assert_eq!(default_backend(), AttentionBackend::Math);
        assert_eq!(parse_backend_env(None), AttentionBackend::Math);
    }

    #[test]
    #[cfg(feature = "flash-attn")]
    fn test_resolve_default_with_feature() {
        assert_eq!(default_backend(), AttentionBackend::Flash);
    }
}
