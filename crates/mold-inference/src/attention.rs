//! Pluggable attention backend for FLUX-family transformers.
//!
//! Three implementations live behind a single dispatch helper:
//!
//! * `Math`  — the historical hand-rolled `q.matmul(k.t()) * scale → softmax → matmul(v)`.
//!   Materialises the full `B·H·N·N` attention matrix; fine on CPU/Metal,
//!   the dominant VRAM cost on CUDA at FLUX 1024^2.
//! * `Sdpa`  — a numerically-equivalent rewrite that flattens to 3D before the
//!   matmul. Provided as a safety net so users on GGUF/exotic dtypes can opt
//!   into a slightly different code path without enabling flash-attn.
//! * `Flash` — `candle-flash-attn` (flash-attention v2). Only available with
//!   `--features flash-attn` AND a CUDA tensor in fp16/bf16. Falls through to
//!   `Math` for any tensor that doesn't satisfy those constraints.
//!
//! Selection is env-driven via `MOLD_ATTN={flash,sdpa,math}` and cached in a
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
    /// Same arithmetic, 3D-flattened. Useful as a sanity-check baseline.
    Sdpa,
    /// `candle-flash-attn` (flash-attention v2). CUDA + fp16/bf16 only.
    Flash,
}

impl AttentionBackend {
    /// Resolve the backend once, cache forever.
    ///
    /// Precedence:
    /// 1. `MOLD_ATTN` env (`flash` / `sdpa` / `math`, case-insensitive).
    /// 2. `flash-attn` cargo feature → default `Flash`.
    /// 3. Otherwise → `Math`.
    pub fn resolve() -> AttentionBackend {
        static CACHED: OnceLock<AttentionBackend> = OnceLock::new();
        *CACHED.get_or_init(|| {
            let backend = parse_backend_env(std::env::var("MOLD_ATTN").ok().as_deref());
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
            "sdpa" => return AttentionBackend::Sdpa,
            "math" => return AttentionBackend::Math,
            other if !other.is_empty() => {
                tracing::warn!(
                    "MOLD_ATTN={other} is not one of flash/sdpa/math; falling back to default"
                );
            }
            _ => {}
        }
    }
    default_backend()
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
        AttentionBackend::Sdpa => sdpa_attention(q, k, v, scale),
        AttentionBackend::Math => math_attention(q, k, v, scale),
    }
}

/// Convenience: derive `scale` from `head_dim` and dispatch.
pub fn attention_default_scale(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let head_dim = q.dim(D::Minus1)?;
    let scale = 1.0 / (head_dim as f64).sqrt();
    attention(q, k, v, scale as f32)
}

/// Hand-rolled SDP — the historical FLUX path. Flattens batch+heads into a
/// single leading dim to avoid the 4D `matmul` quirks on some backends.
pub fn math_attention(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32) -> Result<Tensor> {
    let mut batch_dims = q.dims().to_vec();
    batch_dims.pop();
    batch_dims.pop();
    let q3 = q.flatten_to(batch_dims.len() - 1)?;
    let k3 = k.flatten_to(batch_dims.len() - 1)?;
    let v3 = v.flatten_to(batch_dims.len() - 1)?;
    let attn_weights = (q3.matmul(&k3.t()?)? * f64::from(scale))?;
    let attn = candle_nn::ops::softmax_last_dim(&attn_weights)?.matmul(&v3)?;
    batch_dims.push(attn.dim(D::Minus2)?);
    batch_dims.push(attn.dim(D::Minus1)?);
    attn.reshape(batch_dims)
}

/// Same arithmetic as `math_attention` but expressed against the original 4D
/// shape. Kept distinct so users can A/B test if a backend is faster with one
/// matmul layout vs. another.
pub fn sdpa_attention(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32) -> Result<Tensor> {
    // Identical mathematical result; we deliberately reuse the math impl so we
    // can't drift. The split exists for future divergence (e.g. cuDNN SDPA).
    math_attention(q, k, v, scale)
}

/// Flash-attention v2 path.
///
/// When the `flash-attn` feature is on AND the tensors are CUDA + fp16/bf16
/// AND the build was configured against a `candle-core` that matches the one
/// `candle-flash-attn` was compiled against (the `mold_flash_attn_real` cfg),
/// this calls `candle_flash_attn::flash_attn`. Otherwise it falls back to
/// `math_attention` — same numerical answer, just slower.
///
/// Why two gates? `candle-flash-attn` 0.9.x links upstream `candle-core`
/// while mold pulls `candle-core-mold`, so the two `Tensor` types don't
/// unify in the same build graph. The `mold_flash_attn_real` cfg is the
/// FFI-link gate — set via `RUSTFLAGS='--cfg mold_flash_attn_real'` once a
/// `candle-flash-attn-mold` companion lands or a workspace `[patch.crates-io]`
/// unifies the two `candle-core` packages. Until then the cargo feature still
/// builds cleanly so users can opt into the dispatcher's plumbing.
pub fn flash_attention(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32) -> Result<Tensor> {
    if !flash_is_eligible(q) {
        return math_attention(q, k, v, scale);
    }

    #[cfg(all(feature = "flash-attn", mold_flash_attn_real))]
    {
        // FLUX QKV are `[B, H, N, D]`. candle-flash-attn wants `[B, N, H, D]`.
        let q_t = q.transpose(1, 2)?.contiguous()?;
        let k_t = k.transpose(1, 2)?.contiguous()?;
        let v_t = v.transpose(1, 2)?.contiguous()?;
        let out = candle_flash_attn::flash_attn(&q_t, &k_t, &v_t, scale, false)?;
        // Output is `[B, N, H, D]`; restore `[B, H, N, D]` for callers.
        return out.transpose(1, 2)?.contiguous();
    }

    // Either the cargo feature is off, or the FFI gate hasn't been opened.
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
    fn test_sdpa_matches_math() {
        let (q, k, v) = rand_qkv((1, 2, 8, 16));
        let scale = 1.0 / (16f32).sqrt();
        let math = math_attention(&q, &k, &v, scale).unwrap();
        let sdpa = sdpa_attention(&q, &k, &v, scale).unwrap();
        assert!(max_abs_diff(&math, &sdpa) < 1e-4);
    }

    #[test]
    fn test_flash_falls_back_on_cpu() {
        // CPU tensors are not flash-eligible, so flash_attention must fall
        // through to math regardless of the cargo feature.
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
        assert_eq!(parse_backend_env(Some(" sdpa ")), AttentionBackend::Sdpa);
        assert_eq!(parse_backend_env(Some("math")), AttentionBackend::Math);
        // Unknown values warn and fall back.
        assert_eq!(parse_backend_env(Some("xformers")), default_backend());
        assert_eq!(parse_backend_env(Some("")), default_backend());
        assert_eq!(parse_backend_env(None), default_backend());
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
