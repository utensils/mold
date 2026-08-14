//! Pluggable attention backend for FLUX-family transformers.
//!
//! Two implementations live behind a single dispatch helper:
//!
//! * `Math`  — the historical hand-rolled `q.matmul(k.t()) * scale → softmax → matmul(v)`.
//!   Materialises the full `B·H·N·N` attention matrix; fine on CPU/Metal,
//!   the dominant VRAM cost on CUDA at FLUX 1024^2.
//! * `Flash` — `candle-flash-attn` (flash-attention v2). Only available with
//!   `--features cuda,flash-attn` and a CUDA tensor in fp16/bf16. Falls through
//!   to `Math` silently for an ineligible tensor, or with a one-shot warning
//!   when the binary was not compiled with FlashAttention support (that
//!   warning only exists in a build without the feature).
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
///
/// Only exists in a build without `flash-attn`: once the kernel is compiled
/// in, "requested but not compiled" is not a reachable state.
#[cfg(not(feature = "flash-attn"))]
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
///
/// Absent under `flash-attn`; see [`FLASH_FALLBACK_WARNED`].
#[cfg(not(feature = "flash-attn"))]
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
#[cfg(all(test, not(feature = "flash-attn")))]
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

    // Metal chunks for the same reason CUDA does, but harder: there is no
    // flash path there, so an unchunked math attention materializes the full
    // `[b·h, N, N]` score matrix — 9.5 GB at Wan 1.3B's smallest production
    // shape, 71 GB at TI2V-5B 720p — as a single buffer past every Mac's
    // `maxBufferLength`. CPU stays unchunked: its allocator handles the
    // shape, and the chunk loop only adds overhead there.
    if matches!(q.device(), Device::Cuda(_) | Device::Metal(_)) && q_len > 1024 {
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
/// An ineligible tensor (CPU/Metal, or a dtype other than fp16/bf16) always
/// takes the math path silently — that is the expected shape of a CPU or Metal
/// run, not a misconfiguration. An eligible tensor goes to the config-specific
/// [`flash_attention_eligible`].
pub fn flash_attention(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32) -> Result<Tensor> {
    if !flash_is_eligible(q) {
        return math_attention(q, k, v, scale);
    }
    flash_attention_eligible(q, k, v, scale)
}

/// Eligible tensor and the kernel is compiled in: run FlashAttention v2.
///
/// The workspace patch retains Candle's upstream package names, so the FFI
/// crate and Mold share the same `Tensor` type without an out-of-band build
/// cfg.
#[cfg(feature = "flash-attn")]
fn flash_attention_eligible(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32) -> Result<Tensor> {
    // FLUX QKV are `[B, H, N, D]`. candle-flash-attn wants `[B, N, H, D]`.
    let q_t = to_flash_layout(q)?;
    let k_t = to_flash_layout(k)?;
    let v_t = to_flash_layout(v)?;
    let out = candle_flash_attn::flash_attn(&q_t, &k_t, &v_t, scale, false)?;
    // Output is `[B, N, H, D]`; restore `[B, H, N, D]` for callers, who go on
    // to `reshape` it and so need it contiguous.
    out.transpose(1, 2)?.contiguous()
}

/// Transpose `[B, H, N, D]` to the `[B, N, H, D]` candle-flash-attn expects.
///
/// The kernel reads the outer strides straight off the layout and only
/// requires a unit stride on the last axis, so the transposed *view* of a
/// contiguous input is already an acceptable argument. Copying it anyway cost
/// a full `B·H·N·D` buffer per tensor per attention call. The `contiguous`
/// fallback still covers an input that arrives with a non-unit last stride.
#[cfg(feature = "flash-attn")]
fn to_flash_layout(t: &Tensor) -> Result<Tensor> {
    let t = t.transpose(1, 2)?;
    if t.stride().last() == Some(&1) {
        Ok(t)
    } else {
        t.contiguous()
    }
}

/// Eligible tensor, but this binary omitted the optional kernel: warn once,
/// then fall through to math.
#[cfg(not(feature = "flash-attn"))]
fn flash_attention_eligible(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32) -> Result<Tensor> {
    warn_flash_fallback_once();
    math_attention(q, k, v, scale)
}

/// Flash-attention 2 requires CUDA tensors in fp16 or bf16, at a head dim the
/// kernel accepts.
fn flash_is_eligible(q: &Tensor) -> bool {
    matches!(q.device(), Device::Cuda(_))
        && matches!(q.dtype(), DType::F16 | DType::BF16)
        && q.dim(D::Minus1).is_ok_and(flash_supports_head_dim)
}

/// Head dims `candle_flash_attn::flash_attn` accepts: a multiple of 8, at most
/// 512. Anything else is an `Err` from the FFI wrapper, raised before it
/// touches the tensors.
///
/// FLUX, Flux.2 and LTX-2 are all pinned at 128 by their configs, but Z-Image
/// reads `head_dim` off the checkpoint at runtime and Mold will load an
/// arbitrary `cv:`/`hf:` Z-Image export — so an unsupported value is reachable
/// by a user, and it must degrade to the math path rather than fail the
/// generation. Sizes in range but off a compiled bucket are fine: the kernel
/// rounds up and predicates the loads, which is numerically exact.
fn flash_supports_head_dim(head_dim: usize) -> bool {
    head_dim > 0 && head_dim.is_multiple_of(8) && head_dim <= 512
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

    /// CPU never auto-chunks: its allocator handles the score matrix and the
    /// chunk loop only adds overhead there.
    #[test]
    fn cpu_math_attention_is_not_auto_chunked() {
        let q = Tensor::zeros((1, 1, 2048, 8), DType::F32, &cpu()).unwrap();
        assert_eq!(math_attention_chunk_size(&q), None);
    }

    /// Metal must auto-chunk above the same threshold CUDA does: with no
    /// flash path, an unchunked math attention materializes the full score
    /// matrix as one buffer, which exceeds `maxBufferLength` at every real
    /// Wan shape.
    #[cfg(feature = "metal")]
    #[test]
    fn metal_math_attention_auto_chunks_long_sequences() {
        let metal = Device::new_metal(0).unwrap();
        let long = Tensor::zeros((1, 1025, 8), DType::F32, &metal).unwrap();
        assert_eq!(math_attention_chunk_size(&long), Some(512));
        let short = Tensor::zeros((1, 1024, 8), DType::F32, &metal).unwrap();
        assert_eq!(math_attention_chunk_size(&short), None);
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
        // The load-bearing half of "CPU must not warn" is that ineligibility
        // routes to math *before* the warning site. Assert the predicate
        // rather than the process-global latch, which `flash_fallback_warns_once`
        // sets from another test thread in nondeterministic order.
        assert!(
            !flash_is_eligible(&q),
            "CPU tensors must never be flash-eligible"
        );
    }

    /// `to_flash_layout` hands the kernel a transposed *view* instead of the
    /// copy the code used to make. That is only sound because the FFI wrapper
    /// reads the outer strides off the layout and requires nothing but a unit
    /// stride on the last axis, so this pins the two against each other.
    ///
    /// Requires a CUDA device and so does not execute in CI, which has no GPU
    /// runner; it is a developer-machine guard for a change whose failure mode
    /// is wrong pixels rather than an error. Verified on an RTX 4090.
    #[test]
    #[cfg(feature = "flash-attn")]
    fn flash_layout_view_matches_contiguous_copy() {
        let Ok(device) = Device::new_cuda(0) else {
            return;
        };
        let (b, h, n, d) = (1, 4, 256, 64);
        let mk = || {
            Tensor::randn(0f32, 1.0, (b, h, n, d), &device)
                .and_then(|t| t.to_dtype(DType::BF16))
                .expect("cuda tensor")
        };
        let (q, k, v) = (mk(), mk(), mk());
        let scale = 1.0 / (d as f32).sqrt();

        let viewed = candle_flash_attn::flash_attn(
            &to_flash_layout(&q).unwrap(),
            &to_flash_layout(&k).unwrap(),
            &to_flash_layout(&v).unwrap(),
            scale,
            false,
        )
        .unwrap();
        let copied = candle_flash_attn::flash_attn(
            &q.transpose(1, 2).unwrap().contiguous().unwrap(),
            &k.transpose(1, 2).unwrap().contiguous().unwrap(),
            &v.transpose(1, 2).unwrap().contiguous().unwrap(),
            scale,
            false,
        )
        .unwrap();

        assert_eq!(viewed.dims(), copied.dims());
        assert_eq!(
            max_abs_diff(
                &viewed.to_dtype(DType::F32).unwrap(),
                &copied.to_dtype(DType::F32).unwrap()
            ),
            0.0,
            "a transposed view must give the kernel exactly what the copy did"
        );
    }

    /// FA2 rejects a head dim that is not a multiple of 8, or above 512.
    /// Z-Image takes `head_dim` from whatever checkpoint the user loaded, so
    /// those values are reachable and must route to math instead of failing
    /// the generation.
    #[test]
    fn test_flash_head_dim_support_matches_kernel_contract() {
        for supported in [8, 32, 64, 96, 128, 160, 192, 224, 256, 512] {
            assert!(
                flash_supports_head_dim(supported),
                "head_dim {supported} is accepted by candle-flash-attn"
            );
        }
        // In range and a multiple of 8 but not a compiled bucket: the kernel
        // rounds up to the next bucket and predicates the loads.
        assert!(flash_supports_head_dim(40));
        assert!(flash_supports_head_dim(200));

        for unsupported in [0, 1, 12, 100, 520, 1024] {
            assert!(
                !flash_supports_head_dim(unsupported),
                "head_dim {unsupported} must fall back to math"
            );
        }
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
    ///
    /// Only meaningful without `flash-attn` — with the kernel compiled in, the
    /// helper it exercises does not exist.
    #[test]
    #[cfg(not(feature = "flash-attn"))]
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
