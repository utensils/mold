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
//! `OnceLock` so we don't re-read the environment on every block. The
//! default is `Math` in every build — compiling `flash-attn` makes the kernel
//! available but never switches it on (#736), so a shipped artifact cannot
//! silently change seed reproducibility.
//!
//! [`attention_with_bias`] adds an optional additive `[B, H, Q, K]` bias for
//! callers that must mask keys (Qwen-Image's joint stream when the two batched
//! CFG prompts differ in length). A bias always takes the math path — FA2 has
//! no additive-bias entry point here — but keeps the same query chunking.
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
    /// 2. Otherwise → `Math`, in every build. The `flash-attn` cargo feature
    ///    only makes `flash` *available*; it never changes the default.
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

/// The backend used when `MOLD_ATTN` is unset or unparseable.
///
/// Always `Math`, in every build. Compiling the `flash-attn` feature makes
/// the FlashAttention kernel *available*; it does not make it the default
/// (#736). FA2 is mathematically equivalent to the math path but not
/// bit-identical (fp32 online-softmax accumulator versus an input-dtype
/// reduce), so a build-time default would change the image a CUDA user gets
/// for a given seed based on which artifact they downloaded — against the
/// cross-backend seed determinism the CPU-noise path exists to preserve.
/// `MOLD_ATTN=flash` is the one opt-in.
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

/// Scaled dot-product attention with an optional additive bias broadcast over
/// `[B, H, Q, K]` (`0` keeps a key, `-inf` drops it).
///
/// `None` is the existing [`attention`] dispatch (flash when eligible, chunked
/// math otherwise). `Some` never takes flash — FA2 has no additive-bias entry
/// point here — but keeps the same query-chunk policy, so `MOLD_ATTN_CHUNK`
/// still bounds the score matrix on the masked path.
pub fn attention_with_bias(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    bias: Option<&Tensor>,
) -> Result<Tensor> {
    if !bias_forces_math(bias) {
        return attention(q, k, v, scale);
    }
    let bias = bias.expect("bias_forces_math is true exactly when the bias is Some");
    math_attention_biased_impl(q, k, v, scale, bias, math_attention_chunk_size(q))
}

/// Pure predicate so the dispatch is testable without a GPU.
pub(crate) fn bias_forces_math(bias: Option<&Tensor>) -> bool {
    bias.is_some()
}

/// Biased math attention, kept 4-D on purpose.
///
/// [`math_attention_impl`] flattens `B·H` into one leading dim, which a
/// `[B, 1, 1, K]` bias cannot broadcast against — the batch axis would line up
/// with `B*H`. So this mirrors the arithmetic without the flatten:
/// `QK^T · scale → + bias → softmax → · V`.
fn math_attention_biased_impl(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    bias: &Tensor,
    chunk_size: Option<usize>,
) -> Result<Tensor> {
    let k_t = k.transpose(D::Minus2, D::Minus1)?;
    let biased_chunk = |q_chunk: &Tensor, bias_chunk: &Tensor| -> Result<Tensor> {
        let attn_weights = (q_chunk.matmul(&k_t)? * f64::from(scale))?;
        let attn_weights = attn_weights.broadcast_add(bias_chunk)?;
        candle_nn::ops::softmax_last_dim(&attn_weights)?.matmul(v)
    };

    let Some(chunk_size) = chunk_size else {
        return biased_chunk(q, bias);
    };

    let q_len = q.dim(D::Minus2)?;
    // The Qwen bias is `[B, 1, 1, K]`, so in practice it never narrows; a bias
    // that does vary along the query axis has to follow the chunk.
    let bias_is_per_query = bias.dim(D::Minus2)? > 1;
    let mut chunks = Vec::with_capacity(q_len.div_ceil(chunk_size));
    let mut start = 0;
    while start < q_len {
        let len = (q_len - start).min(chunk_size);
        let q_chunk = q.narrow(D::Minus2, start, len)?;
        let bias_chunk = if bias_is_per_query {
            bias.narrow(D::Minus2, start, len)?
        } else {
            bias.clone()
        };
        chunks.push(biased_chunk(&q_chunk, &bias_chunk)?);
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
    Tensor::cat(&refs, D::Minus2)
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

/// Math attention with an explicit query chunk, forced on every device.
///
/// [`math_attention`]'s `Auto` policy only chunks on CUDA past 1024 query rows,
/// so a Metal or CPU caller always materialises the whole `Q x K` score matrix.
/// That is the right default for a transformer block, whose score matrix is
/// small next to its weights — but not for the Qwen-Image VAE mid-block, which
/// attends over every latent token at once and whose two `N x N` F32 buffers
/// are the single largest allocation in the decode (~3 GB each at a 1328²
/// render). Such a caller forces the chunk here instead of hoping the heuristic
/// agrees with it.
///
/// Chunking is arithmetically a no-op: each chunk softmaxes over the full key
/// axis, so the concatenated result is bit-comparable to the single-pass path.
/// `chunk_size` is clamped to at least one row (zero would not terminate), and
/// a chunk at or above the query length degenerates to a single pass.
pub fn math_attention_with_chunk(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    chunk_size: usize,
) -> Result<Tensor> {
    math_attention_impl(q, k, v, scale, Some(chunk_size.max(1)))
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

/// Query rows per chunk the `Auto` policy uses on CUDA.
pub(crate) const CUDA_AUTO_QUERY_CHUNK: usize = 512;
/// Query length past which the `Auto` policy engages the CUDA chunk.
pub(crate) const CUDA_AUTO_CHUNK_MIN_QUERY_ROWS: usize = 1024;

/// The query chunk the CUDA math path would use for a `q_len`-row score
/// matrix, or `None` when it materializes the whole matrix at once.
///
/// Exposed because the score matrix is the largest allocation in a Qwen-Image
/// denoise step, so `qwen_image::pipeline`'s VRAM estimate has to price the
/// chunk this function actually picks — including a `MOLD_ATTN_CHUNK` override
/// that raises it, or `off`, which restores the full matrix.
pub(crate) fn cuda_query_chunk_rows(q_len: usize) -> Option<usize> {
    match resolved_chunk_policy() {
        AttentionChunkPolicy::Off => None,
        AttentionChunkPolicy::Size(size) => (size < q_len).then_some(size),
        AttentionChunkPolicy::Auto => {
            (q_len > CUDA_AUTO_CHUNK_MIN_QUERY_ROWS).then_some(CUDA_AUTO_QUERY_CHUNK)
        }
    }
}

fn math_attention_chunk_size(q: &Tensor) -> Option<usize> {
    let q_len = q.dim(D::Minus2).ok()?;
    if matches!(q.device(), Device::Cuda(_)) {
        return cuda_query_chunk_rows(q_len);
    }
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
    if matches!(q.device(), Device::Metal(_)) && q_len > CUDA_AUTO_CHUNK_MIN_QUERY_ROWS {
        Some(CUDA_AUTO_QUERY_CHUNK)
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

    /// `math_attention_with_chunk` is the device-independent entry point the
    /// Qwen-Image VAE mid-block uses: `math_attention`'s `Auto` policy only
    /// chunks on CUDA past 1024 rows, so Metal and CPU would otherwise still
    /// materialise the whole `N x N` score matrix. Chunking is an arithmetic
    /// no-op, so the forced path must agree with the unforced one — including
    /// at `H = 1`, which is the VAE's own single-head layout.
    #[test]
    fn math_attention_with_chunk_matches_math_attention() {
        for shape in [(1, 1, 37, 24), (1, 1, 64, 8), (2, 4, 16, 32)] {
            let (q, k, v) = rand_qkv(shape);
            let scale = 1.0 / (shape.3 as f32).sqrt();
            let want = math_attention(&q, &k, &v, scale).unwrap();

            for chunk in [1, 7, 16, shape.2, shape.2 * 2] {
                let got = math_attention_with_chunk(&q, &k, &v, scale, chunk).unwrap();
                assert_eq!(got.dims(), want.dims(), "shape {shape:?} chunk {chunk}");
                assert!(
                    max_abs_diff(&got, &want) < 1e-5,
                    "shape {shape:?} chunk {chunk} diverged from math_attention"
                );
            }
        }
    }

    /// A zero chunk would loop forever in `math_attention_chunked_flat`; the
    /// public entry point clamps it to a single row instead.
    #[test]
    fn math_attention_with_chunk_clamps_zero_to_one_row() {
        let (q, k, v) = rand_qkv((1, 1, 9, 8));
        let scale = 1.0 / (8f32).sqrt();
        let want = math_attention(&q, &k, &v, scale).unwrap();
        let got = math_attention_with_chunk(&q, &k, &v, scale, 0).unwrap();
        assert!(max_abs_diff(&got, &want) < 1e-5);
    }

    /// The invariant the whole Qwen mask removal rests on: attending over
    /// zero-padded keys with an additive `-inf` bias on the pad columns is the
    /// same operation as attending over the keys sliced to their true length.
    ///
    /// If this did not hold, dropping the padding in the pipeline would change
    /// the image stream's output.
    #[test]
    fn padded_keys_with_bias_match_sliced_attention() {
        let dev = cpu();
        let q = Tensor::randn(0.0_f32, 1.0_f32, (1, 2, 4, 8), &dev).unwrap();
        let k_real = Tensor::randn(0.0_f32, 1.0_f32, (1, 2, 3, 8), &dev).unwrap();
        let v_real = Tensor::randn(0.0_f32, 1.0_f32, (1, 2, 3, 8), &dev).unwrap();
        let pad = Tensor::zeros((1, 2, 2, 8), DType::F32, &dev).unwrap();
        let k = Tensor::cat(&[&k_real, &pad], 2).unwrap();
        let v = Tensor::cat(&[&v_real, &pad], 2).unwrap();
        let bias = Tensor::from_vec(
            vec![0.0_f32, 0.0, 0.0, f32::NEG_INFINITY, f32::NEG_INFINITY],
            (1, 1, 1, 5),
            &dev,
        )
        .unwrap();
        let scale = 1.0 / (8f32).sqrt();

        let padded = attention_with_bias(&q, &k, &v, scale, Some(&bias)).unwrap();
        let sliced = attention(&q, &k_real, &v_real, scale).unwrap();

        assert_eq!(padded.dims(), sliced.dims());
        assert!(
            max_abs_diff(&padded, &sliced) < 1e-5,
            "masked padding must equal slicing the padding away"
        );
    }

    /// `MOLD_ATTN_CHUNK` must keep working with a bias: the chunked biased
    /// path is the mirror of `test_chunked_math_attention_matches_full_math`.
    /// Both a key-only bias (`[B,1,1,K]`, the Qwen shape) and a per-query bias
    /// (`[B,1,Q,K]`, which forces the query-axis narrow) are covered.
    #[test]
    fn chunked_biased_matches_full_biased() {
        let dev = cpu();
        let (q, k, v) = rand_qkv((1, 3, 17, 16));
        let scale = 1.0 / (16f32).sqrt();

        let mut key_bias = vec![0.0_f32; 17];
        key_bias[15] = f32::NEG_INFINITY;
        key_bias[16] = f32::NEG_INFINITY;
        let key_bias = Tensor::from_vec(key_bias, (1, 1, 1, 17), &dev).unwrap();

        let full = math_attention_biased_impl(&q, &k, &v, scale, &key_bias, None).unwrap();
        let chunked = math_attention_biased_impl(&q, &k, &v, scale, &key_bias, Some(5)).unwrap();
        assert_eq!(chunked.dims(), full.dims());
        assert!(
            max_abs_diff(&chunked, &full) < 1e-5,
            "chunked biased attention diverged from the full biased pass"
        );

        // Per-query bias: the chunk loop has to narrow the bias too.
        let query_bias = Tensor::randn(0.0_f32, 1.0_f32, (1, 1, 17, 17), &dev).unwrap();
        let full = math_attention_biased_impl(&q, &k, &v, scale, &query_bias, None).unwrap();
        let chunked = math_attention_biased_impl(&q, &k, &v, scale, &query_bias, Some(4)).unwrap();
        assert!(
            max_abs_diff(&chunked, &full) < 1e-5,
            "a per-query bias must be narrowed alongside the query chunk"
        );
    }

    /// FA2 has no additive-bias entry point here, so a bias always takes the
    /// math path — pinned as a pure predicate so the dispatch is testable
    /// without a GPU.
    #[test]
    fn bias_forces_math_path() {
        let dev = cpu();
        let bias = Tensor::zeros((1, 1, 1, 4), DType::F32, &dev).unwrap();
        assert!(bias_forces_math(Some(&bias)));
        assert!(!bias_forces_math(None));
    }

    /// A `None` bias must be byte-for-byte the existing dispatch, so nothing
    /// that does not pad text changes behaviour.
    #[test]
    fn attention_with_no_bias_matches_plain_attention() {
        let (q, k, v) = rand_qkv((1, 2, 8, 16));
        let scale = 1.0 / (16f32).sqrt();
        let plain = attention(&q, &k, &v, scale).unwrap();
        let biased = attention_with_bias(&q, &k, &v, scale, None).unwrap();
        assert_eq!(max_abs_diff(&plain, &biased), 0.0);
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

    /// The default is `Math` in every build, including one compiled with
    /// `flash-attn` (#736). Compiling the kernel makes `flash` *available*;
    /// only an explicit `MOLD_ATTN=flash` turns it on. Pinning this without
    /// a `cfg` guard is the point: a future artifact that ships the kernel
    /// must not silently change the image every CUDA user gets for a seed.
    #[test]
    fn default_backend_is_math_regardless_of_feature() {
        assert_eq!(default_backend(), AttentionBackend::Math);
        assert_eq!(parse_backend_env(None), AttentionBackend::Math);
        assert_eq!(parse_backend_env(Some("")), AttentionBackend::Math);
        assert_eq!(parse_backend_env(Some("xformers")), AttentionBackend::Math);
    }

    /// Opting in still works in both builds: the parser returns `Flash`, and
    /// the dispatcher (not the parser) decides whether the kernel exists.
    #[test]
    fn flash_is_opt_in_via_env() {
        assert_eq!(parse_backend_env(Some("flash")), AttentionBackend::Flash);
        assert_eq!(parse_backend_env(Some(" Flash ")), AttentionBackend::Flash);
    }
}
