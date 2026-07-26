//! Tiled VAE decode with OOM fallback.
//!
//! Implements ComfyUI-style tiled VAE decoding for memory-constrained large
//! image generation: when a full-image VAE decode runs out of VRAM, the
//! pipeline retries by splitting the latent into overlapping tiles, decoding
//! each independently, and blending the results back together.
//!
//! See `comfy/sd.py:980-1032` for the reference behavior — full decode is
//! attempted first, OOM falls back to `decode_tiled_*` with three different
//! starting offsets averaged together to cancel seam artifacts.
//!
//! ## Tile units
//!
//! `TileConfig::tile_size` and `TileConfig::overlap` are in **latent**
//! coordinates. A 64×64 latent tile decodes to a 512×512 image tile when the
//! VAE upsamples 8× (FLUX, FLUX.2, SDXL, SD3 all use 8×). Override the
//! upsample factor with `decode_tiled_with_scale` when wiring against a VAE
//! that uses a different ratio.
//!
//! The default tile size is intentionally smaller than typical generation
//! latents (128² for a 1024² image) so the tiled fallback genuinely
//! subdivides — earlier defaults of `tile_size = 128` produced a single tile
//! covering the full latent at 1024², which made the OOM retry a no-op. If a
//! caller supplies a config where `tile_size` is still ≥ the smaller latent
//! axis, [`decode_with_oom_fallback`] shrinks it to half the smaller axis on
//! the fallback path so the retry always uses less memory than the full
//! decode that just failed.
//!
//! ## Three-offset averaging
//!
//! `offsets == 3` runs the tile pass three times: `(0, 0)`, `(tile/2, 0)`,
//! `(0, tile/2)`. Tile boundaries land in different image positions on each
//! pass, so pixel-wise averaging cancels seams without any per-tile context
//! sharing. `offsets == 1` skips averaging — fastest, but tile seams are
//! visible at low overlap. ComfyUI defaults to 3.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};

/// Configuration for tiled VAE decode.
///
/// Sizes are in **latent** coordinates. Default is 64 latent-px tile with
/// 16 latent-px overlap and 3-offset averaging — matches ComfyUI's
/// `VAE.decode_tiled` defaults (`comfy/sd.py`) and produces a 512×512 image
/// tile through an 8× VAE. At 1024² generation (latent 128²) this gives a
/// 3×3 grid per offset pass, which is the smallest grid that genuinely
/// reduces VRAM pressure relative to a full decode.
#[derive(Debug, Clone, Copy)]
pub struct TileConfig {
    /// Tile edge length in latent space. The VAE upsample factor multiplies
    /// this to get the image-space tile (e.g. 64 latent → 512 image at 8×).
    pub tile_size: usize,
    /// Overlap between adjacent tiles in latent space. Larger values produce
    /// smoother seams at the cost of more redundant work.
    pub overlap: usize,
    /// Number of offset passes to average. `1` skips averaging entirely (fast,
    /// faint seams may be visible). `3` matches ComfyUI's seam-cancellation
    /// behavior with three different start offsets.
    pub offsets: usize,
}

impl Default for TileConfig {
    fn default() -> Self {
        Self {
            tile_size: 64,
            overlap: 16,
            offsets: 3,
        }
    }
}

/// Default VAE upsample factor for FLUX, FLUX.2, SDXL, SD3.
pub const DEFAULT_VAE_SCALE: usize = 8;

/// Resolved mode for the `MOLD_VAE_TILED` env override.
///
/// - `Auto` (default): full decode first, tile only on OOM.
/// - `Force`: skip the full-decode attempt and tile from the start. Useful
///   when the user has already observed OOMs and wants to preempt the cost.
/// - `Off`: never tile, even on OOM. Surfaces the underlying error.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum TiledMode {
    #[default]
    Auto,
    Force,
    Off,
}

/// Read `MOLD_VAE_TILED` once and resolve the process-wide [`TiledMode`].
/// Scheduler plans and execution share this same authority, so changing the
/// environment after admission cannot alter a granted generation.
///
/// Accepts `auto`, `force`, `off`, plus boolean-ish synonyms: `1`, `true`, `yes`
/// map to `Force`; `0`, `false`, `no` map to `Off`. Anything unrecognized
/// (including unset) returns [`TiledMode::Auto`].
pub fn resolve_mode() -> TiledMode {
    static CACHED: std::sync::OnceLock<TiledMode> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| parse_mode(std::env::var("MOLD_VAE_TILED").ok().as_deref()))
}

fn parse_mode(value: Option<&str>) -> TiledMode {
    match value.map(|s| s.trim().to_ascii_lowercase()).as_deref() {
        Some("force") | Some("1") | Some("true") | Some("yes") | Some("on") => TiledMode::Force,
        Some("off") | Some("0") | Some("false") | Some("no") => TiledMode::Off,
        _ => TiledMode::Auto,
    }
}

/// Detect CUDA out-of-memory errors by string-matching the underlying driver
/// message.
///
/// candle today doesn't expose a typed OOM variant — every fallback ladder in
/// the codebase keys off the error text. This helper consolidates the known
/// substrings so we don't drift across engines.
pub fn is_cuda_oom(err: &impl std::fmt::Display) -> bool {
    let msg = err.to_string();
    msg.contains("OUT_OF_MEMORY")
        || msg.contains("out of memory")
        || msg.contains("OutOfMemory")
        || msg.contains("CUDA_ERROR_OUT_OF_MEMORY")
        || msg.contains("cudaErrorMemoryAllocation")
}

/// Tile-decode a latent tensor with the default 8× VAE upscale.
///
/// `latents` is expected to be `[1, C, H, W]` in latent coordinates. The
/// returned tensor is `[1, 3, H*8, W*8]` on CPU (`f32`). Callers that need a
/// specific device/dtype should cast after this returns — the accumulator is
/// CPU-bound to keep VRAM pressure low (the whole reason we're tiling).
pub fn decode_tiled<F>(latents: &Tensor, decode_fn: F, config: &TileConfig) -> Result<Tensor>
where
    F: Fn(&Tensor) -> Result<Tensor>,
{
    decode_tiled_with_scale(latents, decode_fn, config, DEFAULT_VAE_SCALE)
}

/// Tile-decode a latent tensor with a configurable upscale factor.
pub fn decode_tiled_with_scale<F>(
    latents: &Tensor,
    decode_fn: F,
    config: &TileConfig,
    vae_scale: usize,
) -> Result<Tensor>
where
    F: Fn(&Tensor) -> Result<Tensor>,
{
    if config.tile_size == 0 {
        anyhow::bail!("decode_tiled: tile_size must be > 0");
    }
    if config.overlap >= config.tile_size {
        anyhow::bail!(
            "decode_tiled: overlap ({}) must be smaller than tile_size ({})",
            config.overlap,
            config.tile_size
        );
    }
    if vae_scale == 0 {
        anyhow::bail!("decode_tiled: vae_scale must be > 0");
    }

    let offsets = match config.offsets {
        0 | 1 => vec![(0usize, 0usize)],
        3 => vec![(0, 0), (config.tile_size / 2, 0), (0, config.tile_size / 2)],
        n => anyhow::bail!("decode_tiled: offsets={} unsupported (use 1 or 3)", n),
    };

    let (_, _, lat_h, lat_w) = latents.dims4()?;
    let img_h = lat_h * vae_scale;
    let img_w = lat_w * vae_scale;

    // Sum over offset passes, then divide.
    let mut sum_acc = vec![0f32; 3 * img_h * img_w];

    for (off_y, off_x) in &offsets {
        let pass = decode_one_offset(
            latents,
            &decode_fn,
            config.tile_size,
            config.overlap,
            vae_scale,
            *off_y,
            *off_x,
        )?;
        // pass is on CPU as [1, 3, H, W] f32
        let pass_data: Vec<f32> = pass.flatten_all()?.to_vec1()?;
        debug_assert_eq!(pass_data.len(), sum_acc.len());
        for (s, p) in sum_acc.iter_mut().zip(pass_data.iter()) {
            *s += *p;
        }
    }

    let n_offsets = offsets.len() as f32;
    for s in sum_acc.iter_mut() {
        *s /= n_offsets;
    }

    let out = Tensor::from_vec(sum_acc, (1, 3, img_h, img_w), &Device::Cpu)?;
    Ok(out)
}

/// VAE decode with automatic OOM-driven tiled fallback.
///
/// Behavior is controlled by the `MOLD_VAE_TILED` env var (see [`resolve_mode`]):
///
/// - `auto` (default): try `decode_fn(latents)` first. On a CUDA-OOM error,
///   call `on_oom_recover` (typically a `device.synchronize()`) so freed
///   async memory actually returns to the allocator, then retry with tiled
///   decode using [`TileConfig::default`].
/// - `force`: skip the full-decode attempt and tile from the start. Result
///   stays on CPU (`f32`) — caller is responsible for moving back to GPU /
///   target dtype if needed.
/// - `off`: never tile, even on OOM. Surfaces the underlying error.
///
/// On success the tensor is returned in whatever device/dtype `decode_fn`
/// produced when no fallback was triggered. When tiling is used the result
/// lives on CPU (`f32`); callers that need GPU/BF16 must `to_device` /
/// `to_dtype` after.
/// Defense-in-depth: ensure the tiled fallback config genuinely subdivides
/// the input latent.
///
/// When `cfg.tile_size` is ≥ the smaller latent axis the tile grid collapses
/// to a single tile covering the whole input, so the "tiled retry" runs the
/// exact same allocation as the full decode that just OOM'd. This helper
/// shrinks `tile_size` to roughly half the smaller axis (rounded down to a
/// multiple of 8) and adjusts `overlap` proportionally so the retry actually
/// produces multiple smaller decodes. The minimum tile size is `MIN_TILE`
/// to keep the decode count bounded — for very small latents (sub-256² gen)
/// the full decode shouldn't OOM in the first place, so this floor is
/// preferred over collapsing to micro-tiles.
pub(crate) fn shrink_tile_for_latent(
    mut cfg: TileConfig,
    lat_h: usize,
    lat_w: usize,
) -> TileConfig {
    /// Minimum subdivided tile size in latent space. 32 latent → 256 image at
    /// 8× — small enough to relieve VRAM pressure from a 1024–2048² full
    /// decode, large enough to keep tile-count overhead bounded.
    const MIN_TILE: usize = 32;
    let min_axis = lat_h.min(lat_w);
    if min_axis == 0 || cfg.tile_size < min_axis {
        return cfg;
    }
    let half = (min_axis / 2) & !7;
    let shrunk = half.max(MIN_TILE);
    if shrunk >= min_axis {
        // Latent is already at or below the floor — single tile is the only
        // option. Leave cfg as-is; the failed full decode will surface the
        // OOM rather than retrying redundantly.
        return cfg;
    }
    tracing::debug!(
        requested_tile = cfg.tile_size,
        shrunk_tile = shrunk,
        latent_h = lat_h,
        latent_w = lat_w,
        "tile_size ≥ latent axis — shrinking so tiled fallback subdivides"
    );
    cfg.tile_size = shrunk;
    if cfg.overlap >= cfg.tile_size {
        cfg.overlap = cfg.tile_size / 4;
    }
    cfg
}

pub fn decode_with_oom_fallback<F, R>(
    latents: &Tensor,
    decode_fn: F,
    on_oom_recover: R,
) -> Result<Tensor>
where
    F: Fn(&Tensor) -> Result<Tensor>,
    R: FnOnce(),
{
    let mode = resolve_mode();
    let mut cfg = TileConfig::default();

    if let Ok((_, _, lat_h, lat_w)) = latents.dims4() {
        cfg = shrink_tile_for_latent(cfg, lat_h, lat_w);
    }

    if matches!(mode, TiledMode::Force) {
        tracing::info!(
            tile_size = cfg.tile_size,
            overlap = cfg.overlap,
            offsets = cfg.offsets,
            "MOLD_VAE_TILED=force — tiling VAE decode without trying full decode first"
        );
        return decode_tiled(latents, decode_fn, &cfg);
    }

    match decode_fn(latents) {
        Ok(t) => Ok(t),
        Err(e) if matches!(mode, TiledMode::Off) => Err(e),
        Err(e) if is_cuda_oom(&e) => {
            tracing::warn!(
                error = %e,
                tile_size = cfg.tile_size,
                overlap = cfg.overlap,
                offsets = cfg.offsets,
                "VAE decode OOM — retrying with tiled decode"
            );
            // Caller-provided recovery (typically `device.synchronize()`) so
            // any async-freed VRAM from the failed decode actually returns
            // to the allocator before we start tiling.
            on_oom_recover();
            decode_tiled(latents, decode_fn, &cfg)
        }
        Err(e) => Err(e),
    }
}

/// Run a single tile pass with the given start offset.
///
/// The offset shifts the *anchor* of the tile grid: `off_y=0, off_x=0` puts
/// the first tile at the top-left; `off_y=tile/2` shifts the grid down by
/// half a tile so a tile boundary that landed at row `tile` in the no-offset
/// pass instead lands at row `tile + tile/2`. We achieve this by reflecting
/// the input edges (so seams from edge-tile boundaries average out cleanly
/// against the no-offset pass).
fn decode_one_offset<F>(
    latents: &Tensor,
    decode_fn: &F,
    tile_size: usize,
    overlap: usize,
    vae_scale: usize,
    off_y: usize,
    off_x: usize,
) -> Result<Tensor>
where
    F: Fn(&Tensor) -> Result<Tensor>,
{
    let (_, _, lat_h, lat_w) = latents.dims4()?;
    let tiles = calculate_tiles_offset(lat_w, lat_h, tile_size, overlap, off_y, off_x);

    let img_h = lat_h * vae_scale;
    let img_w = lat_w * vae_scale;

    let mut output_acc = vec![0f32; 3 * img_h * img_w];
    let mut weight_acc = vec![0f32; img_h * img_w];

    for tile in &tiles {
        let tile_input = latents
            .narrow(2, tile.y, tile.h)?
            .narrow(3, tile.x, tile.w)?;
        let tile_output = decode_fn(&tile_input)?;
        let tile_output = tile_output.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
        let (_, channels, out_th, out_tw) = tile_output.dims4()?;
        if channels != 3 {
            anyhow::bail!(
                "decode_tiled: expected 3-channel VAE output, got {} channels",
                channels
            );
        }
        debug_assert_eq!(out_th, tile.h * vae_scale);
        debug_assert_eq!(out_tw, tile.w * vae_scale);

        let tile_data: Vec<f32> = tile_output.flatten_all()?.to_vec1()?;
        let out_x = tile.x * vae_scale;
        let out_y = tile.y * vae_scale;

        let weights = build_blend_weights_2d(
            tile.x, tile.y, tile.w, tile.h, lat_w, lat_h, overlap, vae_scale,
        );

        for c in 0..3 {
            for row in 0..out_th {
                for col in 0..out_tw {
                    let w = weights[row * out_tw + col];
                    let val = tile_data[c * out_th * out_tw + row * out_tw + col];
                    let dst_row = out_y + row;
                    let dst_col = out_x + col;
                    output_acc[c * img_h * img_w + dst_row * img_w + dst_col] += val * w;
                    if c == 0 {
                        weight_acc[dst_row * img_w + dst_col] += w;
                    }
                }
            }
        }
    }

    for c in 0..3 {
        for i in 0..img_h * img_w {
            if weight_acc[i] > 0.0 {
                output_acc[c * img_h * img_w + i] /= weight_acc[i];
            }
        }
    }

    Tensor::from_vec(output_acc, (1, 3, img_h, img_w), &Device::Cpu).map_err(Into::into)
}

/// A tile region in latent space.
struct TileRegion {
    x: usize,
    y: usize,
    w: usize,
    h: usize,
}

/// Calculate the tile grid for a latent of size `lat_w × lat_h`, with the
/// first tile anchored at `(off_x, off_y)`.
///
/// When `off > 0` the anchor sits *inside* the latent — we still need a tile
/// covering `[0, off)` along that axis. We emit a leading "stub" tile of size
/// `min(tile_size, off)` anchored at zero so the leading region is decoded
/// and the offset shift happens at a deeper interior boundary. This is
/// simpler than reflecting the input, and keeps every tile's coordinates
/// inside the actual latent bounds.
fn calculate_tiles_offset(
    lat_w: usize,
    lat_h: usize,
    tile_size: usize,
    overlap: usize,
    off_y: usize,
    off_x: usize,
) -> Vec<TileRegion> {
    let step = tile_size.saturating_sub(overlap).max(1);

    // Build per-axis start positions.
    let xs = axis_starts(lat_w, tile_size, step, off_x);
    let ys = axis_starts(lat_h, tile_size, step, off_y);

    let mut tiles = Vec::with_capacity(xs.len() * ys.len());
    for &y in &ys {
        let h = tile_size.min(lat_h - y);
        for &x in &xs {
            let w = tile_size.min(lat_w - x);
            tiles.push(TileRegion { x, y, w, h });
        }
    }
    tiles
}

/// Compute per-axis tile start positions. With `offset = 0` this matches the
/// non-offset grid: 0, step, 2*step, …, capped at `len - tile`. With
/// `offset > 0` the leading partial region `[0, offset)` is covered by an
/// extra tile starting at 0 (whose interior overlaps the offset-anchored
/// tile), so seams from the no-offset pass at multiples of `step` fall on
/// different pixels than seams from this pass.
fn axis_starts(len: usize, tile_size: usize, step: usize, offset: usize) -> Vec<usize> {
    let mut out = Vec::new();
    if len == 0 {
        return out;
    }
    if len <= tile_size {
        out.push(0);
        return out;
    }
    if offset > 0 && offset < len {
        // Leading stub anchored at 0 covers the [0, offset+tile/?] region.
        out.push(0);
    }
    let mut x = offset;
    loop {
        let clamped = x.min(len.saturating_sub(tile_size));
        if out.last().is_none_or(|&last| last != clamped) {
            out.push(clamped);
        }
        if clamped + tile_size >= len {
            break;
        }
        x += step;
    }
    out
}

/// Build a feathered blend-weight buffer for one tile, output as a flat
/// `out_h * out_w` `f32` vector. Edges that touch the latent boundary get
/// weight 1.0; interior edges ramp linearly from 0 → 1 over the overlap
/// region in image space.
#[allow(clippy::too_many_arguments)]
fn build_blend_weights_2d(
    tile_x: usize,
    tile_y: usize,
    tile_w: usize,
    tile_h: usize,
    lat_w: usize,
    lat_h: usize,
    overlap: usize,
    scale: usize,
) -> Vec<f32> {
    let out_w = tile_w * scale;
    let out_h = tile_h * scale;
    let out_overlap = overlap * scale;
    let mut weights = vec![1.0f32; out_h * out_w];

    if tile_x > 0 && out_overlap > 0 {
        let ramp_len = out_overlap.min(out_w);
        for row in 0..out_h {
            for col in 0..ramp_len {
                let factor = (col as f32 + 1.0) / ramp_len as f32;
                weights[row * out_w + col] *= factor;
            }
        }
    }
    if tile_y > 0 && out_overlap > 0 {
        let ramp_len = out_overlap.min(out_h);
        for row in 0..ramp_len {
            let factor = (row as f32 + 1.0) / ramp_len as f32;
            for col in 0..out_w {
                weights[row * out_w + col] *= factor;
            }
        }
    }
    if tile_x + tile_w < lat_w && out_overlap > 0 {
        let ramp_len = out_overlap.min(out_w);
        for row in 0..out_h {
            for col in 0..ramp_len {
                let factor = (col as f32 + 1.0) / ramp_len as f32;
                weights[row * out_w + (out_w - 1 - col)] *= factor;
            }
        }
    }
    if tile_y + tile_h < lat_h && out_overlap > 0 {
        let ramp_len = out_overlap.min(out_h);
        for row in 0..ramp_len {
            let factor = (row as f32 + 1.0) / ramp_len as f32;
            for col in 0..out_w {
                weights[(out_h - 1 - row) * out_w + col] *= factor;
            }
        }
    }
    weights
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Synthetic decode_fn: 8× nearest-neighbor upsample of channel 0 (broadcast
    /// to 3 RGB channels) plus a constant bias of 0.1. Deterministic and
    /// independent of tile size, so tile decode should match full decode
    /// exactly in the interior.
    fn synthetic_decode(input: &Tensor) -> Result<Tensor> {
        let (_, _, h, w) = input.dims4()?;
        let scale = 8usize;
        // Take channel 0, repeat each value scale times in both spatial dims.
        let ch0 = input.narrow(1, 0, 1)?.to_device(&Device::Cpu)?;
        let upsampled = ch0.upsample_nearest2d(h * scale, w * scale)?;
        let stacked = Tensor::cat(&[&upsampled, &upsampled, &upsampled], 1)?;
        let biased = (stacked + 0.1f64)?;
        Ok(biased.to_dtype(DType::F32)?)
    }

    fn random_latent(c: usize, h: usize, w: usize) -> Tensor {
        // Deterministic fake latent: x[c, y, x] = sin(c + y/3 + x/5)
        let mut data = Vec::with_capacity(c * h * w);
        for ci in 0..c {
            for y in 0..h {
                for x in 0..w {
                    let v = (ci as f32 + (y as f32) / 3.0 + (x as f32) / 5.0).sin();
                    data.push(v);
                }
            }
        }
        Tensor::from_vec(data, (1, c, h, w), &Device::Cpu).unwrap()
    }

    #[test]
    fn test_tile_config_default() {
        let cfg = TileConfig::default();
        assert_eq!(cfg.tile_size, 64);
        assert_eq!(cfg.overlap, 16);
        assert_eq!(cfg.offsets, 3);
    }

    /// Regression: at 1024² generation the latent is 128×128. If the default
    /// tile_size is ≥ 128 the tiled fallback emits a single tile covering the
    /// whole latent and the OOM retry just re-runs the same full decode that
    /// already failed. The default must subdivide a 128×128 latent.
    #[test]
    fn test_default_tile_size_subdivides_1024_latent() {
        let cfg = TileConfig::default();
        assert!(
            cfg.tile_size < 128,
            "default tile_size ({}) must be < 128 so the OOM fallback actually \
             tiles a 1024² latent (128×128). With tile_size ≥ 128, axis_starts \
             returns a single tile and the retry equals the failed full decode.",
            cfg.tile_size,
        );
    }

    /// At the typical 1024² latent (128×128) the new default tile_size of 64
    /// already subdivides — shrink should leave the config untouched.
    #[test]
    fn test_shrink_tile_no_op_when_default_already_subdivides_1024() {
        let cfg = TileConfig::default();
        let out = shrink_tile_for_latent(cfg, 128, 128);
        assert_eq!(out.tile_size, cfg.tile_size);
        assert_eq!(out.overlap, cfg.overlap);
    }

    /// Defense in depth: a config with tile_size ≥ the smaller latent axis
    /// (e.g. a hand-tuned 128 against a 1024² latent, or a future default
    /// regression) must shrink so the tiled retry actually subdivides.
    #[test]
    fn test_shrink_tile_subdivides_when_tile_ge_latent() {
        let cfg = TileConfig {
            tile_size: 128,
            overlap: 32,
            offsets: 3,
        };
        let out = shrink_tile_for_latent(cfg, 128, 128);
        assert!(
            out.tile_size < 128,
            "shrunk tile_size ({}) must be < latent dim 128 so the retry \
             produces multiple tiles",
            out.tile_size,
        );
        assert!(
            out.overlap < out.tile_size,
            "overlap ({}) must remain < tile_size ({})",
            out.overlap,
            out.tile_size,
        );
        // Sanity: shrunk tile must be a sane multiple-of-8 size.
        assert_eq!(out.tile_size % 8, 0);
    }

    /// For tiny latents (smaller than the floor), shrink is a no-op — the
    /// full decode shouldn't OOM here in practice, so leaving cfg untouched
    /// surfaces any underlying error rather than redundantly re-running.
    #[test]
    fn test_shrink_tile_no_op_when_latent_below_floor() {
        let cfg = TileConfig::default();
        let out = shrink_tile_for_latent(cfg, 16, 16);
        assert_eq!(out.tile_size, cfg.tile_size);
    }

    /// Asymmetric latents (e.g. 1024×768 → 128×96) shrink based on the
    /// smaller axis.
    #[test]
    fn test_shrink_tile_uses_smaller_axis() {
        let cfg = TileConfig {
            tile_size: 128,
            overlap: 32,
            offsets: 3,
        };
        let out = shrink_tile_for_latent(cfg, 96, 128);
        assert!(out.tile_size < 96);
    }

    #[test]
    fn test_is_cuda_oom_matches_known_strings() {
        // Common driver strings the helper must recognize.
        assert!(is_cuda_oom(&"CUDA out of memory"));
        assert!(is_cuda_oom(&"CUDA_ERROR_OUT_OF_MEMORY"));
        assert!(is_cuda_oom(&"cudaErrorMemoryAllocation"));
        assert!(is_cuda_oom(&"OutOfMemory: ..."));
        assert!(is_cuda_oom(&"some prefix: out of memory: ..."));
        assert!(is_cuda_oom(&"OUT_OF_MEMORY: requested 5GB"));

        // Negative cases.
        assert!(!is_cuda_oom(&"some other error"));
        assert!(!is_cuda_oom(&"shape mismatch"));
        assert!(!is_cuda_oom(&""));
    }

    #[test]
    fn test_resolve_mode_env() {
        // Default → Auto.
        assert_eq!(parse_mode(None), TiledMode::Auto);
        assert_eq!(parse_mode(Some("")), TiledMode::Auto);
        assert_eq!(parse_mode(Some("auto")), TiledMode::Auto);
        assert_eq!(parse_mode(Some("AUTO")), TiledMode::Auto);
        assert_eq!(parse_mode(Some("garbage")), TiledMode::Auto);

        // Force aliases.
        assert_eq!(parse_mode(Some("force")), TiledMode::Force);
        assert_eq!(parse_mode(Some("FORCE")), TiledMode::Force);
        assert_eq!(parse_mode(Some("1")), TiledMode::Force);
        assert_eq!(parse_mode(Some("true")), TiledMode::Force);
        assert_eq!(parse_mode(Some("yes")), TiledMode::Force);
        assert_eq!(parse_mode(Some("on")), TiledMode::Force);

        // Off aliases.
        assert_eq!(parse_mode(Some("off")), TiledMode::Off);
        assert_eq!(parse_mode(Some("0")), TiledMode::Off);
        assert_eq!(parse_mode(Some("false")), TiledMode::Off);
        assert_eq!(parse_mode(Some("no")), TiledMode::Off);
    }

    #[test]
    fn test_axis_starts_no_offset() {
        // Single tile fits.
        assert_eq!(axis_starts(16, 32, 16, 0), vec![0]);
        // step=8, len=32, tile=16 → 0, 8, 16
        let starts = axis_starts(32, 16, 8, 0);
        assert_eq!(starts.first(), Some(&0));
        assert!(*starts.last().unwrap() + 16 == 32);
    }

    #[test]
    fn test_axis_starts_with_offset_includes_zero_stub() {
        // With offset=4, we should still cover [0, 4).
        let starts = axis_starts(32, 16, 8, 4);
        assert_eq!(starts.first(), Some(&0));
    }

    #[test]
    fn test_decode_tiled_single_offset_matches_full() {
        // 16x16 latent → 128x128 image at 8×.
        let latents = random_latent(4, 16, 16);
        let cfg = TileConfig {
            tile_size: 8,
            overlap: 2,
            offsets: 1,
        };
        let full = synthetic_decode(&latents).unwrap();
        let full_data: Vec<f32> = full.flatten_all().unwrap().to_vec1().unwrap();

        let tiled = decode_tiled(&latents, synthetic_decode, &cfg).unwrap();
        let tiled_data: Vec<f32> = tiled.flatten_all().unwrap().to_vec1().unwrap();

        assert_eq!(full_data.len(), tiled_data.len());

        // Synthetic decode is location-independent (each output pixel is just a
        // function of the corresponding input pixel + bias), so even with
        // tile boundaries the blended output should match the full decode
        // tightly across the entire image.
        let mut max_diff = 0.0f32;
        for (a, b) in full_data.iter().zip(tiled_data.iter()) {
            let d = (a - b).abs();
            if d > max_diff {
                max_diff = d;
            }
        }
        assert!(
            max_diff < 1e-2,
            "tiled decode diverges from full decode: max_diff={max_diff}"
        );
    }

    #[test]
    fn test_decode_tiled_three_offset_smooths_seams() {
        // Same setup; with offsets=3 the result should be at least as close
        // to the full decode as offsets=1 (and never worse on average).
        let latents = random_latent(4, 16, 16);
        let full = synthetic_decode(&latents).unwrap();
        let full_data: Vec<f32> = full.flatten_all().unwrap().to_vec1().unwrap();

        let cfg1 = TileConfig {
            tile_size: 8,
            overlap: 2,
            offsets: 1,
        };
        let cfg3 = TileConfig {
            tile_size: 8,
            overlap: 2,
            offsets: 3,
        };

        let one = decode_tiled(&latents, synthetic_decode, &cfg1).unwrap();
        let three = decode_tiled(&latents, synthetic_decode, &cfg3).unwrap();
        let one_data: Vec<f32> = one.flatten_all().unwrap().to_vec1().unwrap();
        let three_data: Vec<f32> = three.flatten_all().unwrap().to_vec1().unwrap();

        let mse = |a: &[f32], b: &[f32]| -> f32 {
            let n = a.len() as f32;
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f32>()
                / n
        };

        let mse1 = mse(&full_data, &one_data);
        let mse3 = mse(&full_data, &three_data);

        // 3-offset should be no worse than 1-offset on this synthetic
        // (location-independent) test. With a real seam-introducing decoder
        // it should be strictly better.
        assert!(
            mse3 <= mse1 + 1e-6,
            "3-offset MSE ({mse3}) should not exceed 1-offset MSE ({mse1})"
        );
        assert!(
            mse3 < 1e-3,
            "3-offset MSE ({mse3}) should be tight on synthetic decode"
        );
    }

    #[test]
    fn test_decode_tiled_rejects_zero_tile_size() {
        let latents = random_latent(4, 16, 16);
        let cfg = TileConfig {
            tile_size: 0,
            overlap: 0,
            offsets: 1,
        };
        let res = decode_tiled(&latents, synthetic_decode, &cfg);
        assert!(res.is_err());
    }

    #[test]
    fn test_decode_tiled_rejects_overlap_geq_tile() {
        let latents = random_latent(4, 16, 16);
        let cfg = TileConfig {
            tile_size: 8,
            overlap: 8,
            offsets: 1,
        };
        let res = decode_tiled(&latents, synthetic_decode, &cfg);
        assert!(res.is_err());
    }

    #[test]
    fn test_decode_tiled_rejects_unsupported_offsets() {
        let latents = random_latent(4, 16, 16);
        let cfg = TileConfig {
            tile_size: 8,
            overlap: 2,
            offsets: 5,
        };
        let res = decode_tiled(&latents, synthetic_decode, &cfg);
        assert!(res.is_err());
    }

    #[test]
    fn test_decode_tiled_single_tile_fits_inside_tile_size() {
        // Latent smaller than tile_size — we still produce a correct decode.
        let latents = random_latent(4, 4, 4);
        let cfg = TileConfig {
            tile_size: 16,
            overlap: 4,
            offsets: 1,
        };
        let full = synthetic_decode(&latents).unwrap();
        let tiled = decode_tiled(&latents, synthetic_decode, &cfg).unwrap();
        let full_data: Vec<f32> = full.flatten_all().unwrap().to_vec1().unwrap();
        let tiled_data: Vec<f32> = tiled.flatten_all().unwrap().to_vec1().unwrap();
        let max_diff = full_data
            .iter()
            .zip(tiled_data.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(max_diff < 1e-3, "single-tile decode mismatch: {max_diff}");
    }
}
