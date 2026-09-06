//! Tiled VAE decode with OOM fallback.
//!
//! Implements ComfyUI-style tiled VAE decoding for memory-constrained large
//! image generation: when a full-image VAE decode runs out of VRAM, the
//! pipeline retries by splitting the latent into overlapping tiles, decoding
//! each independently, and blending the results back together.
//!
//! See ComfyUI's `comfy/sd.py` (`VAE.decode` / `decode_tiled_`) for the
//! reference behavior — full decode is attempted first, and only an OOM falls
//! back to a tiled decode. ComfyUI's fallback averages three *tile shapes*
//! (square, wide, tall) at the same overlap; mold's `offsets == 3` variant
//! achieves the same seam cancellation by shifting the grid anchor instead.
//! Neither is ComfyUI's default decode path, and no upstream tiles
//! proactively at 1024² — a single feathered pass (`offsets == 1`) is what
//! diffusers (`AutoencoderKL.tiled_decode`) and stable-diffusion.cpp
//! (`process_tiles_2d` + smootherstep merge) use whenever they tile at all.
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
//! ## Offset averaging vs a single feathered pass
//!
//! `offsets == 3` runs the tile pass three times: `(0, 0)`, `(tile/2, 0)`,
//! `(0, tile/2)`. Tile boundaries land in different image positions on each
//! pass, so pixel-wise averaging cancels residual seams. It costs three
//! times the decode work and is reserved for the reactive OOM fallback,
//! whose halved tiles have less context per tile. `offsets == 1` runs one
//! pass; the normalized smootherstep blend already crossfades overlaps with
//! zero slope at both ends, which is all any upstream tiler does.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};

/// Configuration for tiled VAE decode.
///
/// Sizes are in **latent** coordinates. Default is 64 latent-px tile with
/// 16 latent-px overlap and 3-offset averaging — the same numbers as
/// ComfyUI's *OOM-fallback* `decode_tiled_` (`comfy/sd.py`), producing a
/// 512×512 image tile through an 8× VAE. This default exists for the
/// reactive OOM retry, where minimizing per-tile memory matters more than
/// decode count. Proactive tiling (a correctness cap, not an OOM) should use
/// [`proactive_tile_config`] instead, which sizes the fewest cap-fitting
/// tiles and runs a single pass.
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

pub use crate::vae_recovery::TiledMode;

/// Read `MOLD_VAE_TILED` once and resolve the process-wide [`TiledMode`].
/// Scheduler plans and execution share this same authority, so changing the
/// environment after admission cannot alter a granted generation.
///
/// Accepts `auto`, `force`, `off`, plus boolean-ish synonyms: `1`, `true`, `yes`
/// map to `Force`; `0`, `false`, `no` map to `Off`. Anything unrecognized
/// (including unset) returns [`TiledMode::Auto`].
pub fn resolve_mode() -> TiledMode {
    static CACHED: std::sync::OnceLock<TiledMode> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| parse_mode(crate::runtime_env::value("MOLD_VAE_TILED").as_deref()))
}

fn parse_mode(value: Option<&str>) -> TiledMode {
    match value.map(|s| s.trim().to_ascii_lowercase()).as_deref() {
        Some("force") | Some("1") | Some("true") | Some("yes") | Some("on") => TiledMode::Force,
        Some("off") | Some("0") | Some("false") | Some("no") => TiledMode::Off,
        _ => TiledMode::Auto,
    }
}

/// Detect an out-of-memory error from any GPU backend by string-matching the
/// underlying driver message.
///
/// candle today doesn't expose a typed OOM variant — every fallback ladder in
/// the codebase keys off the error text. This helper consolidates the known
/// substrings so we don't drift across engines or backends.
///
/// Metal is the reason this is not CUDA-only. Its exhaustion arrives as
/// `Insufficient Memory (…kIOGPUCommandBufferCallbackErrorOutOfMemory)`, which
/// shares no substring with the CUDA spellings, so a ladder that only knew the
/// CUDA strings silently turned a recoverable OOM into a hard failure on Apple
/// silicon.
pub fn is_out_of_memory_error(err: &impl std::fmt::Display) -> bool {
    let msg = err.to_string();
    msg.contains("OUT_OF_MEMORY")
        || msg.contains("out of memory")
        || msg.contains("OutOfMemory")
        || msg.contains("CUDA_ERROR_OUT_OF_MEMORY")
        || msg.contains("cudaErrorMemoryAllocation")
        || msg.contains("Insufficient Memory")
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
/// Tile configuration for a *proactive* tiled decode, where tiling exists to
/// stay under a measured per-axis correctness cap rather than to recover from
/// an OOM.
///
/// Sizes the tile with the fewest-tiles policy (the same one
/// `ltx2::tiling::axis_tiling` uses): split each axis into the minimum number
/// of tiles that all fit `cap`, then size the tile so the grid covers the
/// latent with the requested overlap. Single pass — the seam-cancellation
/// offset passes exist for OOM-fallback quality on tiny tiles and cost a
/// linear multiple of the whole decode; a 2×2 grid of near-cap tiles with a
/// normalized feathered blend does not need them.
pub(crate) fn proactive_tile_config(
    cap: usize,
    overlap: usize,
    lat_h: usize,
    lat_w: usize,
) -> TileConfig {
    debug_assert!(overlap < cap, "overlap {overlap} must be below cap {cap}");
    let lat = lat_h.max(lat_w);
    if lat <= cap || cap <= overlap {
        // Nothing to split (or a degenerate cap): one tile covering the
        // latent. decode_tiled short-circuits this into a single decode.
        return TileConfig {
            tile_size: cap.max(overlap + 1),
            overlap,
            offsets: 1,
        };
    }
    // Fewest tiles n whose every tile fits `cap` after accounting for the
    // shared overlap, then the smallest tile that still covers the axis:
    // n·tile - (n-1)·overlap >= lat. The closed form for n (not a naive
    // ceil(lat/cap)) is what keeps tile <= cap at every size — at lat 240,
    // 2 tiles would need tile 128 > cap.
    let n = (lat - overlap).div_ceil(cap - overlap);
    let tile_size = (lat + (n - 1) * overlap).div_ceil(n);
    debug_assert!(tile_size <= cap);
    TileConfig {
        tile_size,
        overlap,
        offsets: 1,
    }
}

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
    decode_with_fallible_oom_recovery(latents, decode_fn, || {
        on_oom_recover();
        Ok(())
    })
}

/// Whole decode followed by tiled recovery, with fallible device cleanup.
/// A repeated OOM from cleanup is consumed; unrelated cleanup errors propagate.
pub fn decode_with_fallible_oom_recovery<F, R>(
    latents: &Tensor,
    decode_fn: F,
    on_oom_recover: R,
) -> Result<Tensor>
where
    F: Fn(&Tensor) -> Result<Tensor>,
    R: FnOnce() -> Result<()>,
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
    }

    crate::vae_recovery::decode_with_recovery(
        mode,
        || {
            decode_fn(latents).inspect_err(|error| {
                if mode == TiledMode::Auto && is_out_of_memory_error(error) {
                    tracing::warn!(
                        error = %error,
                        tile_size = cfg.tile_size,
                        overlap = cfg.overlap,
                        offsets = cfg.offsets,
                        "VAE decode OOM — retrying with tiled decode"
                    );
                }
            })
        },
        || decode_tiled(latents, &decode_fn, &cfg),
        on_oom_recover,
        is_out_of_memory_error,
    )
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

/// Blend-ramp weight at position `i` (0-based, counted from the tile edge)
/// over a ramp of `ramp_len` cells.
///
/// Smootherstep `S(t) = 6t⁵ - 15t⁴ + 10t³` over `t = (i+1)/(ramp_len+1)` —
/// the `+1` denominator keeps opposing ramps exactly complementary
/// (`t_out = 1 - t_in` cell for cell, the same trick as
/// `ltx2::tiling::trapezoidal_mask`), so `S(t) + S(1-t) = 1` makes the
/// normalized blend a pure crossfade with zero slope at both ends of the
/// overlap. sd.cpp uses the same curve (`ggml_extend.hpp` smootherstep).
fn ramp_weight(i: usize, ramp_len: usize) -> f32 {
    let t = (i as f32 + 1.0) / (ramp_len as f32 + 1.0);
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

/// Build a feathered blend-weight buffer for one tile, output as a flat
/// `out_h * out_w` `f32` vector. Edges that touch the latent boundary get
/// weight 1.0; interior edges ramp from 0 → 1 over the overlap region in
/// image space via [`ramp_weight`]'s smootherstep.
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
                weights[row * out_w + col] *= ramp_weight(col, ramp_len);
            }
        }
    }
    if tile_y > 0 && out_overlap > 0 {
        let ramp_len = out_overlap.min(out_h);
        for row in 0..ramp_len {
            let factor = ramp_weight(row, ramp_len);
            for col in 0..out_w {
                weights[row * out_w + col] *= factor;
            }
        }
    }
    if tile_x + tile_w < lat_w && out_overlap > 0 {
        let ramp_len = out_overlap.min(out_w);
        for row in 0..out_h {
            for col in 0..ramp_len {
                weights[row * out_w + (out_w - 1 - col)] *= ramp_weight(col, ramp_len);
            }
        }
    }
    if tile_y + tile_h < lat_h && out_overlap > 0 {
        let ramp_len = out_overlap.min(out_h);
        for row in 0..ramp_len {
            let factor = ramp_weight(row, ramp_len);
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
    fn test_is_out_of_memory_error_matches_known_strings() {
        // Common driver strings the helper must recognize.
        assert!(is_out_of_memory_error(&"CUDA out of memory"));
        assert!(is_out_of_memory_error(&"CUDA_ERROR_OUT_OF_MEMORY"));
        assert!(is_out_of_memory_error(&"cudaErrorMemoryAllocation"));
        assert!(is_out_of_memory_error(&"OutOfMemory: ..."));
        assert!(is_out_of_memory_error(&"some prefix: out of memory: ..."));
        assert!(is_out_of_memory_error(&"OUT_OF_MEMORY: requested 5GB"));

        // Metal's exhaustion arrives with a different spelling entirely. It is
        // the message a Z-Image VAE decode actually fails with on Apple
        // silicon, and it has to reach the same CPU fallback ladder.
        assert!(is_out_of_memory_error(
            &"Metal error Command buffer had following error: Insufficient Memory \
              (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)"
        ));
        assert!(is_out_of_memory_error(&"Insufficient Memory"));

        // Negative cases.
        assert!(!is_out_of_memory_error(&"some other error"));
        assert!(!is_out_of_memory_error(&"shape mismatch"));
        assert!(!is_out_of_memory_error(&""));
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

    #[test]
    fn proactive_config_at_1024_is_a_2x2_single_pass() {
        // Latent 128 with a 120-per-axis cap must become a 2×2 grid of
        // 72-tiles with the standard 16 overlap and no offset passes: 4
        // decode calls, 1.27× the whole-decode work, versus the 27 calls /
        // 6.75× the previous 64/16/3 config cost.
        let cfg = proactive_tile_config(120, 16, 128, 128);
        assert_eq!(
            (cfg.tile_size, cfg.overlap, cfg.offsets),
            (72, 16, 1),
            "expected fewest-tiles sizing from the cap"
        );
        let starts = axis_starts(128, cfg.tile_size, cfg.tile_size - cfg.overlap, 0);
        assert_eq!(
            starts,
            vec![0, 56],
            "grid must be exactly two tiles per axis"
        );
    }

    #[test]
    fn proactive_config_tiles_fit_cap_and_cover_every_latent() {
        // The closed form must keep every tile at or under the cap and cover
        // the full axis at any latent size — a naive ceil(lat/cap) split
        // breaks at latent 240, where 2 tiles would need size 128 > cap.
        for lat in [121usize, 128, 160, 192, 240, 256, 320, 512] {
            let cfg = proactive_tile_config(120, 16, lat, lat);
            assert!(
                cfg.tile_size <= 120,
                "latent {lat}: tile {} exceeds the correctness cap",
                cfg.tile_size
            );
            assert!(
                cfg.overlap < cfg.tile_size,
                "latent {lat}: degenerate overlap"
            );
            assert_eq!(
                cfg.offsets, 1,
                "latent {lat}: proactive tiling is single-pass"
            );
            let starts = axis_starts(lat, cfg.tile_size, cfg.tile_size - cfg.overlap, 0);
            let last = *starts.last().unwrap();
            assert!(
                last + cfg.tile_size >= lat,
                "latent {lat}: grid ends at {} and leaves a gap",
                last + cfg.tile_size
            );
            // Adjacent tiles must genuinely overlap so the blend has a seam
            // region to feather.
            for pair in starts.windows(2) {
                assert!(
                    pair[0] + cfg.tile_size > pair[1],
                    "latent {lat}: tiles at {} and {} do not overlap",
                    pair[0],
                    pair[1]
                );
            }
        }
    }

    #[test]
    fn proactive_config_decodes_exactly_four_tiles_at_1024() {
        use std::cell::Cell;
        let latents = random_latent(16, 128, 128);
        let cfg = proactive_tile_config(120, 16, 128, 128);
        let calls = Cell::new(0usize);
        let counting_decode = |t: &Tensor| {
            calls.set(calls.get() + 1);
            synthetic_decode(t)
        };
        let full = synthetic_decode(&latents).unwrap();
        let tiled = decode_tiled(&latents, counting_decode, &cfg).unwrap();
        assert_eq!(calls.get(), 4, "2×2 single-pass grid must decode 4 tiles");
        let full_data: Vec<f32> = full.flatten_all().unwrap().to_vec1().unwrap();
        let tiled_data: Vec<f32> = tiled.flatten_all().unwrap().to_vec1().unwrap();
        let max_diff = full_data
            .iter()
            .zip(tiled_data.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            max_diff < 1e-3,
            "proactive tiled decode mismatch: {max_diff}"
        );
    }

    #[test]
    fn blend_ramps_normalize_to_a_smooth_crossfade() {
        // With two opposing ramps normalized by weight_acc, the effective
        // crossfade weight at ramp position t is S(t)/(S(t)+S(1-t)). For the
        // smootherstep ramp this is exactly S(t) (its symmetry identity), so
        // the crossfade has zero slope at both ends of the overlap — no C1
        // kink where a tile takes over, unlike a linear ramp.
        let overlap = 4usize;
        let scale = 8usize;
        let ramp = overlap * scale;
        // Right edge of a left tile vs left edge of a right tile, mid-row.
        let left = build_blend_weights_2d(0, 0, 8, 8, 12, 8, overlap, scale);
        let right = build_blend_weights_2d(4, 0, 8, 8, 12, 8, overlap, scale);
        // First cell of the fade-in ramp: smootherstep((col+1)/(ramp+1)) with
        // the LTX-2 trapezoid denominator (ramp+1, so opposing ramps are
        // exactly complementary). A linear ramp would put ~0.03 here;
        // smootherstep is an order of magnitude smaller and has zero slope.
        let first_in = right[0];
        let t = 1.0f32 / (ramp as f32 + 1.0);
        let s = t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
        assert!(
            (first_in - s).abs() < 1e-6,
            "expected smootherstep ramp start {s}, got {first_in}"
        );
        // Partition check at an interior fade position: the left tile's
        // fade-out and the right tile's fade-in must still sum to one so the
        // normalized blend is a pure crossfade.
        // Left tile spans x 0..8 (cols 0..64); right tile x 4..12 (cols 32..96).
        // Overlap in image space: cols 32..64 → left col 32+k, right col k.
        for k in 0..ramp {
            let sum = left[32 + k] + right[k];
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "ramp position {k}: weights sum to {sum}, not 1"
            );
        }
    }
}
