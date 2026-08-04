//! Latent-space tiling for LTX-2 stage-2 refinement.
//!
//! Upstream reaches resolutions the transformer never saw in training by
//! running stage 2 over overlapping latent tiles, each denoised at a shape the
//! model handles well, then blending them back with a separable trapezoidal
//! window. See `packages/ltx-core/src/ltx_core/tiling.py` and
//! `modality_tiling.py` in <https://github.com/Lightricks/LTX-2>, driven by
//! `packages/ltx-pipelines/src/ltx_pipelines/hdr_ic_lora.py` — the only
//! single-GPU reference for the technique.
//!
//! This module is the pure arithmetic: interval splitting, the blend window,
//! and tile enumeration. It has no tensor or device dependency so it can be
//! tested exhaustively on CPU, which matters because a silent desync between
//! the blend weights and the token order produces plausible-looking output
//! rather than an error.

use anyhow::{bail, Result};

/// Half-open `[start, end)` slice of one latent axis, plus the ramp widths
/// that fade it into its neighbours.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DimensionInterval {
    pub start: usize,
    pub end: usize,
    pub left_ramp: usize,
    pub right_ramp: usize,
}

impl DimensionInterval {
    pub fn len(self) -> usize {
        self.end - self.start
    }
}

/// Per-axis tile count and overlap, in **latent** cells.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DimensionTiling {
    pub num_tiles: usize,
    pub overlap: usize,
}

impl DimensionTiling {
    pub const fn new(num_tiles: usize, overlap: usize) -> Self {
        Self { num_tiles, overlap }
    }

    /// A single un-ramped tile — the axis is not split.
    pub const fn none() -> Self {
        Self {
            num_tiles: 1,
            overlap: 0,
        }
    }
}

/// Tile layout across the three latent axes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TileCountConfig {
    pub frames: DimensionTiling,
    pub height: DimensionTiling,
    pub width: DimensionTiling,
}

impl TileCountConfig {
    /// Upstream's shipped stage-2 layout: 2 tiles per axis, overlaps 8/6/6
    /// latent cells (`hdr_ic_lora.py:89-93`).
    pub const fn upstream_stage2() -> Self {
        Self {
            frames: DimensionTiling::new(2, 8),
            height: DimensionTiling::new(2, 6),
            width: DimensionTiling::new(2, 6),
        }
    }

    pub const fn untiled() -> Self {
        Self {
            frames: DimensionTiling::none(),
            height: DimensionTiling::none(),
            width: DimensionTiling::none(),
        }
    }
}

/// One tile: which latent cells it covers per axis, and its blend window.
#[derive(Debug, Clone)]
pub(crate) struct Tile {
    pub frames: DimensionInterval,
    pub height: DimensionInterval,
    pub width: DimensionInterval,
    /// Separable per-axis trapezoids, in tile-local order `(f, h, w)`.
    pub masks: [Vec<f32>; 3],
}

impl Tile {
    /// Flattened `(f, h, w)` row-major blend window — the product of the three
    /// per-axis trapezoids (`tiling.py:332-357`). Length matches this tile's
    /// token count, in the same order `patchify` emits.
    pub fn blend_window(&self) -> Vec<f32> {
        let [mf, mh, mw] = &self.masks;
        let mut out = Vec::with_capacity(mf.len() * mh.len() * mw.len());
        for f in mf {
            for h in mh {
                for w in mw {
                    out.push(f * h * w);
                }
            }
        }
        out
    }

    /// Indices of this tile's tokens within the full latent's flattened
    /// `(f, h, w)` token sequence. Strides come from the FULL latent, not the
    /// tile (`modality_tiling.py:207-211`).
    pub fn token_indices(&self, full_height: usize, full_width: usize) -> Vec<usize> {
        let mut out = Vec::with_capacity(self.token_count());
        for f in self.frames.start..self.frames.end {
            for h in self.height.start..self.height.end {
                for w in self.width.start..self.width.end {
                    out.push(f * full_height * full_width + h * full_width + w);
                }
            }
        }
        out
    }

    pub fn token_count(&self) -> usize {
        self.frames.len() * self.height.len() * self.width.len()
    }

    /// Pixel dimensions this tile is denoised at. The inverse of
    /// `VideoLatentShape::from_pixel_shape`: the causal VAE's first latent
    /// frame maps to a single pixel frame, so this is `(f - 1) * 8 + 1` and
    /// emphatically not `f * 8` (`hdr_ic_lora.py:548-550`).
    pub fn pixel_shape(&self) -> (usize, usize, usize) {
        (
            self.width.len() * LATENT_SPATIAL_STRIDE,
            self.height.len() * LATENT_SPATIAL_STRIDE,
            (self.frames.len() - 1) * LATENT_TEMPORAL_STRIDE + 1,
        )
    }
}

const LATENT_SPATIAL_STRIDE: usize = 32;
const LATENT_TEMPORAL_STRIDE: usize = 8;

/// Trapezoidal window for one axis (`tiling.py:11-47`).
///
/// The ramp values are `i / (ramp + 1)` for `i = 1..=ramp` — deliberately
/// never reaching 0 or 1, which is exactly what makes two adjacent ramps sum
/// to one across an overlap.
pub(crate) fn trapezoidal_mask(length: usize, left_ramp: usize, right_ramp: usize) -> Vec<f32> {
    let left = left_ramp.min(length);
    let right = right_ramp.min(length);
    let mut mask = vec![1.0f32; length];
    for (i, slot) in mask.iter_mut().take(left).enumerate() {
        *slot *= (i + 1) as f32 / (left + 1) as f32;
    }
    if right > 0 {
        let base = length - right;
        for (j, slot) in mask.iter_mut().skip(base).enumerate() {
            // Upstream multiplies, so a left and right ramp that overlap
            // compound rather than clamp.
            *slot *= 1.0 - (j + 1) as f32 / (right + 1) as f32;
        }
    }
    for value in &mut mask {
        *value = value.clamp(0.0, 1.0);
    }
    mask
}

/// Split an axis into `size`-long intervals overlapping by `overlap`
/// (`tiling.py:133-170`).
pub(crate) fn split_by_size(
    dimension_size: usize,
    size: usize,
    overlap: usize,
) -> Result<Vec<DimensionInterval>> {
    if size == 0 {
        bail!("tile size must be > 0");
    }
    if overlap >= size {
        bail!("overlap must satisfy 0 <= overlap < size, got overlap={overlap}, size={size}");
    }
    if dimension_size <= size {
        return Ok(vec![DimensionInterval {
            start: 0,
            end: dimension_size,
            left_ramp: 0,
            right_ramp: 0,
        }]);
    }
    let stride = size - overlap;
    let amount = (dimension_size + size - 2 * overlap - 1) / stride;
    let mut out = Vec::with_capacity(amount);
    out.push(DimensionInterval {
        start: 0,
        end: size,
        left_ramp: 0,
        right_ramp: overlap,
    });
    for i in 1..amount.saturating_sub(1) {
        out.push(DimensionInterval {
            start: i * stride,
            end: i * stride + size,
            left_ramp: overlap,
            right_ramp: overlap,
        });
    }
    if amount > 1 {
        out.push(DimensionInterval {
            start: (amount - 1) * stride,
            end: dimension_size,
            left_ramp: overlap,
            right_ramp: 0,
        });
    }
    Ok(out)
}

/// Split an axis into exactly `num_tiles` intervals (`tiling.py:246-291`).
pub(crate) fn split_by_count(
    dimension_size: usize,
    num_tiles: usize,
    overlap: usize,
) -> Result<Vec<DimensionInterval>> {
    if num_tiles == 0 {
        bail!("num_tiles must be >= 1");
    }
    if num_tiles > dimension_size {
        bail!("cannot split {dimension_size} cells into {num_tiles} tiles");
    }
    if num_tiles == 1 {
        return Ok(vec![DimensionInterval {
            start: 0,
            end: dimension_size,
            left_ramp: 0,
            right_ramp: 0,
        }]);
    }
    let total = dimension_size + overlap * (num_tiles - 1);
    let tile_size = total / num_tiles;
    let remainder = total % num_tiles;
    let base = split_by_size(dimension_size - remainder, tile_size, overlap)?;
    Ok(base
        .into_iter()
        .enumerate()
        .map(|(i, interval)| {
            let shift = i.min(remainder);
            let grow = usize::from(i < remainder);
            DimensionInterval {
                start: interval.start + shift,
                end: interval.end + shift + grow,
                ..interval
            }
        })
        .collect())
}

/// Shrink a requested layout to something this latent can actually be split
/// into (`hdr_ic_lora.py:100-142`), and additionally refuse a layout whose
/// windows would not sum to one.
pub(crate) fn clamp_dimension_tiling(
    cfg: DimensionTiling,
    dimension_size: usize,
) -> DimensionTiling {
    if cfg.num_tiles <= 1 {
        return DimensionTiling::none();
    }
    if dimension_size < cfg.num_tiles {
        return DimensionTiling::none();
    }
    // `split_by_count` needs `overlap < tile_size`, which reduces to this.
    let max_overlap = dimension_size - cfg.num_tiles;
    let mut overlap = cfg.overlap.min(max_overlap);

    // Upstream stops here, which is unsound for more than two tiles: once the
    // stride drops below the overlap, tiles i and i+2 also overlap, three
    // trapezoids stack, and the windows sum to as much as 1.16 — a silently
    // brightened seam, because `blend` has no normalization pass.
    //
    // No triple overlap requires `tile_size >= 2 * overlap`. Substituting
    // `tile_size = (dim + overlap * (n - 1)) / n` and solving gives
    // `overlap <= dim / (n + 1)`. Two tiles can never triple-overlap at all,
    // so upstream's shipped 2/8-6-6 layout is untouched.
    if cfg.num_tiles > 2 {
        overlap = overlap.min(dimension_size / (cfg.num_tiles + 1));
    }
    DimensionTiling {
        num_tiles: cfg.num_tiles,
        overlap,
    }
}

/// Enumerate tiles in `(frames, height, width)` row-major order.
///
/// The order is load-bearing: it defines each tile's index, which upstream
/// uses to seed that tile's noise (`hdr_ic_lora.py:547`).
pub(crate) fn create_tiles(
    latent_frames: usize,
    latent_height: usize,
    latent_width: usize,
    cfg: TileCountConfig,
) -> Result<Vec<Tile>> {
    let frames = clamp_dimension_tiling(cfg.frames, latent_frames);
    let height = clamp_dimension_tiling(cfg.height, latent_height);
    let width = clamp_dimension_tiling(cfg.width, latent_width);

    let f_intervals = split_by_count(latent_frames, frames.num_tiles, frames.overlap)?;
    let h_intervals = split_by_count(latent_height, height.num_tiles, height.overlap)?;
    let w_intervals = split_by_count(latent_width, width.num_tiles, width.overlap)?;

    let mut tiles = Vec::with_capacity(f_intervals.len() * h_intervals.len() * w_intervals.len());
    for f in &f_intervals {
        for h in &h_intervals {
            for w in &w_intervals {
                tiles.push(Tile {
                    frames: *f,
                    height: *h,
                    width: *w,
                    masks: [
                        trapezoidal_mask(f.len(), f.left_ramp, f.right_ramp),
                        trapezoidal_mask(h.len(), h.left_ramp, h.right_ramp),
                        trapezoidal_mask(w.len(), w.left_ramp, w.right_ramp),
                    ],
                });
            }
        }
    }
    Ok(tiles)
}

/// Per-axis span, in latent cells, that the checkpoints' RoPE was trained on.
///
/// `positional_embedding_max_pos = [20, 2048, 2048]` in pixels, and the video
/// VAE compresses space by 32, so the spatial axes are trained over 64 latent
/// cells. Past that, positions land outside the trained range.
pub(crate) const TRAINED_SPATIAL_LATENT_SPAN: usize = 2_048 / LATENT_SPATIAL_STRIDE;

/// Choose a tile layout for a stage-2 latent.
///
/// Returns [`TileCountConfig::untiled`] when the shape is already inside the
/// span the model was trained on — tiling costs a full denoise pass per tile,
/// so it must not be paid unless it buys something. Otherwise split each
/// oversized axis into just enough tiles to bring every tile back inside the
/// span, using upstream's shipped overlaps.
pub(crate) fn plan_stage2_tiling(
    latent_frames: usize,
    latent_height: usize,
    latent_width: usize,
) -> TileCountConfig {
    let axis = |size: usize, overlap: usize| -> DimensionTiling {
        if size <= TRAINED_SPATIAL_LATENT_SPAN {
            return DimensionTiling::none();
        }
        // Ceiling division: the smallest tile count whose tiles fit the span.
        let tiles = size.div_ceil(TRAINED_SPATIAL_LATENT_SPAN).max(2);
        clamp_dimension_tiling(DimensionTiling::new(tiles, overlap), size)
    };

    let upstream = TileCountConfig::upstream_stage2();
    let planned = TileCountConfig {
        // Time is not the axis that blows past the trained span at high
        // resolution — the duration budget already caps it — so frames stay
        // whole and only space is tiled.
        frames: DimensionTiling::none(),
        height: axis(latent_height, upstream.height.overlap),
        width: axis(latent_width, upstream.width.overlap),
    };
    let _ = latent_frames;
    planned
}

// ── Policy: when does tiling engage at all? ──────────────────────────────────
//
// Tiling costs a full denoise pass per tile and blends across seams, so it must
// never touch a shape that renders correctly today. Both consumers — stage-2
// refinement and the VAE decode — resolve one `SpatialTilePolicy`, and `Auto`
// is deliberately conservative enough that it cannot fire on any shape
// `mold_core::validation::LTX2_MAX_AXIS_PIXELS` currently admits.

/// Default VAE spatial decode tile, in pixels. Upstream ships 768/64
/// (`video_vae/tiling.py:67`); mold decodes larger tiles because its
/// consumer-GPU target has already paid for a spatially tiled *denoise* by the
/// time it gets here, and a wider tile means fewer decoder passes.
pub(crate) const DEFAULT_VAE_SPATIAL_TILE_PIXELS: usize = 1_280;

/// Default overlap between VAE decode tiles, in pixels. The decoder pads at
/// tile edges instead of seeing real neighbours, so upstream discards a latent
/// cell per edge and requires at least 64 px of overlap
/// (`video_vae.py:437-446`). 256 px is 8 latent cells per side — enough that
/// the trapezoid's ramp, not the padded edge, is what reaches the seam.
pub(crate) const DEFAULT_VAE_SPATIAL_OVERLAP_PIXELS: usize = 256;

/// The pixel stride of one latent cell, re-exported for callers that reason in
/// pixels (the knob's unit) rather than latent cells (the tiler's unit).
pub(crate) const LATENT_PIXEL_STRIDE: usize = LATENT_SPATIAL_STRIDE;

/// How spatial work may be split, resolved from `MOLD_LTX2_SPATIAL_TILE`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SpatialTilePolicy {
    /// Never tile. Exactly today's behaviour at every resolution, including
    /// the out-of-distribution ones.
    Off,
    /// Tile only where it buys something: stage 2 past the trained span, and
    /// a VAE decode that both exceeds that span and does not fit.
    Auto,
    /// Always tile at this size, whatever the shape. This is the only way to
    /// exercise tiling at a resolution that does not need it, which is what
    /// makes tiled-vs-untiled equivalence testable on one consumer GPU.
    Forced {
        tile_pixels: usize,
        overlap_pixels: usize,
    },
}

impl SpatialTilePolicy {
    /// Parse the `MOLD_LTX2_SPATIAL_TILE` / `--spatial-tile` value:
    /// `off`, `auto`, `<pixels>`, or `<pixels>:<overlap>`.
    pub(crate) fn parse(value: &str) -> Result<Self> {
        let value = value.trim();
        match value.to_ascii_lowercase().as_str() {
            "" | "auto" => return Ok(Self::Auto),
            "off" | "none" | "0" | "false" => return Ok(Self::Off),
            _ => {}
        }
        let (size, overlap) = match value.split_once(':') {
            Some((size, overlap)) => (size, Some(overlap)),
            None => (value, None),
        };
        let tile_pixels: usize = size.trim().parse().map_err(|_| {
            anyhow::anyhow!(
                "invalid spatial tile size '{size}'; expected off, auto, <pixels>, or <pixels>:<overlap>"
            )
        })?;
        let overlap_pixels = match overlap {
            Some(overlap) => overlap.trim().parse().map_err(|_| {
                anyhow::anyhow!("invalid spatial tile overlap '{overlap}'; expected a pixel count")
            })?,
            None => DEFAULT_VAE_SPATIAL_OVERLAP_PIXELS,
        };
        // Upstream's own validation (`video_vae/tiling.py:15-25`): a tile has
        // to be a whole number of latent cells, and an overlap that reaches
        // the tile size leaves no stride at all.
        if tile_pixels < 2 * LATENT_PIXEL_STRIDE {
            bail!(
                "spatial tile size must be at least {} px",
                2 * LATENT_PIXEL_STRIDE
            );
        }
        if !tile_pixels.is_multiple_of(LATENT_PIXEL_STRIDE) {
            bail!("spatial tile size must be a multiple of {LATENT_PIXEL_STRIDE} px, got {tile_pixels}");
        }
        if !overlap_pixels.is_multiple_of(LATENT_PIXEL_STRIDE) {
            bail!("spatial tile overlap must be a multiple of {LATENT_PIXEL_STRIDE} px, got {overlap_pixels}");
        }
        if overlap_pixels >= tile_pixels {
            bail!("spatial tile overlap ({overlap_pixels} px) must be smaller than the tile ({tile_pixels} px)");
        }
        Ok(Self::Forced {
            tile_pixels,
            overlap_pixels,
        })
    }
}

/// Split one axis into the fewest tiles whose every tile fits `span` cells.
fn axis_tiling(size: usize, span: usize, overlap: usize) -> DimensionTiling {
    if span == 0 || size <= span {
        return DimensionTiling::none();
    }
    let tiles = size.div_ceil(span).max(2);
    clamp_dimension_tiling(DimensionTiling::new(tiles, overlap), size)
}

/// Choose a stage-2 tile layout under `policy`.
///
/// `Auto` reproduces [`plan_stage2_tiling`]: split only what the checkpoints
/// were never trained to see. `Forced` splits to the requested tile size even
/// when the shape does not need it.
pub(crate) fn plan_stage2_tiling_with_policy(
    latent_frames: usize,
    latent_height: usize,
    latent_width: usize,
    policy: SpatialTilePolicy,
) -> TileCountConfig {
    match policy {
        SpatialTilePolicy::Off => TileCountConfig::untiled(),
        SpatialTilePolicy::Auto => plan_stage2_tiling(latent_frames, latent_height, latent_width),
        SpatialTilePolicy::Forced {
            tile_pixels,
            overlap_pixels,
        } => {
            let span = tile_pixels / LATENT_PIXEL_STRIDE;
            let overlap = overlap_pixels / LATENT_PIXEL_STRIDE;
            TileCountConfig {
                // Time is never split: the duration budget already caps it,
                // and a temporal seam is far more visible than a spatial one.
                frames: DimensionTiling::none(),
                height: axis_tiling(latent_height, span, overlap),
                width: axis_tiling(latent_width, span, overlap),
            }
        }
    }
}

/// Pixel-space tile geometry for the VAE decode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SpatialDecodeTiling {
    /// Latent cells per tile along each spatial axis.
    pub tile_cells: usize,
    /// Latent cells shared with each neighbour.
    pub overlap_cells: usize,
}

/// Choose a VAE spatial decode layout.
///
/// `needed` is the caller's memory verdict — the same projected-workspace
/// check that already selects temporal chunking. `Auto` requires *both* that
/// verdict and a frame past the trained span, so a 1080p or 2K decode that
/// chunks temporally today keeps decoding exactly as it does today.
pub(crate) fn plan_spatial_decode_tiling(
    latent_height: usize,
    latent_width: usize,
    policy: SpatialTilePolicy,
    needed: bool,
) -> Option<SpatialDecodeTiling> {
    let (tile_cells, overlap_cells) = match policy {
        SpatialTilePolicy::Off => return None,
        SpatialTilePolicy::Auto => {
            let past_trained_span = latent_height.max(latent_width) > TRAINED_SPATIAL_LATENT_SPAN;
            if !needed || !past_trained_span {
                return None;
            }
            (
                DEFAULT_VAE_SPATIAL_TILE_PIXELS / LATENT_PIXEL_STRIDE,
                DEFAULT_VAE_SPATIAL_OVERLAP_PIXELS / LATENT_PIXEL_STRIDE,
            )
        }
        SpatialTilePolicy::Forced {
            tile_pixels,
            overlap_pixels,
        } => (
            tile_pixels / LATENT_PIXEL_STRIDE,
            overlap_pixels / LATENT_PIXEL_STRIDE,
        ),
    };
    if latent_height <= tile_cells && latent_width <= tile_cells {
        return None;
    }
    Some(SpatialDecodeTiling {
        tile_cells,
        overlap_cells: overlap_cells.min(tile_cells.saturating_sub(1)),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Expected values were produced by running upstream's own
    /// `split_by_count` in its virtualenv, so this pins the port to the
    /// reference rather than to my reading of it.
    #[test]
    fn split_by_count_matches_upstream() {
        /// `(start, end, left_ramp, right_ramp)` as upstream reports it.
        type ExpectedInterval = (usize, usize, usize, usize);
        /// `(dimension, tiles, overlap, expected intervals)`.
        type SplitCase = (usize, usize, usize, &'static [ExpectedInterval]);

        let cases: &[SplitCase] = &[
            (21, 2, 8, &[(0, 15, 0, 8), (7, 21, 8, 0)]),
            (34, 2, 6, &[(0, 20, 0, 6), (14, 34, 6, 0)]),
            (60, 2, 6, &[(0, 33, 0, 6), (27, 60, 6, 0)]),
            (10, 2, 8, &[(0, 9, 0, 8), (1, 10, 8, 0)]),
            (
                20,
                4,
                2,
                &[(0, 7, 0, 2), (5, 12, 2, 2), (10, 16, 2, 2), (14, 20, 2, 0)],
            ),
        ];
        for &(dim, n, overlap, expected) in cases {
            let got: Vec<_> = split_by_count(dim, n, overlap)
                .unwrap()
                .into_iter()
                .map(|i| (i.start, i.end, i.left_ramp, i.right_ramp))
                .collect();
            assert_eq!(got, expected, "split_by_count({dim}, {n}, {overlap})");
        }
    }

    #[test]
    fn split_by_count_rejects_impossible_layouts() {
        // tile_size == overlap == 8
        assert!(split_by_count(9, 2, 8).is_err());
        assert!(split_by_count(4, 5, 0).is_err());
        assert_eq!(split_by_count(12, 1, 0).unwrap().len(), 1);
    }

    #[test]
    fn trapezoid_ramps_never_reach_zero_or_one() {
        let mask = trapezoidal_mask(15, 0, 8);
        assert_eq!(&mask[..7], &[1.0; 7]);
        for (i, value) in mask[7..].iter().enumerate() {
            let expected = 1.0 - (i + 1) as f32 / 9.0;
            assert!((value - expected).abs() < 1e-6, "slot {i}: {value}");
        }
        assert!(mask.iter().all(|v| *v > 0.0 && *v <= 1.0));
    }

    /// The whole technique rests on this: `blend` has no normalization pass,
    /// so if the windows do not sum to one the seam is silently wrong.
    fn assert_partition_of_unity(dim: usize, n: usize, overlap: usize) {
        let intervals = split_by_count(dim, n, overlap).unwrap();
        let mut acc = vec![0.0f32; dim];
        for interval in &intervals {
            let mask = trapezoidal_mask(interval.len(), interval.left_ramp, interval.right_ramp);
            for (offset, weight) in mask.iter().enumerate() {
                acc[interval.start + offset] += weight;
            }
        }
        for (i, total) in acc.iter().enumerate() {
            assert!(
                (total - 1.0).abs() < 1e-6,
                "cell {i} of split({dim}, {n}, {overlap}) sums to {total}, not 1"
            );
        }
    }

    #[test]
    fn blend_windows_sum_to_one_across_tiles() {
        for &(dim, n, overlap) in &[
            (21, 2, 8),
            (34, 2, 6),
            (60, 2, 6),
            (21, 3, 4),
            (20, 4, 2),
            (9, 2, 7),
            (12, 2, 0),
        ] {
            assert_partition_of_unity(dim, n, overlap);
        }
    }

    /// Upstream clamps only enough to avoid a `ValueError`, so a three-tile
    /// layout with a wide overlap triple-stacks and sums to ~1.07. Our clamp
    /// narrows the overlap instead of shipping a brightened seam.
    #[test]
    fn wide_overlaps_are_narrowed_rather_than_triple_stacked() {
        for &(dim, n, requested) in &[(21, 3, 8), (21, 4, 8), (34, 3, 12)] {
            let clamped = clamp_dimension_tiling(DimensionTiling::new(n, requested), dim);
            assert!(clamped.overlap <= requested);
            assert_partition_of_unity(dim, clamped.num_tiles, clamped.overlap);
        }
        // Two tiles can never triple-overlap, so the shipped layout is
        // untouched.
        assert_eq!(
            clamp_dimension_tiling(DimensionTiling::new(2, 8), 21),
            DimensionTiling::new(2, 8)
        );
    }

    #[test]
    fn clamping_degrades_to_a_single_tile_when_the_axis_is_too_small() {
        assert_eq!(
            clamp_dimension_tiling(DimensionTiling::new(4, 2), 3),
            DimensionTiling::none()
        );
        assert_eq!(
            clamp_dimension_tiling(DimensionTiling::new(2, 99), 10),
            DimensionTiling::new(2, 8)
        );
    }

    /// Tile order defines each tile's noise seed upstream, so it is part of
    /// the contract, not an implementation detail.
    /// Fixture uses 2 tiles / overlap 2 per axis, matching the layout the
    /// upstream numbers below were captured with.
    fn two_by_two() -> TileCountConfig {
        TileCountConfig {
            frames: DimensionTiling::new(2, 2),
            height: DimensionTiling::new(2, 2),
            width: DimensionTiling::new(2, 2),
        }
    }

    #[test]
    fn tiles_enumerate_frames_then_height_then_width() {
        let tiles = create_tiles(9, 8, 10, two_by_two()).unwrap();
        let coords: Vec<_> = tiles
            .iter()
            .map(|t| {
                (
                    (t.frames.start, t.frames.end),
                    (t.height.start, t.height.end),
                    (t.width.start, t.width.end),
                )
            })
            .collect();
        assert_eq!(coords.len(), 8);
        assert_eq!(coords[0], ((0, 6), (0, 5), (0, 6)));
        assert_eq!(coords[1], ((0, 6), (0, 5), (4, 10)));
        assert_eq!(coords[2], ((0, 6), (3, 8), (0, 6)));
        assert_eq!(coords[3], ((0, 6), (3, 8), (4, 10)));
        assert_eq!(coords[4], ((4, 9), (0, 5), (0, 6)));
    }

    /// A tile's window must line up cell-for-cell with the tokens it covers,
    /// in the same `(f, h, w)` order `patchify` emits. A transpose here would
    /// desync the two silently.
    #[test]
    fn blend_windows_reconstruct_the_full_field() {
        let (frames, height, width) = (9usize, 8usize, 10usize);
        let tiles = create_tiles(frames, height, width, two_by_two()).unwrap();
        let mut acc = vec![0.0f32; frames * height * width];
        for tile in &tiles {
            let window = tile.blend_window();
            let indices = tile.token_indices(height, width);
            assert_eq!(window.len(), indices.len());
            assert_eq!(window.len(), tile.token_count());
            for (token, weight) in indices.iter().zip(window) {
                acc[*token] += weight;
            }
        }
        for (token, total) in acc.iter().enumerate() {
            assert!(
                (total - 1.0).abs() < 1e-5,
                "token {token} accumulated {total}, not 1"
            );
        }
    }

    #[test]
    fn tile_pixel_shape_inverts_the_latent_grid() {
        let tiles = create_tiles(9, 8, 10, two_by_two()).unwrap();
        // First tile is 6 latent frames x 5 rows x 6 cols.
        let (w, h, f) = tiles[0].pixel_shape();
        // 6 latent frames -> (6 - 1) * 8 + 1 = 41 pixel frames, not 48.
        assert_eq!((w, h, f), (6 * 32, 5 * 32, 41));
    }

    /// Tiling costs a full denoise pass per tile, so a shape the model was
    /// trained on must not pay for it.
    #[test]
    fn shapes_inside_the_trained_span_are_not_tiled() {
        // 1216x704 -> 38x22 latent cells: the LTX-2 default, comfortably inside.
        assert_eq!(plan_stage2_tiling(13, 22, 38), TileCountConfig::untiled());
        // 1920x1088 -> 60x34: still inside, which is why #668 could ship it
        // without tiling.
        assert_eq!(plan_stage2_tiling(13, 34, 60), TileCountConfig::untiled());
        // Exactly at the span.
        assert_eq!(plan_stage2_tiling(13, 64, 64), TileCountConfig::untiled());
    }

    /// Every tile of a 4K plan must land back inside the trained span, or
    /// tiling has not bought anything.
    #[test]
    fn oversized_axes_are_split_until_every_tile_fits_the_trained_span() {
        // 3840x2160 -> 120x68 latent cells.
        let cfg = plan_stage2_tiling(13, 68, 120);
        assert_ne!(cfg, TileCountConfig::untiled());
        assert_eq!(cfg.frames, DimensionTiling::none(), "time stays whole");

        let tiles = create_tiles(13, 68, 120, cfg).unwrap();
        for tile in &tiles {
            assert!(
                tile.height.len() <= TRAINED_SPATIAL_LATENT_SPAN,
                "tile is {} cells tall, past the trained span",
                tile.height.len()
            );
            assert!(
                tile.width.len() <= TRAINED_SPATIAL_LATENT_SPAN,
                "tile is {} cells wide, past the trained span",
                tile.width.len()
            );
        }

        // And the windows still reconstruct the field exactly.
        let mut acc = vec![0.0f32; 13 * 68 * 120];
        for tile in &tiles {
            for (token, weight) in tile.token_indices(68, 120).iter().zip(tile.blend_window()) {
                acc[*token] += weight;
            }
        }
        assert!(acc.iter().all(|total| (total - 1.0).abs() < 1e-5));
    }

    #[test]
    fn an_untiled_layout_yields_one_full_cover_tile() {
        let tiles = create_tiles(9, 8, 10, TileCountConfig::untiled()).unwrap();
        assert_eq!(tiles.len(), 1);
        assert_eq!(tiles[0].token_count(), 9 * 8 * 10);
        assert!(tiles[0].blend_window().iter().all(|w| *w == 1.0));
    }

    // ── Policy ───────────────────────────────────────────────────────────

    #[test]
    fn policy_parses_off_auto_and_explicit_sizes() {
        assert_eq!(
            SpatialTilePolicy::parse("").unwrap(),
            SpatialTilePolicy::Auto
        );
        assert_eq!(
            SpatialTilePolicy::parse(" AUTO ").unwrap(),
            SpatialTilePolicy::Auto
        );
        for off in ["off", "none", "0", "false", "Off"] {
            assert_eq!(
                SpatialTilePolicy::parse(off).unwrap(),
                SpatialTilePolicy::Off
            );
        }
        assert_eq!(
            SpatialTilePolicy::parse("1280").unwrap(),
            SpatialTilePolicy::Forced {
                tile_pixels: 1_280,
                overlap_pixels: DEFAULT_VAE_SPATIAL_OVERLAP_PIXELS,
            }
        );
        assert_eq!(
            SpatialTilePolicy::parse("256:64").unwrap(),
            SpatialTilePolicy::Forced {
                tile_pixels: 256,
                overlap_pixels: 64,
            }
        );
    }

    #[test]
    fn policy_rejects_geometry_the_tiler_cannot_honour() {
        // Not a whole number of latent cells.
        assert!(SpatialTilePolicy::parse("1000").is_err());
        assert!(SpatialTilePolicy::parse("1280:100").is_err());
        // Smaller than two latent cells, so it can never have a ramp.
        assert!(SpatialTilePolicy::parse("32").is_err());
        // Overlap swallows the stride.
        assert!(SpatialTilePolicy::parse("256:256").is_err());
        assert!(SpatialTilePolicy::parse("garbage").is_err());
    }

    /// The compatibility constraint that matters more than anything else here:
    /// no shape that renders today may start tiling because this landed.
    #[test]
    fn auto_never_tiles_a_shape_the_validator_admits_today() {
        // `LTX2_MAX_AXIS_PIXELS` is 2048, so 64 latent cells is the largest
        // axis that can reach the engine. Walk every 32-px rung up to it.
        for cells in 1..=TRAINED_SPATIAL_LATENT_SPAN {
            assert_eq!(
                plan_stage2_tiling_with_policy(13, cells, cells, SpatialTilePolicy::Auto),
                TileCountConfig::untiled(),
                "{cells} latent cells is inside the trained span and must not tile"
            );
            assert_eq!(
                plan_spatial_decode_tiling(cells, cells, SpatialTilePolicy::Auto, true),
                None,
                "{cells} latent cells must decode exactly as it does today"
            );
        }
    }

    #[test]
    fn off_disables_tiling_even_past_the_trained_span() {
        // 3840x2160.
        assert_eq!(
            plan_stage2_tiling_with_policy(13, 68, 120, SpatialTilePolicy::Off),
            TileCountConfig::untiled()
        );
        assert_eq!(
            plan_spatial_decode_tiling(68, 120, SpatialTilePolicy::Off, true),
            None
        );
    }

    #[test]
    fn forced_tiling_splits_a_shape_that_would_not_need_it() {
        // 512x512 -> 16x16 latent cells, forced into 256-px (8-cell) tiles.
        let cfg =
            plan_stage2_tiling_with_policy(4, 16, 16, SpatialTilePolicy::parse("256:64").unwrap());
        assert_eq!(cfg.frames, DimensionTiling::none(), "time is never split");
        assert_eq!(cfg.height.num_tiles, 2);
        assert_eq!(cfg.width.num_tiles, 2);

        let tiles = create_tiles(4, 16, 16, cfg).unwrap();
        assert_eq!(tiles.len(), 4);
        let mut acc = vec![0.0f32; 4 * 16 * 16];
        for tile in &tiles {
            for (token, weight) in tile.token_indices(16, 16).iter().zip(tile.blend_window()) {
                acc[*token] += weight;
            }
        }
        assert!(acc.iter().all(|total| (total - 1.0).abs() < 1e-5));
    }

    /// `Ltx2OutputRung` promises clients a stage-1 shape and a tile count for
    /// every advertised rung, and `mold-core` cannot see the engine that has to
    /// produce them. Pin the two together here: a rung that names arithmetic
    /// the engine does not perform is a lie told at picker resolution.
    #[test]
    fn advertised_rungs_match_the_engines_own_arithmetic() {
        use crate::ltx2::model::derive_stage1_render_shape;
        use mold_core::validation::LTX2_OUTPUT_RUNGS;
        use mold_core::Ltx2SpatialUpscale;

        for rung in LTX2_OUTPUT_RUNGS {
            let engine_stage1 = derive_stage1_render_shape(
                rung.width,
                rung.height,
                25,
                24,
                Some(Ltx2SpatialUpscale::X2),
                None,
            );
            assert_eq!(
                rung.stage1_shape(),
                (engine_stage1.width, engine_stage1.height),
                "{} stage-1 shape must be what the engine renders",
                rung.id
            );

            // Stage 2 refines the target's own latent grid.
            let latent_width = rung.width as usize / LATENT_SPATIAL_STRIDE;
            let latent_height = rung.height as usize / LATENT_SPATIAL_STRIDE;
            let planned = plan_stage2_tiling(13, latent_height, latent_width);
            assert_eq!(
                rung.stage2_tiles(),
                (
                    planned.width.num_tiles as u32,
                    planned.height.num_tiles as u32
                ),
                "{} tile counts must be what the tiler plans",
                rung.id
            );
            assert_eq!(
                rung.requires_tiled_stage2(),
                planned != TileCountConfig::untiled(),
                "{} must agree with the tiler on whether it needs tiling at all",
                rung.id
            );
        }
    }

    #[test]
    fn a_decode_that_fits_is_not_tiled_even_at_4k() {
        // 3840x2160 latent cells, but the caller says the workspace fits.
        assert_eq!(
            plan_spatial_decode_tiling(68, 120, SpatialTilePolicy::Auto, false),
            None
        );
        assert_eq!(
            plan_spatial_decode_tiling(68, 120, SpatialTilePolicy::Auto, true),
            Some(SpatialDecodeTiling {
                tile_cells: DEFAULT_VAE_SPATIAL_TILE_PIXELS / LATENT_PIXEL_STRIDE,
                overlap_cells: DEFAULT_VAE_SPATIAL_OVERLAP_PIXELS / LATENT_PIXEL_STRIDE,
            })
        );
    }
}
