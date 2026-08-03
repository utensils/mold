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

    pub fn is_untiled(self) -> bool {
        self.frames.num_tiles <= 1 && self.height.num_tiles <= 1 && self.width.num_tiles <= 1
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
fn split_by_size(
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Expected values were produced by running upstream's own
    /// `split_by_count` in its virtualenv, so this pins the port to the
    /// reference rather than to my reading of it.
    #[test]
    fn split_by_count_matches_upstream() {
        let cases: &[(usize, usize, usize, &[(usize, usize, usize, usize)])] = &[
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
        assert!(plan_stage2_tiling(13, 22, 38).is_untiled());
        // 1920x1088 -> 60x34: still inside, which is why #668 could ship it
        // without tiling.
        assert!(plan_stage2_tiling(13, 34, 60).is_untiled());
        // Exactly at the span.
        assert!(plan_stage2_tiling(13, 64, 64).is_untiled());
    }

    /// Every tile of a 4K plan must land back inside the trained span, or
    /// tiling has not bought anything.
    #[test]
    fn oversized_axes_are_split_until_every_tile_fits_the_trained_span() {
        // 3840x2160 -> 120x68 latent cells.
        let cfg = plan_stage2_tiling(13, 68, 120);
        assert!(!cfg.is_untiled());
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
}
