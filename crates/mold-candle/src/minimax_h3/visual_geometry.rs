use candle::{bail, Result};

/// The released H3 visual VAE's temporal chunk contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VisualTemporalGeometry {
    pub clip_length: usize,
    pub token_drop: usize,
    pub temporal_compression: usize,
}

impl Default for VisualTemporalGeometry {
    fn default() -> Self {
        Self {
            clip_length: 17,
            token_drop: 3,
            temporal_compression: 4,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TemporalEncodeChunk {
    pub input_start: usize,
    pub valid_frames: usize,
    pub repeated_tail_frames: usize,
}

/// A bounded-memory encode plan. Chunks are independent causal evaluations;
/// no activation from one 17-frame chunk is carried into the next.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TemporalEncodePlan {
    pub chunks: Vec<TemporalEncodeChunk>,
    pub latent_frames_before_drop: usize,
    pub latent_frames: usize,
    pub trailing_latents_to_drop: usize,
    pub single_frame: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TemporalDecodeChunk {
    pub latent_start: usize,
    pub latent_frames: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TemporalDecodePlan {
    pub chunks: Vec<TemporalDecodeChunk>,
    pub repeated_tail_tokens: usize,
    pub output_frames: usize,
    pub tokens_per_chunk: usize,
    pub token_overlap: usize,
    pub frame_pre_padding: usize,
    pub frame_overlap: usize,
}

impl VisualTemporalGeometry {
    pub fn validate(self) -> Result<()> {
        if self.clip_length == 0 || self.temporal_compression == 0 {
            bail!("H3 visual VAE clip length and temporal compression must be positive")
        }
        let tokens = self.tokens_per_chunk();
        if self.token_drop >= tokens {
            bail!(
                "H3 visual VAE token_drop {} must be smaller than tokens_per_chunk {}",
                self.token_drop,
                tokens
            )
        }
        Ok(())
    }

    pub const fn tokens_per_chunk(self) -> usize {
        self.clip_length.div_ceil(self.temporal_compression)
    }

    pub const fn frame_pre_padding(self) -> usize {
        (self.temporal_compression - self.clip_length % self.temporal_compression)
            % self.temporal_compression
    }

    pub const fn token_overlap(self) -> usize {
        (self.tokens_per_chunk() - self.token_drop % self.tokens_per_chunk())
            % self.tokens_per_chunk()
    }

    pub const fn frame_overlap(self) -> usize {
        self.token_overlap()
            .saturating_mul(self.temporal_compression)
            .saturating_sub(self.frame_pre_padding())
    }

    pub fn encode_plan(self, pixel_frames: usize) -> Result<TemporalEncodePlan> {
        self.validate()?;
        if pixel_frames == 0 {
            bail!("H3 visual VAE cannot encode an empty frame sequence")
        }
        if pixel_frames == 1 {
            return Ok(TemporalEncodePlan {
                chunks: vec![TemporalEncodeChunk {
                    input_start: 0,
                    valid_frames: 1,
                    repeated_tail_frames: 0,
                }],
                latent_frames_before_drop: 1,
                latent_frames: 1,
                trailing_latents_to_drop: 0,
                single_frame: true,
            });
        }

        let num_chunks = pixel_frames.div_ceil(self.clip_length);
        let mut chunks = Vec::with_capacity(num_chunks);
        for index in 0..num_chunks {
            let input_start = index * self.clip_length;
            let valid_frames = (pixel_frames - input_start).min(self.clip_length);
            chunks.push(TemporalEncodeChunk {
                input_start,
                valid_frames,
                repeated_tail_frames: self.clip_length - valid_frames,
            });
        }
        let before_drop = num_chunks
            .checked_mul(self.tokens_per_chunk())
            .ok_or_else(|| candle::Error::Msg("H3 encoded frame count overflow".into()))?;
        Ok(TemporalEncodePlan {
            chunks,
            latent_frames_before_drop: before_drop,
            latent_frames: before_drop - self.token_drop,
            trailing_latents_to_drop: self.token_drop,
            single_frame: false,
        })
    }

    pub fn encoded_frames(self, pixel_frames: usize) -> Result<usize> {
        Ok(self.encode_plan(pixel_frames)?.latent_frames)
    }

    pub fn decode_plan(self, latent_frames: usize) -> Result<TemporalDecodePlan> {
        self.validate()?;
        if latent_frames == 0 {
            bail!("H3 visual VAE cannot decode an empty latent sequence")
        }
        if latent_frames == 1 {
            return Ok(TemporalDecodePlan {
                chunks: vec![TemporalDecodeChunk {
                    latent_start: 0,
                    latent_frames: 1,
                }],
                repeated_tail_tokens: 0,
                output_frames: 1,
                tokens_per_chunk: self.tokens_per_chunk(),
                token_overlap: self.token_overlap(),
                frame_pre_padding: self.frame_pre_padding(),
                frame_overlap: self.frame_overlap(),
            });
        }

        let tokens_per_chunk = self.tokens_per_chunk();
        let mut pseudo_total_tokens = latent_frames
            .checked_add(self.token_drop)
            .ok_or_else(|| candle::Error::Msg("H3 decoded token count overflow".into()))?;
        let mut pad_tokens = 0;
        let remainder = pseudo_total_tokens % tokens_per_chunk;
        if remainder != 0 {
            pad_tokens = tokens_per_chunk - remainder;
            pseudo_total_tokens = pseudo_total_tokens
                .checked_add(pad_tokens)
                .ok_or_else(|| candle::Error::Msg("H3 padded token count overflow".into()))?;
        }
        let mut num_chunks =
            pseudo_total_tokens / tokens_per_chunk - usize::from(self.token_drop > 0);
        if num_chunks < 1 {
            // The released decoder pads a two-token single-image-style latent
            // by a whole extra chunk so it still has a real decode window.
            pad_tokens = pad_tokens
                .checked_add(tokens_per_chunk)
                .ok_or_else(|| candle::Error::Msg("H3 padded token count overflow".into()))?;
            num_chunks += 1;
        }
        let padded_len = latent_frames
            .checked_add(pad_tokens)
            .ok_or_else(|| candle::Error::Msg("H3 padded latent length overflow".into()))?;
        let token_overlap = self.token_overlap();
        let mut chunks = Vec::with_capacity(num_chunks);
        for index in 0..num_chunks {
            let start = index
                .checked_mul(tokens_per_chunk)
                .ok_or_else(|| candle::Error::Msg("H3 decode chunk offset overflow".into()))?;
            chunks.push(TemporalDecodeChunk {
                latent_start: start,
                latent_frames: (tokens_per_chunk + token_overlap).min(padded_len - start),
            });
        }

        let chunk_decoded_frames = tokens_per_chunk
            .checked_mul(self.temporal_compression)
            .ok_or_else(|| candle::Error::Msg("H3 decoded chunk length overflow".into()))?;
        let split_count = usize::from(self.token_drop > 0) + 1;
        let mut total_frames = 0usize;
        let mut final_overlap_frames = 0usize;
        for chunk in &chunks {
            let clip_frames = chunk
                .latent_frames
                .checked_mul(self.temporal_compression)
                .ok_or_else(|| candle::Error::Msg("H3 decoded clip length overflow".into()))?;
            for split in 0..split_count {
                let start = split
                    .checked_mul(chunk_decoded_frames)
                    .ok_or_else(|| candle::Error::Msg("H3 decoded split offset overflow".into()))?;
                let end = start.saturating_add(chunk_decoded_frames).min(clip_frames);
                let content_start =
                    start.checked_add(self.frame_pre_padding()).ok_or_else(|| {
                        candle::Error::Msg("H3 decoded content offset overflow".into())
                    })?;
                let frames = end.saturating_sub(content_start);
                if split == 0 {
                    total_frames = total_frames.checked_add(frames).ok_or_else(|| {
                        candle::Error::Msg("H3 decoded output length overflow".into())
                    })?;
                } else {
                    final_overlap_frames = frames;
                }
            }
        }
        total_frames = total_frames
            .checked_add(final_overlap_frames)
            .ok_or_else(|| candle::Error::Msg("H3 decoded output length overflow".into()))?;
        let pad_frames = self.decode_pad_frames(padded_len, pad_tokens)?;
        total_frames = total_frames.checked_sub(pad_frames).ok_or_else(|| {
            candle::Error::Msg("H3 decode padding exceeds the planned output".into())
        })?;

        Ok(TemporalDecodePlan {
            chunks,
            repeated_tail_tokens: pad_tokens,
            output_frames: total_frames,
            tokens_per_chunk,
            token_overlap,
            frame_pre_padding: self.frame_pre_padding(),
            frame_overlap: self.frame_overlap(),
        })
    }

    fn decode_pad_frames(self, padded_len: usize, pad_tokens: usize) -> Result<usize> {
        if pad_tokens == 0 {
            return Ok(0);
        }
        let intra_tail = self.clip_length % self.temporal_compression;
        if intra_tail == 0 {
            return pad_tokens
                .checked_mul(self.temporal_compression)
                .ok_or_else(|| candle::Error::Msg("H3 decode padding overflow".into()));
        }
        let before_pad = padded_len - pad_tokens;
        (0..pad_tokens).try_fold(0usize, |total, offset| {
            let frames = if (before_pad + offset).is_multiple_of(self.tokens_per_chunk()) {
                intra_tail
            } else {
                self.temporal_compression
            };
            total
                .checked_add(frames)
                .ok_or_else(|| candle::Error::Msg("H3 decode padding overflow".into()))
        })
    }
}

/// Exact spatial tile layout used by the released 256px/64px overlap path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpatialTilePlan {
    pub starts: Vec<usize>,
    pub lengths: Vec<usize>,
    pub overlaps: Vec<usize>,
}

impl SpatialTilePlan {
    pub fn new(
        input_len: usize,
        tile_size: usize,
        minimum_overlap: usize,
        spatial_compression: usize,
    ) -> Result<Self> {
        if input_len == 0 || tile_size == 0 || spatial_compression == 0 {
            bail!("H3 visual VAE tile dimensions and compression must be positive")
        }
        if !input_len.is_multiple_of(spatial_compression)
            || !tile_size.is_multiple_of(spatial_compression)
        {
            bail!(
                "H3 visual VAE tile input {} and tile size {} must be divisible by spatial compression {}",
                input_len,
                tile_size,
                spatial_compression
            )
        }
        if minimum_overlap >= tile_size || !minimum_overlap.is_multiple_of(spatial_compression) {
            bail!(
                "H3 visual VAE overlap {} must be smaller than tile size {} and latent-aligned to {}",
                minimum_overlap,
                tile_size,
                spatial_compression
            )
        }
        if tile_size >= input_len {
            return Ok(Self {
                starts: vec![0],
                lengths: vec![input_len],
                overlaps: vec![],
            });
        }

        let covered = |count: usize| -> Result<usize> {
            tile_size
                .checked_mul(count)
                .and_then(|total| {
                    minimum_overlap
                        .checked_mul(count - 1)
                        .and_then(|overlap| total.checked_sub(overlap))
                })
                .ok_or_else(|| candle::Error::Msg("H3 tile coverage overflow".into()))
        };
        let mut num_tiles = input_len.div_ceil(tile_size);
        while covered(num_tiles)? < input_len {
            num_tiles += 1;
        }
        let mut overlaps = vec![minimum_overlap; num_tiles - 1];
        let remaining = covered(num_tiles)? - input_len;
        if !remaining.is_multiple_of(spatial_compression) {
            bail!("H3 visual VAE tile slack is not latent aligned")
        }
        for index in 0..remaining / spatial_compression {
            overlaps[index % (num_tiles - 1)] += spatial_compression;
        }
        let mut starts: Vec<usize> = Vec::with_capacity(num_tiles);
        starts.push(0);
        for overlap in &overlaps {
            let previous = starts
                .last()
                .copied()
                .ok_or_else(|| candle::Error::Msg("H3 tile plan lost its initial offset".into()))?;
            starts.push(
                previous
                    .checked_add(tile_size - overlap)
                    .ok_or_else(|| candle::Error::Msg("H3 tile offset overflow".into()))?,
            );
        }
        Ok(Self {
            starts,
            lengths: vec![tile_size; num_tiles],
            overlaps,
        })
    }
}

/// Geometry-only plan. The caller must execute the resize with LANCZOS.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EndpointResizePlan {
    /// The first/geometry-anchor image is stretched directly onto the canvas.
    Stretch { width: usize, height: usize },
    /// The follower preserves aspect ratio, covers the canvas, then uses the
    /// released centered integer crop arithmetic.
    CoverCrop {
        resized_width: usize,
        resized_height: usize,
        left: usize,
        top: usize,
        width: usize,
        height: usize,
    },
}

impl EndpointResizePlan {
    pub fn geometry_anchor(target_width: usize, target_height: usize) -> Result<Self> {
        positive_dims(target_width, target_height)?;
        Ok(Self::Stretch {
            width: target_width,
            height: target_height,
        })
    }

    pub fn follower(
        source_width: usize,
        source_height: usize,
        target_width: usize,
        target_height: usize,
    ) -> Result<Self> {
        positive_dims(source_width, source_height)?;
        positive_dims(target_width, target_height)?;
        let scale = (target_width as f64 / source_width as f64)
            .max(target_height as f64 / source_height as f64);
        // Python's round() uses ties-to-even; this is intentionally not
        // Rust's ties-away-from-zero f64::round().
        let resized_width =
            target_width.max((source_width as f64 * scale).round_ties_even() as usize);
        let resized_height =
            target_height.max((source_height as f64 * scale).round_ties_even() as usize);
        Ok(Self::CoverCrop {
            resized_width,
            resized_height,
            left: (resized_width - target_width) / 2,
            top: (resized_height - target_height) / 2,
            width: target_width,
            height: target_height,
        })
    }
}

/// Geometry-only reference-image plan. Official Ref2VA always targets a 2048
/// short edge (including upscaling); Comfy's optional deployment shortcut is
/// represented separately so it cannot silently replace the model contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReferenceImageResizePlan {
    pub width: usize,
    pub height: usize,
    pub upscaled: bool,
}

impl ReferenceImageResizePlan {
    pub fn official_2048(
        source_width: usize,
        source_height: usize,
        multiple: usize,
    ) -> Result<Self> {
        reference_short_edge_plan(source_width, source_height, 2048, multiple, true)
    }

    pub fn comfy_max_2048_down_only(
        source_width: usize,
        source_height: usize,
        multiple: usize,
    ) -> Result<Self> {
        reference_short_edge_plan(source_width, source_height, 2048, multiple, false)
    }
}

fn reference_short_edge_plan(
    width: usize,
    height: usize,
    short_edge: usize,
    multiple: usize,
    upscale: bool,
) -> Result<ReferenceImageResizePlan> {
    positive_dims(width, height)?;
    if multiple == 0 || short_edge == 0 {
        bail!("H3 reference-image short edge and alignment must be positive")
    }
    if width > height.saturating_mul(4) || height > width.saturating_mul(4) {
        bail!("H3 reference image must be within 1:4 and 4:1, got {width}x{height}")
    }
    let mut scale = short_edge as f64 / width.min(height) as f64;
    if !upscale {
        scale = scale.min(1.0);
    }
    let round_aligned = |value: usize| {
        multiple.max((value as f64 * scale / multiple as f64).round_ties_even() as usize * multiple)
    };
    let target_width = round_aligned(width);
    let target_height = round_aligned(height);
    Ok(ReferenceImageResizePlan {
        width: target_width,
        height: target_height,
        upscaled: target_width > width || target_height > height,
    })
}

fn positive_dims(width: usize, height: usize) -> Result<()> {
    if width == 0 || height == 0 {
        bail!("H3 visual dimensions must be positive, got {width}x{height}")
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn official_frame_grid_is_exact_across_chunk_seams() {
        let geometry = VisualTemporalGeometry::default();
        for (pixels, latents) in [
            (1, 1),
            (5, 2),
            (17, 2),
            (22, 7),
            (39, 12),
            (124, 37),
            (362, 107),
        ] {
            assert_eq!(
                geometry.encoded_frames(pixels).unwrap(),
                latents,
                "{pixels}"
            );
        }
        assert_eq!(geometry.decode_plan(2).unwrap().output_frames, 5);
        assert_eq!(geometry.decode_plan(7).unwrap().output_frames, 22);
        assert_eq!(geometry.decode_plan(12).unwrap().output_frames, 39);
        assert_eq!(geometry.decode_plan(37).unwrap().output_frames, 124);
        assert_eq!(geometry.decode_plan(107).unwrap().output_frames, 362);
    }

    #[test]
    fn chunks_repeat_only_the_final_input_and_drop_once_globally() {
        let plan = VisualTemporalGeometry::default().encode_plan(22).unwrap();
        assert_eq!(
            plan.chunks,
            vec![
                TemporalEncodeChunk {
                    input_start: 0,
                    valid_frames: 17,
                    repeated_tail_frames: 0,
                },
                TemporalEncodeChunk {
                    input_start: 17,
                    valid_frames: 5,
                    repeated_tail_frames: 12,
                },
            ]
        );
        assert_eq!(plan.latent_frames_before_drop, 10);
        assert_eq!(plan.trailing_latents_to_drop, 3);
    }

    #[test]
    fn released_tile_slack_is_distributed_in_latent_units() {
        let plan = SpatialTilePlan::new(544, 256, 64, 16).unwrap();
        assert_eq!(plan.starts, vec![0, 144, 288]);
        assert_eq!(plan.overlaps, vec![112, 112]);
        assert_eq!(plan.starts[2] + plan.lengths[2], 544);
        assert!(SpatialTilePlan::new(545, 256, 64, 16).is_err());
    }

    #[test]
    fn endpoint_plans_keep_the_asymmetric_reference_arithmetic() {
        assert_eq!(
            EndpointResizePlan::geometry_anchor(1344, 768).unwrap(),
            EndpointResizePlan::Stretch {
                width: 1344,
                height: 768
            }
        );
        assert_eq!(
            EndpointResizePlan::follower(1000, 1000, 1344, 768).unwrap(),
            EndpointResizePlan::CoverCrop {
                resized_width: 1344,
                resized_height: 1344,
                left: 0,
                top: 288,
                width: 1344,
                height: 768,
            }
        );
    }

    #[test]
    fn official_and_comfy_reference_image_policies_are_not_conflated() {
        let official = ReferenceImageResizePlan::official_2048(640, 480, 16).unwrap();
        let comfy = ReferenceImageResizePlan::comfy_max_2048_down_only(640, 480, 16).unwrap();
        assert_eq!((official.width, official.height), (2736, 2048));
        assert!(official.upscaled);
        assert_eq!((comfy.width, comfy.height), (640, 480));
        assert!(!comfy.upscaled);
    }
}
