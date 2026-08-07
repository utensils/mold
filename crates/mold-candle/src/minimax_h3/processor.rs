use thiserror::Error;

use super::presentation::{H3_IMAGE_PAD_TOKEN_ID, H3_VIDEO_PAD_TOKEN_ID};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum QwenMmTokenType {
    Text = 0,
    Image = 1,
    Video = 2,
}

pub fn create_mm_token_type_ids(token_ids: &[u32]) -> Vec<QwenMmTokenType> {
    token_ids
        .iter()
        .map(|id| match *id {
            H3_IMAGE_PAD_TOKEN_ID => QwenMmTokenType::Image,
            H3_VIDEO_PAD_TOKEN_ID => QwenMmTokenType::Video,
            _ => QwenMmTokenType::Text,
        })
        .collect()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GridThw {
    pub temporal: usize,
    pub height: usize,
    pub width: usize,
}

impl GridThw {
    pub fn merged_tokens(self, spatial_merge_size: usize) -> Result<usize, ProcessorError> {
        if self.temporal == 0 || self.height == 0 || self.width == 0 {
            return Err(ProcessorError::UnsupportedGrid(self));
        }
        if spatial_merge_size == 0
            || !self.height.is_multiple_of(spatial_merge_size)
            || !self.width.is_multiple_of(spatial_merge_size)
        {
            return Err(ProcessorError::UnsupportedGrid(self));
        }
        Ok(self.temporal * (self.height / spatial_merge_size) * (self.width / spatial_merge_size))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SampledVideo {
    pub frame_indices: Vec<usize>,
    pub block_timestamps_seconds: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PackedVisionPatches {
    pub values: Vec<f32>,
    pub patch_count: usize,
    pub patch_width: usize,
    pub grid: GridThw,
}

#[derive(Debug, Error, PartialEq)]
pub enum ProcessorError {
    #[error("source video fps must be finite and positive, got {0}")]
    InvalidFps(f64),
    #[error("a reference at {source_fps} fps needs at least {minimum_frames} frames for Qwen's 2 fps temporal-patch-2 input, got {actual_frames}")]
    VideoTooShort {
        source_fps: f64,
        minimum_frames: usize,
        actual_frames: usize,
    },
    #[error("unsupported Qwen3-VL processor grid {0:?}")]
    UnsupportedGrid(GridThw),
    #[error("Qwen3-VL multimodal run mismatch: {0}")]
    MultimodalRun(String),
    #[error("unsupported Qwen3-VL pixel tensor: {0}")]
    UnsupportedPixels(String),
}

/// Normalize RGB bytes to `[-1, 1]` and reproduce Qwen's exact
/// temporal/spatial patch permutation after the caller has resized frames to
/// processor-authorized dimensions. A single image is repeated across the
/// temporal pair; an odd video repeats its final frame.
pub fn pack_qwen_vision_u8(
    rgb_frames: &[u8],
    frame_count: usize,
    height: usize,
    width: usize,
) -> Result<PackedVisionPatches, ProcessorError> {
    const CHANNELS: usize = 3;
    const TEMPORAL_PATCH: usize = 2;
    const PATCH: usize = 16;
    const MERGE: usize = 2;
    let expected = frame_count
        .checked_mul(height)
        .and_then(|value| value.checked_mul(width))
        .and_then(|value| value.checked_mul(CHANNELS))
        .ok_or_else(|| ProcessorError::UnsupportedPixels("shape byte count overflow".into()))?;
    if frame_count == 0 || rgb_frames.len() != expected {
        return Err(ProcessorError::UnsupportedPixels(format!(
            "{frame_count}x{height}x{width}x3 requires {expected} bytes, got {}",
            rgb_frames.len()
        )));
    }
    let factor = PATCH * MERGE;
    if height == 0 || width == 0 || !height.is_multiple_of(factor) || !width.is_multiple_of(factor)
    {
        return Err(ProcessorError::UnsupportedPixels(format!(
            "resized dimensions {width}x{height} must be positive multiples of {factor}"
        )));
    }
    let padded_frames = frame_count.next_multiple_of(TEMPORAL_PATCH);
    let grid = GridThw {
        temporal: padded_frames / TEMPORAL_PATCH,
        height: height / PATCH,
        width: width / PATCH,
    };
    let patch_count = grid.temporal * grid.height * grid.width;
    let patch_width = CHANNELS * TEMPORAL_PATCH * PATCH * PATCH;
    let mut values = Vec::with_capacity(patch_count * patch_width);
    for temporal_block in 0..grid.temporal {
        for outer_row in 0..grid.height / MERGE {
            for outer_column in 0..grid.width / MERGE {
                for inner_row in 0..MERGE {
                    for inner_column in 0..MERGE {
                        let patch_row = (outer_row * MERGE + inner_row) * PATCH;
                        let patch_column = (outer_column * MERGE + inner_column) * PATCH;
                        for channel in 0..CHANNELS {
                            for temporal_inner in 0..TEMPORAL_PATCH {
                                let frame = (temporal_block * TEMPORAL_PATCH + temporal_inner)
                                    .min(frame_count - 1);
                                for row in patch_row..patch_row + PATCH {
                                    for column in patch_column..patch_column + PATCH {
                                        let index = (((frame * height + row) * width + column)
                                            * CHANNELS)
                                            + channel;
                                        values.push(f32::from(rgb_frames[index]) / 127.5 - 1.0);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    debug_assert_eq!(values.len(), patch_count * patch_width);
    Ok(PackedVisionPatches {
        values,
        patch_count,
        patch_width,
        grid,
    })
}

/// Reproduce the official 2 fps cursor sampling, including Python's ties-to-
/// even `round`, deduplication, temporal-patch-2 padding, and mean labels.
pub fn sample_video_frames(
    frame_count: usize,
    source_fps: f64,
) -> Result<SampledVideo, ProcessorError> {
    const SAMPLE_FPS: f64 = 2.0;
    const TEMPORAL_PATCH: usize = 2;
    if !source_fps.is_finite() || source_fps <= 0.0 {
        return Err(ProcessorError::InvalidFps(source_fps));
    }
    let stride = source_fps / SAMPLE_FPS;
    let mut cursor = 0.0_f64;
    let mut frame_indices = Vec::new();
    loop {
        let index = cursor.round_ties_even() as usize;
        if index >= frame_count {
            break;
        }
        if frame_indices
            .last()
            .is_none_or(|previous| index > *previous)
        {
            frame_indices.push(index);
        }
        cursor += stride;
    }
    if frame_indices.len() < TEMPORAL_PATCH {
        let minimum_frames = (((TEMPORAL_PATCH - 1) as f64 * stride).round_ties_even() as usize
            + 1)
        .max(TEMPORAL_PATCH);
        return Err(ProcessorError::VideoTooShort {
            source_fps,
            minimum_frames,
            actual_frames: frame_count,
        });
    }
    let mut timestamps: Vec<_> = (0..frame_indices.len())
        .map(|index| index as f64 / SAMPLE_FPS)
        .collect();
    while !timestamps.len().is_multiple_of(TEMPORAL_PATCH) {
        timestamps.push(*timestamps.last().expect("at least two timestamps"));
    }
    let block_timestamps_seconds = timestamps
        .chunks_exact(TEMPORAL_PATCH)
        .map(|chunk| (chunk[0] + chunk[TEMPORAL_PATCH - 1]) / 2.0)
        .collect();
    Ok(SampledVideo {
        frame_indices,
        block_timestamps_seconds,
    })
}

/// Qwen3-VL's three-axis position ids for one unpadded presentation.
/// Video grids are split into one temporal block per contiguous video-pad run,
/// matching the timestamped H3 presentation.
pub fn qwen_mrope_positions(
    token_types: &[QwenMmTokenType],
    image_grids: &[GridThw],
    video_grids: &[GridThw],
    spatial_merge_size: usize,
) -> Result<[Vec<u32>; 3], ProcessorError> {
    let mut expanded_video_grids = Vec::new();
    for grid in video_grids {
        grid.merged_tokens(spatial_merge_size)?;
        expanded_video_grids.extend(std::iter::repeat_n(
            GridThw {
                temporal: 1,
                height: grid.height,
                width: grid.width,
            },
            grid.temporal,
        ));
    }
    let mut image_iter = image_grids.iter();
    let mut video_iter = expanded_video_grids.iter();
    let mut positions: [Vec<u32>; 3] =
        std::array::from_fn(|_| Vec::with_capacity(token_types.len()));
    let mut current = 0_u32;
    let mut start = 0;
    while start < token_types.len() {
        let kind = token_types[start];
        let mut end = start + 1;
        while end < token_types.len() && token_types[end] == kind {
            end += 1;
        }
        let run_len = end - start;
        match kind {
            QwenMmTokenType::Text => {
                for offset in 0..run_len as u32 {
                    for axis in &mut positions {
                        axis.push(current + offset);
                    }
                }
                current += run_len as u32;
            }
            QwenMmTokenType::Image | QwenMmTokenType::Video => {
                let grid = match kind {
                    QwenMmTokenType::Image => image_iter.next(),
                    QwenMmTokenType::Video => video_iter.next(),
                    QwenMmTokenType::Text => unreachable!(),
                }
                .ok_or_else(|| ProcessorError::MultimodalRun(format!("missing {kind:?} grid")))?;
                let expected = grid.merged_tokens(spatial_merge_size)?;
                if expected != run_len {
                    return Err(ProcessorError::MultimodalRun(format!(
                        "{kind:?} pad run has {run_len} tokens but grid {grid:?} yields {expected}"
                    )));
                }
                let merged_h = grid.height / spatial_merge_size;
                let merged_w = grid.width / spatial_merge_size;
                for temporal in 0..grid.temporal {
                    for height in 0..merged_h {
                        for width in 0..merged_w {
                            positions[0].push(current + temporal as u32);
                            positions[1].push(current + height as u32);
                            positions[2].push(current + width as u32);
                        }
                    }
                }
                current += grid.temporal.max(merged_h).max(merged_w) as u32;
            }
        }
        start = end;
    }
    if image_iter.next().is_some() || video_iter.next().is_some() {
        return Err(ProcessorError::MultimodalRun(
            "processor returned more grids than the presentation contains".into(),
        ));
    }
    Ok(positions)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn video_sampling_matches_official_two_fps_and_half_even_labels() {
        let sampled = sample_video_frames(49, 24.0).unwrap();
        assert_eq!(sampled.frame_indices, vec![0, 12, 24, 36, 48]);
        assert_eq!(sampled.block_timestamps_seconds, vec![0.25, 1.25, 2.0]);
        assert_eq!(format!("{:.1}", sampled.block_timestamps_seconds[0]), "0.2");
    }

    #[test]
    fn video_sampling_reports_exact_minimum() {
        assert_eq!(
            sample_video_frames(12, 24.0).unwrap_err(),
            ProcessorError::VideoTooShort {
                source_fps: 24.0,
                minimum_frames: 13,
                actual_frames: 12,
            }
        );
        assert_eq!(
            sample_video_frames(1, 1.0).unwrap_err(),
            ProcessorError::VideoTooShort {
                source_fps: 1.0,
                minimum_frames: 2,
                actual_frames: 1,
            }
        );
    }

    #[test]
    fn mrope_uses_text_runs_and_visual_axes() {
        let types = [
            QwenMmTokenType::Text,
            QwenMmTokenType::Text,
            QwenMmTokenType::Image,
            QwenMmTokenType::Image,
            QwenMmTokenType::Image,
            QwenMmTokenType::Image,
            QwenMmTokenType::Text,
        ];
        let positions = qwen_mrope_positions(
            &types,
            &[GridThw {
                temporal: 1,
                height: 4,
                width: 4,
            }],
            &[],
            2,
        )
        .unwrap();
        assert_eq!(positions[0], vec![0, 1, 2, 2, 2, 2, 4]);
        assert_eq!(positions[1], vec![0, 1, 2, 2, 3, 3, 4]);
        assert_eq!(positions[2], vec![0, 1, 2, 3, 2, 3, 4]);
    }

    #[test]
    fn mismatched_vision_run_is_rejected() {
        let error = qwen_mrope_positions(
            &[QwenMmTokenType::Image],
            &[GridThw {
                temporal: 1,
                height: 4,
                width: 4,
            }],
            &[],
            2,
        )
        .unwrap_err();
        assert!(error.to_string().contains("has 1 tokens"));
    }

    #[test]
    fn vision_packing_duplicates_single_frame_and_uses_merge_block_order() {
        let mut pixels = vec![0_u8; 32 * 32 * 3];
        for row in 0..32 {
            for column in 0..32 {
                pixels[(row * 32 + column) * 3] = (row * 32 + column) as u8;
            }
        }
        let packed = pack_qwen_vision_u8(&pixels, 1, 32, 32).unwrap();
        assert_eq!(
            packed.grid,
            GridThw {
                temporal: 1,
                height: 2,
                width: 2
            }
        );
        assert_eq!(packed.patch_count, 4);
        assert_eq!(packed.patch_width, 1_536);
        // Channel 0, temporal frame 0 begins at (0, 0); frame 1 is the
        // repeated image and starts after one 16x16 plane.
        assert_eq!(packed.values[0], -1.0);
        assert_eq!(packed.values[16 * 16], -1.0);
        // The second emitted patch is the adjacent inner merge column and
        // starts at source column 16.
        assert_eq!(
            packed.values[packed.patch_width],
            f32::from(pixels[16 * 3]) / 127.5 - 1.0
        );
    }

    #[test]
    fn vision_packing_rejects_unresized_shapes() {
        let error = pack_qwen_vision_u8(&vec![0; 31 * 32 * 3], 1, 31, 32).unwrap_err();
        assert!(error.to_string().contains("multiples of 32"));
    }
}
