//! Tiled upscaling with overlap blending for memory-efficient large image processing.
//!
//! Splits the input image into overlapping tiles, processes each through the
//! upscaler model, and blends overlapping regions using linear gradient weights
//! (feathered blending) to eliminate visible seams.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use std::time::Instant;

use crate::progress::ProgressReporter;

/// Configuration for tiled upscaling.
#[derive(Debug, Clone)]
pub struct TilingConfig {
    /// Tile size in pixels (input space). Default: 512.
    pub tile_size: u32,
    /// Overlap between adjacent tiles in pixels (input space). Default: 32.
    pub overlap: u32,
    /// Minimum tile size before giving up on OOM reduction. Default: 128.
    /// Reserved for future OOM-retry logic.
    #[allow(dead_code)]
    pub min_tile_size: u32,
}

impl Default for TilingConfig {
    fn default() -> Self {
        Self {
            tile_size: 512,
            overlap: 32,
            min_tile_size: 128,
        }
    }
}

/// A tile region in the input image.
struct TileRegion {
    x: usize,
    y: usize,
    w: usize,
    h: usize,
}

/// Calculate tile grid for a given image size and tiling config.
fn calculate_tiles(img_w: usize, img_h: usize, config: &TilingConfig) -> Vec<TileRegion> {
    let tile = config.tile_size as usize;
    let overlap = config.overlap as usize;
    let step = tile.saturating_sub(overlap).max(1);

    let mut tiles = Vec::new();
    let mut y = 0;
    while y < img_h {
        let mut x = 0;
        let h = tile.min(img_h - y);
        while x < img_w {
            let w = tile.min(img_w - x);
            tiles.push(TileRegion { x, y, w, h });
            if x + w >= img_w {
                break;
            }
            x += step;
        }
        if y + h >= img_h {
            break;
        }
        y += step;
    }
    tiles
}

/// Build a feathered blending weight tensor for a tile at a given position.
/// Edges that border the image boundary get weight 1.0; interior edges
/// get a linear ramp from 0 to 1 over the overlap region.
#[allow(clippy::too_many_arguments)]
fn build_blend_weights(
    tile_x: usize,
    tile_y: usize,
    tile_w: usize,
    tile_h: usize,
    img_w: usize,
    img_h: usize,
    overlap: usize,
    scale: u32,
    device: &Device,
) -> Result<Tensor> {
    let out_w = tile_w * scale as usize;
    let out_h = tile_h * scale as usize;
    let out_overlap = overlap * scale as usize;

    let mut weights = vec![1.0f32; out_h * out_w];

    // Left edge ramp (unless at image left boundary)
    if tile_x > 0 && out_overlap > 0 {
        let ramp_len = out_overlap.min(out_w);
        for row in 0..out_h {
            for col in 0..ramp_len {
                let factor = (col as f32 + 1.0) / ramp_len as f32;
                weights[row * out_w + col] *= factor;
            }
        }
    }

    // Top edge ramp (unless at image top boundary)
    if tile_y > 0 && out_overlap > 0 {
        let ramp_len = out_overlap.min(out_h);
        for row in 0..ramp_len {
            let factor = (row as f32 + 1.0) / ramp_len as f32;
            for col in 0..out_w {
                weights[row * out_w + col] *= factor;
            }
        }
    }

    // Right edge ramp (unless at image right boundary)
    if tile_x + tile_w < img_w && out_overlap > 0 {
        let ramp_len = out_overlap.min(out_w);
        for row in 0..out_h {
            for col in 0..ramp_len {
                let factor = (col as f32 + 1.0) / ramp_len as f32;
                weights[row * out_w + (out_w - 1 - col)] *= factor;
            }
        }
    }

    // Bottom edge ramp (unless at image bottom boundary)
    if tile_y + tile_h < img_h && out_overlap > 0 {
        let ramp_len = out_overlap.min(out_h);
        for row in 0..ramp_len {
            let factor = (row as f32 + 1.0) / ramp_len as f32;
            for col in 0..out_w {
                weights[(out_h - 1 - row) * out_w + col] *= factor;
            }
        }
    }

    let weights = Tensor::from_vec(weights, (1, 1, out_h, out_w), device)?;
    Ok(weights)
}

/// Upscale an image tensor using tiled inference with overlap blending.
///
/// # Arguments
/// * `input` - Input tensor of shape [1, 3, H, W], values in [0, 1]
/// * `forward_fn` - Function that runs the upscaler model on a single tile
/// * `scale` - Upscaling factor (2 or 4)
/// * `config` - Tiling configuration
/// * `device` - Compute device
/// * `progress` - Progress reporter for tile-by-tile updates
pub fn upscale_with_tiling<F>(
    input: &Tensor,
    forward_fn: &F,
    scale: u32,
    config: &TilingConfig,
    device: &Device,
    progress: &ProgressReporter,
) -> Result<Tensor>
where
    F: Fn(&Tensor) -> Result<Tensor>,
{
    let (_, _, img_h, img_w) = input.dims4()?;

    // If the image fits in a single tile, skip tiling overhead
    if img_w <= config.tile_size as usize && img_h <= config.tile_size as usize {
        progress.emit(crate::progress::ProgressEvent::DenoiseStep {
            step: 1,
            total: 1,
            elapsed: std::time::Duration::ZERO,
        });
        return forward_fn(input);
    }

    let tiles = calculate_tiles(img_w, img_h, config);
    let num_tiles = tiles.len();
    progress.info(&format!(
        "Tiling: {}x{} image -> {} tiles ({}px with {}px overlap)",
        img_w, img_h, num_tiles, config.tile_size, config.overlap
    ));

    let out_h = img_h * scale as usize;
    let out_w = img_w * scale as usize;

    // Accumulate on CPU using raw f32 arrays for simple slice-add operations.
    // candle tensors don't have ergonomic mutable slice-assign, so we
    // accumulate in raw buffers and convert to tensor at the end.
    let mut output_acc = vec![0f32; 3 * out_h * out_w];
    let mut weight_acc = vec![0f32; out_h * out_w];

    let start = Instant::now();

    for (idx, tile) in tiles.iter().enumerate() {
        // Extract tile from input
        let tile_input = input.narrow(2, tile.y, tile.h)?.narrow(3, tile.x, tile.w)?;

        // Run upscaler on this tile
        let tile_output = forward_fn(&tile_input)?;
        let tile_output = tile_output.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;

        // Get tile output as flat f32 vec [C, H, W]
        let tile_data: Vec<f32> = tile_output.flatten_all()?.to_vec1()?;
        let out_tw = tile.w * scale as usize;
        let out_th = tile.h * scale as usize;
        let out_x = tile.x * scale as usize;
        let out_y = tile.y * scale as usize;

        // Build blending weights as raw vec
        let weights = build_blend_weights(
            tile.x,
            tile.y,
            tile.w,
            tile.h,
            img_w,
            img_h,
            config.overlap as usize,
            scale,
            &Device::Cpu,
        )?;
        let weight_data: Vec<f32> = weights.flatten_all()?.to_vec1()?;

        // Accumulate weighted output and weights
        for c in 0..3 {
            for row in 0..out_th {
                for col in 0..out_tw {
                    let w = weight_data[row * out_tw + col];
                    let val = tile_data[c * out_th * out_tw + row * out_tw + col];
                    let dst_row = out_y + row;
                    let dst_col = out_x + col;
                    output_acc[c * out_h * out_w + dst_row * out_w + dst_col] += val * w;
                    if c == 0 {
                        weight_acc[dst_row * out_w + dst_col] += w;
                    }
                }
            }
        }

        progress.emit(crate::progress::ProgressEvent::DenoiseStep {
            step: idx + 1,
            total: num_tiles,
            elapsed: start.elapsed(),
        });
    }

    // Normalize by weight sum
    for c in 0..3 {
        for i in 0..out_h * out_w {
            if weight_acc[i] > 0.0 {
                output_acc[c * out_h * out_w + i] /= weight_acc[i];
            }
        }
    }

    // Convert back to tensor
    let output = Tensor::from_vec(output_acc, (1, 3, out_h, out_w), device)?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calculate_tiles_single() {
        let config = TilingConfig {
            tile_size: 512,
            overlap: 32,
            min_tile_size: 128,
        };
        let tiles = calculate_tiles(256, 256, &config);
        assert_eq!(tiles.len(), 1);
        assert_eq!(
            (tiles[0].x, tiles[0].y, tiles[0].w, tiles[0].h),
            (0, 0, 256, 256)
        );
    }

    #[test]
    fn calculate_tiles_multiple() {
        let config = TilingConfig {
            tile_size: 128,
            overlap: 32,
            min_tile_size: 64,
        };
        let tiles = calculate_tiles(256, 256, &config);
        // step = 128 - 32 = 96, so x positions: 0, 96, (96+128=224 > 256, so clip)
        assert!(
            tiles.len() > 1,
            "expected multiple tiles, got {}",
            tiles.len()
        );
    }

    #[test]
    fn blend_weights_corner_tile() {
        let device = Device::Cpu;
        // Top-left corner: no left/top ramp needed
        let weights = build_blend_weights(0, 0, 64, 64, 256, 256, 16, 1, &device).unwrap();
        let dims = weights.dims4().unwrap();
        assert_eq!(dims, (1, 1, 64, 64));
        // Top-left pixel should be 1.0 (no ramp)
        let val: f32 = weights
            .flatten_all()
            .unwrap()
            .get(0)
            .unwrap()
            .to_scalar()
            .unwrap();
        assert!((val - 1.0).abs() < 1e-6);
    }

    #[test]
    fn blend_weights_interior_tile() {
        let device = Device::Cpu;
        // Interior tile: all edges get ramps
        let weights = build_blend_weights(64, 64, 64, 64, 256, 256, 16, 1, &device).unwrap();
        let dims = weights.dims4().unwrap();
        assert_eq!(dims, (1, 1, 64, 64));
        // Top-left pixel should be less than 1.0 (has ramp from both top and left)
        let val: f32 = weights
            .flatten_all()
            .unwrap()
            .get(0)
            .unwrap()
            .to_scalar()
            .unwrap();
        assert!(
            val < 1.0,
            "interior tile corner should have ramp, got {val}"
        );
    }
}
