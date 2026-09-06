//! Composition of mesh propagation and pixel-space texture filling.

use super::{
    paint_ns_fill,
    paint_vertex_fill::{fill_vertices, VertexFillInput},
};
use anyhow::{ensure, Result};
use image::RgbImage;

/// Tencent MeshRender.py:1406-1411: propagate mesh vertices, multiply in
/// float32 and truncate bytes, then run RGB Navier–Stokes. Material channels
/// remain data throughout; no gamma transform is introduced for MR.
pub fn fill_texture(
    input: &VertexFillInput<'_>,
    checkpoint: &mut dyn FnMut() -> Result<()>,
) -> Result<RgbImage> {
    checkpoint()?;
    ensure!(
        input.width >= 2 && input.height >= 2,
        "invalid texture fill dimensions"
    );
    for chunk in input.texture.chunks(4096) {
        checkpoint()?;
        ensure!(
            chunk
                .iter()
                .flatten()
                .all(|v| v.is_finite() && (0.0..=1.0).contains(v)),
            "texture fill requires normalized finite colors"
        );
    }
    let (propagated, trust) = fill_vertices(input, checkpoint)?;
    let mut pixels = Vec::with_capacity(propagated.len());
    for chunk in propagated.chunks(4096) {
        checkpoint()?;
        pixels.extend(chunk.iter().map(|pixel| pixel.map(|v| (v * 255.0) as u8)));
    }
    drop(propagated);
    let filled = paint_ns_fill::fill_rgb(&pixels, &trust, input.width, input.height, checkpoint)?;
    drop(pixels);
    let mut raw = Vec::with_capacity(filled.len() * 3);
    for chunk in filled.chunks(4096) {
        checkpoint()?;
        raw.extend(chunk.iter().flatten().copied());
    }
    checkpoint()?;
    RgbImage::from_raw(input.width as u32, input.height as u32, raw)
        .ok_or_else(|| anyhow::anyhow!("invalid filled texture length"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn texture_fill_truncates_float_bytes_before_ns() {
        let pixels = [[0.5, 0.1, 1.0]; 4];
        let input = VertexFillInput {
            width: 2,
            height: 2,
            texture: &pixels,
            mask: &[255; 4],
            positions: &[],
            uv: &[],
            faces: &[],
            uv_faces: &[],
        };
        let mut calls = 0;
        let result = fill_texture(&input, &mut || {
            calls += 1;
            Ok(())
        })
        .unwrap();
        for cancel_at in 1..=calls {
            let mut seen = 0;
            let error = fill_texture(&input, &mut || {
                seen += 1;
                anyhow::ensure!(seen != cancel_at, "cancelled");
                Ok(())
            })
            .unwrap_err();
            assert_eq!(error.to_string(), "cancelled");
        }
        assert!(result.pixels().all(|p| p.0 == [127, 25, 255]));
    }

    #[test]
    #[ignore = "requires retained chair fill oracle and new output directory"]
    fn retained_chair_texture_fill_matches_oracle() -> anyhow::Result<()> {
        use candle_core::{Device, Tensor};
        use std::path::PathBuf;
        let oracle = PathBuf::from(std::env::var("MOLD_TEXTURE_FILL_ORACLE")?);
        let output = PathBuf::from(std::env::var("MOLD_TEXTURE_FILL_OUTPUT")?);
        std::fs::create_dir(&output)?;
        let mut reports = Vec::new();
        let triples = |tensor: &Tensor| -> anyhow::Result<Vec<[f32; 3]>> {
            Ok(tensor
                .reshape(((), 3))?
                .to_vec2::<f32>()?
                .into_iter()
                .map(|v| [v[0], v[1], v[2]])
                .collect())
        };
        let mut failed = false;
        for stream in 0..2 {
            let tensors = candle_core::safetensors::load(
                oracle.join(format!("fill-{stream}.safetensors")),
                &Device::Cpu,
            )?;
            let (height, width, _) = tensors["texture"].dims3()?;
            let texture = triples(&tensors["texture"])?;
            let positions = triples(&tensors["positions"])?;
            let uv: Vec<[f32; 2]> = tensors["uv"]
                .to_vec2::<f32>()?
                .into_iter()
                .map(|v| [v[0], v[1]])
                .collect();
            let faces: Vec<[u32; 3]> = tensors["faces"]
                .to_vec2::<u32>()?
                .into_iter()
                .map(|v| [v[0], v[1], v[2]])
                .collect();
            let mask = tensors["mask"].flatten_all()?.to_vec1::<u8>()?;
            let input = VertexFillInput {
                width,
                height,
                texture: &texture,
                mask: &mask,
                positions: &positions,
                uv: &uv,
                faces: &faces,
                uv_faces: &faces,
            };
            let start = std::time::Instant::now();
            let actual = fill_texture(&input, &mut || Ok(()))?;
            let seconds = start.elapsed().as_secs_f64();
            actual.save(output.join(format!("actual-{stream}.png")))?;
            anyhow::ensure!(
                tensors["final"].dims() == [height, width, 3],
                "oracle final shape differs"
            );
            let expected = tensors["final"].flatten_all()?.to_vec1::<u8>()?;
            anyhow::ensure!(
                expected.len() == actual.as_raw().len(),
                "oracle final byte count differs"
            );
            let max = actual
                .as_raw()
                .iter()
                .zip(&expected)
                .map(|(a, b)| a.abs_diff(*b))
                .max()
                .unwrap_or(0);
            let different = actual
                .as_raw()
                .iter()
                .zip(&expected)
                .filter(|(a, b)| a != b)
                .count();
            reports.push(serde_json::json!({"stream":stream,"size":width,"max_byte":max,"different_channels":different,"seconds":seconds}));
            failed |= max != 0;
        }
        std::fs::write(
            output.join("comparison.json"),
            serde_json::to_vec_pretty(&reports)?,
        )?;
        anyhow::ensure!(!failed, "chair texture fill differs from oracle");
        Ok(())
    }
}
