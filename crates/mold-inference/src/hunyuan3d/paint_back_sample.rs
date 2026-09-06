//! Project visible material image samples into UV texels.

use anyhow::{ensure, Result};

/// Square view buffers after angle/edge reliability masking. Depth and
/// projected positions use camera-space Z; XY is normalized image space.
pub struct BackSampleView<'a> {
    pub size: u32,
    pub colors: &'a [[f32; 3]],
    pub depth: &'a [f32],
    pub visible: &'a [bool],
    pub cosine: &'a [f32],
    pub boundary: &'a [bool],
}

pub struct ProjectedTexture {
    pub size: u32,
    pub colors: Vec<[f32; 3]>,
    pub cosine: Vec<f32>,
    pub boundary: Vec<bool>,
}

/// Tencent MeshRender.py:1248-1321. Sampling deliberately uses resolution,
/// not resolution-1, and tests visibility/depth at the lower sample before
/// interpolating color. Normals/visibility are not bilinearly interpolated.
pub fn back_sample(
    view: &BackSampleView<'_>,
    projected: &[[f32; 3]],
    texels: &[usize],
    texture_size: u32,
    checkpoint: &mut dyn FnMut() -> Result<()>,
) -> Result<ProjectedTexture> {
    ensure!(
        (1..=2048).contains(&view.size) && (1..=4096).contains(&texture_size),
        "invalid paint projection extent"
    );
    let view_pixels = view.size as usize * view.size as usize;
    let texture_pixels = texture_size as usize * texture_size as usize;
    ensure!(
        view.colors.len() == view_pixels
            && view.depth.len() == view_pixels
            && view.visible.len() == view_pixels
            && view.cosine.len() == view_pixels
            && view.boundary.len() == view_pixels,
        "paint view buffer dimensions differ"
    );
    ensure!(
        projected.len() == texels.len() && projected.len() <= texture_pixels,
        "paint UV projection dimensions differ"
    );
    checkpoint()?;
    for index in 0..view_pixels {
        if index.is_multiple_of(4096) {
            checkpoint()?;
        }
        ensure!(
            view.colors[index]
                .iter()
                .all(|v| v.is_finite() && (0. ..=1.).contains(v)),
            "invalid paint view color"
        );
        ensure!(
            view.depth[index].is_finite()
                && view.cosine[index].is_finite()
                && view.cosine[index] >= 0.,
            "nonfinite or invalid paint projection"
        );
    }
    checkpoint()?;
    let mut occupied = vec![false; texture_pixels];
    for (index, (&texel, position)) in texels.iter().zip(projected).enumerate() {
        if index.is_multiple_of(4096) {
            checkpoint()?;
        }
        ensure!(
            position.iter().all(|v| v.is_finite()),
            "nonfinite projected position"
        );
        ensure!(
            texel < texture_pixels && !occupied[texel],
            "invalid or duplicate projected UV texel"
        );
        occupied[texel] = true;
    }
    drop(occupied);
    checkpoint()?;
    let colors = vec![[0.; 3]; texture_pixels];
    checkpoint()?;
    let cosine = vec![0.; texture_pixels];
    checkpoint()?;
    let boundary = vec![false; texture_pixels];
    checkpoint()?;
    let mut output = ProjectedTexture {
        size: texture_size,
        colors,
        cosine,
        boundary,
    };
    let size = view.size as usize;
    for (index, (&position, &texel)) in projected.iter().zip(texels).enumerate() {
        if index.is_multiple_of(4096) {
            checkpoint()?;
        }
        if position[0].abs() > 1. || position[1].abs() > 1. {
            continue;
        }
        let px = (position[0] * 0.5 + 0.5) * view.size as f32;
        let py = (position[1] * 0.5 + 0.5) * view.size as f32;
        let x = (px as usize).min(size - 1);
        let y = (py as usize).min(size - 1);
        let pixel = y * size + x;
        if (position[2] - view.depth[pixel]).abs() >= 3e-3
            || !view.visible[pixel]
            || view.cosine[pixel] <= 0.
        {
            continue;
        }
        let right = (x + 1).min(size - 1);
        let bottom = (y + 1).min(size - 1);
        let wx = px - x as f32;
        let wy = py - y as f32;
        output.colors[texel] = std::array::from_fn(|channel| {
            let top = view.colors[pixel][channel] * (1. - wx)
                + view.colors[y * size + right][channel] * wx;
            let bottom = view.colors[bottom * size + x][channel] * (1. - wx)
                + view.colors[bottom * size + right][channel] * wx;
            top * (1. - wy) + bottom * wy
        });
        output.cosine[texel] = view.cosine[pixel];
        output.boundary[texel] = view.boundary[pixel];
    }
    checkpoint()?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    #[test]
    fn paint_back_sample_matches_tencent_visibility_and_edges() -> anyhow::Result<()> {
        let fixture = candle_core::safetensors::load_buffer(
            include_bytes!("../../../../tests/fixtures/hunyuan3d/back-sample.safetensors"),
            &Device::Cpu,
        )?;
        let read = |name: &str| -> candle_core::Result<Vec<f32>> {
            fixture[name].flatten_all()?.to_vec1::<f32>()
        };
        let triples = |values: Vec<f32>| {
            values
                .chunks_exact(3)
                .map(|v| [v[0], v[1], v[2]])
                .collect::<Vec<_>>()
        };
        let colors = triples(read("image")?);
        let depth = read("depth")?;
        let cosine = read("cosine")?;
        let visible: Vec<bool> = read("visible")?.iter().map(|v| *v > 0.).collect();
        let edges: Vec<bool> = read("edges")?.iter().map(|v| *v > 0.).collect();
        let view = BackSampleView {
            size: 5,
            colors: &colors,
            depth: &depth,
            visible: &visible,
            cosine: &cosine,
            boundary: &edges,
        };
        let projected = triples(read("projected")?);
        let texels: Vec<usize> = (0..projected.len()).collect();
        let actual = back_sample(&view, &projected, &texels, 4, &mut || Ok(()))?;
        let max = actual
            .colors
            .iter()
            .flatten()
            .zip(read("texture")?)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(max <= 1e-7, "back-sampled color max {max}");
        assert_eq!(actual.cosine, read("output_cosine")?);
        assert_eq!(
            actual.boundary,
            read("output_edges")?
                .iter()
                .map(|v| *v > 0.)
                .collect::<Vec<_>>()
        );
        Ok(())
    }

    #[test]
    fn paint_back_sample_rejects_bad_texels_and_cancels() -> anyhow::Result<()> {
        let view = BackSampleView {
            size: 1,
            colors: &[[0.5; 3]],
            depth: &[-1.],
            visible: &[true],
            cosine: &[1.],
            boundary: &[false],
        };
        let position = [[0., 0., -1.]];
        assert!(back_sample(&view, &position, &[1], 1, &mut || Ok(())).is_err());
        assert!(back_sample(&view, &[position[0]; 2], &[0, 0], 2, &mut || Ok(())).is_err());
        let mut checks = 0;
        let error = back_sample(&view, &position, &[0], 1, &mut || {
            checks += 1;
            anyhow::bail!("cancel projection")
        })
        .err()
        .unwrap();
        assert_eq!(checks, 1);
        assert_eq!(error.to_string(), "cancel projection");
        Ok(())
    }
}
