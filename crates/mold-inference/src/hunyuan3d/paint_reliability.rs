//! Depth, angle and boundary reliability for texture back-projection.

use anyhow::{ensure, Result};

pub struct ReliabilityMask {
    pub visible: Vec<bool>,
    pub cosine: Vec<f32>,
    pub boundary: Vec<bool>,
}

fn view_cosine(n: [f32; 3]) -> f32 {
    // cosine_similarity uses the same CUDA three-element reduction as
    // normalize: sum components 0+2 before adding component 1.
    let norm = (n[0] * n[0] + n[2] * n[2] + n[1] * n[1]).sqrt().max(1e-8);
    -n[2] / norm
}

impl ReliabilityMask {
    /// Tencent MeshRender.back_project:1182-1214. Input normals and depth are
    /// camera-space values; radius is the renderer's unreliable-kernel radius
    /// (default floor(render_size * 2 / 512), eight at the 2048 bake size).
    pub fn from_geometry(
        depth: &[f32],
        normal: &[[f32; 3]],
        visible: &[bool],
        size: usize,
        radius: usize,
        checkpoint: &mut dyn FnMut() -> Result<()>,
    ) -> Result<Self> {
        ensure!(
            (1..=2048).contains(&size) && radius <= 8,
            "invalid paint reliability dimensions"
        );
        let count = size * size;
        ensure!(
            depth.len() == count && normal.len() == count && visible.len() == count,
            "paint reliability geometry lengths differ"
        );
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        for index in 0..count {
            if index.is_multiple_of(4096) {
                checkpoint()?;
            }
            ensure!(
                depth[index].is_finite()
                    && normal[index]
                        .iter()
                        .all(|v| v.is_finite() && v.abs() <= 1.01),
                "invalid paint depth or unit normal"
            );
            if visible[index] {
                min = min.min(depth[index]);
                max = max.max(depth[index]);
            }
        }
        ensure!(min.is_finite(), "paint camera has no visible surface");
        let range = max - min;
        ensure!(range.is_finite(), "paint depth range overflows");
        checkpoint()?;
        let mut bytes = Vec::with_capacity(count);
        for index in 0..count {
            if index.is_multiple_of(4096) {
                checkpoint()?;
            }
            // Upstream flat-depth division yields NaN, and NumPy converts that
            // to zero U8. Keep the same bytes without introducing an epsilon.
            let value = if range == 0. || !visible[index] {
                0.
            } else {
                (depth[index] - min) / range
            };
            bytes.push((value * 255.) as u8);
        }
        let mut boundary = super::paint_edges::depth_edges(&bytes, size, size, checkpoint)?;
        checkpoint()?;
        let mut reliable = visible.to_vec();
        if radius > 0 {
            let invisible = square_dilate(visible, size, radius, true, checkpoint)?;
            boundary = square_dilate(&boundary, size, radius, false, checkpoint)?;
            for index in 0..count {
                if index.is_multiple_of(4096) {
                    checkpoint()?;
                }
                reliable[index] = !invisible[index] && !boundary[index];
            }
        }
        checkpoint()?;
        let mut cosine = Vec::with_capacity(count);
        let threshold = 75f64.to_radians().cos() as f32;
        for (index, n) in normal.iter().enumerate() {
            if index.is_multiple_of(4096) {
                checkpoint()?;
            }
            let cos = view_cosine(*n);
            cosine.push(if !reliable[index] || cos < threshold {
                0.
            } else {
                cos
            });
        }
        checkpoint()?;
        Ok(Self {
            visible: reliable,
            cosine,
            boundary,
        })
    }
}

/// Exact binary square convolution >0 via an integral image. Samples outside
/// the image contribute zero, including when dilating the inverted visibility.
fn square_dilate(
    mask: &[bool],
    size: usize,
    radius: usize,
    invert: bool,
    checkpoint: &mut dyn FnMut() -> Result<()>,
) -> Result<Vec<bool>> {
    checkpoint()?;
    let stride = size + 1;
    let mut sum = vec![0u32; stride * stride];
    for y in 0..size {
        checkpoint()?;
        let mut row = 0;
        for x in 0..size {
            row += u32::from(mask[y * size + x] != invert);
            sum[(y + 1) * stride + x + 1] = sum[y * stride + x + 1] + row;
        }
    }
    checkpoint()?;
    let mut output = Vec::with_capacity(mask.len());
    for y in 0..size {
        checkpoint()?;
        let top = y.saturating_sub(radius) * stride;
        let bottom = (y + radius + 1).min(size) * stride;
        for x in 0..size {
            let left = x.saturating_sub(radius);
            let right = (x + radius + 1).min(size);
            output.push(
                sum[bottom + right] + sum[top + left] - sum[top + right] - sum[bottom + left] > 0,
            );
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use candle_core::Device;

    #[test]
    fn cosine_retains_cuda_three_element_reduction_order() {
        let cosine = super::view_cosine([
            f32::from_bits(1_045_635_208),
            f32::from_bits(1_063_664_835),
            f32::from_bits(3_200_607_285),
        ]);
        assert_eq!(cosine.to_bits(), 1_053_123_637);
    }

    #[test]
    fn reliability_matches_tencent() -> anyhow::Result<()> {
        let fixture = candle_core::safetensors::load_buffer(
            include_bytes!("../../../../tests/fixtures/hunyuan3d/paint-reliability.safetensors"),
            &Device::Cpu,
        )?;
        for (name, radius) in [
            ("full", 2),
            ("hole", 1),
            ("step", 0),
            ("step_dilated", 2),
            ("flat", 8),
            ("angle", 0),
        ] {
            let floats = |field: &str| {
                fixture[&format!("{name}.{field}")]
                    .flatten_all()?
                    .to_vec1::<f32>()
            };
            let bools = |field: &str| -> candle_core::Result<Vec<bool>> {
                Ok(fixture[&format!("{name}.{field}")]
                    .flatten_all()?
                    .to_vec1::<u8>()?
                    .iter()
                    .map(|v| *v != 0)
                    .collect())
            };
            let normals: Vec<[f32; 3]> = floats("normal")?
                .chunks_exact(3)
                .map(|v| [v[0], v[1], v[2]])
                .collect();
            let mask = super::ReliabilityMask::from_geometry(
                &floats("depth")?,
                &normals,
                &bools("visible")?,
                16,
                radius,
                &mut || Ok(()),
            )?;
            assert_eq!(mask.visible, bools("reliable")?, "{name}: visible");
            assert_eq!(mask.boundary, bools("boundary")?, "{name}: boundary");
            let max = mask
                .cosine
                .iter()
                .zip(floats("cosine")?)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(max <= 1e-7, "{name}: cosine max {max}");
        }
        Ok(())
    }

    #[test]
    fn reliability_validates_and_cancels_without_mutating_geometry() -> anyhow::Result<()> {
        use super::ReliabilityMask;
        let depth: Vec<f32> = (0..256).map(|v| v as f32 / 256.).collect();
        let normal = vec![[0., 0., -1.]; 256];
        let visible = vec![true; 256];
        assert!(
            ReliabilityMask::from_geometry(&depth, &normal, &visible, 0, 1, &mut || Ok(()))
                .is_err()
        );
        assert!(
            ReliabilityMask::from_geometry(&depth, &normal, &visible, 16, 9, &mut || Ok(()))
                .is_err()
        );
        assert!(ReliabilityMask::from_geometry(
            &depth[..255],
            &normal,
            &visible,
            16,
            1,
            &mut || Ok(())
        )
        .is_err());
        assert!(
            ReliabilityMask::from_geometry(&depth, &normal, &[false; 256], 16, 1, &mut || Ok(()))
                .is_err()
        );
        let mut bad = depth.clone();
        bad[100] = f32::NAN;
        assert!(
            ReliabilityMask::from_geometry(&bad, &normal, &visible, 16, 1, &mut || Ok(())).is_err()
        );
        let mut calls = 0;
        let expected =
            ReliabilityMask::from_geometry(&depth, &normal, &visible, 16, 2, &mut || {
                calls += 1;
                Ok(())
            })?;
        for cancel_at in 0..calls {
            let mut remaining = cancel_at;
            let result =
                ReliabilityMask::from_geometry(&depth, &normal, &visible, 16, 2, &mut || {
                    anyhow::ensure!(remaining > 0, "cancelled");
                    remaining -= 1;
                    Ok(())
                });
            assert_eq!(result.err().unwrap().to_string(), "cancelled");
        }
        let actual =
            ReliabilityMask::from_geometry(&depth, &normal, &visible, 16, 2, &mut || Ok(()))?;
        assert_eq!(actual.visible, expected.visible);
        assert_eq!(actual.boundary, expected.boundary);
        assert_eq!(actual.cosine, expected.cosine);
        Ok(())
    }

    #[test]
    #[ignore = "requires retained Tencent real-mesh reliability oracle; CPU only"]
    fn real_mesh_reliability_matches_tencent() -> anyhow::Result<()> {
        use candle_core::Tensor;
        use std::{collections::HashMap, path::PathBuf};
        let oracle = PathBuf::from(std::env::var("MOLD_PAINT_RELIABILITY_ORACLE")?);
        let output = PathBuf::from(std::env::var("MOLD_PAINT_RELIABILITY_OUTPUT")?);
        std::fs::create_dir(&output)?;
        let metadata: serde_json::Value =
            serde_json::from_slice(&std::fs::read(oracle.join("completed.json"))?)?;
        let size = metadata["size"].as_u64().unwrap() as usize;
        let radius = metadata["radius"].as_u64().unwrap() as usize;
        let mut reports = Vec::new();
        for view in 0..6 {
            let fixture = candle_core::safetensors::load(
                oracle.join(format!("view.{view}.safetensors")),
                &Device::Cpu,
            )?;
            let floats = |field: &str| fixture[field].flatten_all()?.to_vec1::<f32>();
            let bools = |field: &str| -> candle_core::Result<Vec<bool>> {
                Ok(fixture[field]
                    .flatten_all()?
                    .to_vec1::<u8>()?
                    .iter()
                    .map(|v| *v != 0)
                    .collect())
            };
            let normal: Vec<[f32; 3]> = floats("normal")?
                .chunks_exact(3)
                .map(|v| [v[0], v[1], v[2]])
                .collect();
            let start = std::time::Instant::now();
            let mask = super::ReliabilityMask::from_geometry(
                &floats("depth")?,
                &normal,
                &bools("visible")?,
                size,
                radius,
                &mut || Ok(()),
            )?;
            let bits = |v: &[bool]| {
                Tensor::from_vec(
                    v.iter().map(|b| u8::from(*b)).collect::<Vec<_>>(),
                    (size, size, 1),
                    &Device::Cpu,
                )
            };
            candle_core::safetensors::save(
                &HashMap::from([
                    ("visible".to_string(), bits(&mask.visible)?),
                    ("boundary".to_string(), bits(&mask.boundary)?),
                    (
                        "cosine".to_string(),
                        Tensor::from_vec(mask.cosine.clone(), (size, size, 1), &Device::Cpu)?,
                    ),
                ]),
                output.join(format!("view.{view}.safetensors")),
            )?;
            let visible_diff = mask
                .visible
                .iter()
                .zip(bools("reliable")?)
                .filter(|(a, b)| **a != *b)
                .count();
            let boundary_diff = mask
                .boundary
                .iter()
                .zip(bools("boundary")?)
                .filter(|(a, b)| **a != *b)
                .count();
            let max = mask
                .cosine
                .iter()
                .zip(floats("cosine")?)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            let report = serde_json::json!({"view":view,"visible_mismatch":visible_diff,"boundary_mismatch":boundary_diff,"cosine_max":max,"seconds":start.elapsed().as_secs_f64()});
            eprintln!("{report}");
            reports.push(report);
        }
        std::fs::write(
            output.join("comparison.json"),
            serde_json::to_vec_pretty(&reports)?,
        )?;
        anyhow::ensure!(
            reports.iter().all(|r| r["visible_mismatch"] == 0
                && r["boundary_mismatch"] == 0
                && r["cosine_max"].as_f64().unwrap() <= 2e-6),
            "real-mesh reliability comparison failed; see retained comparison.json"
        );
        Ok(())
    }
}
