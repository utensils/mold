//! Camera-space geometry feeding paint reliability and texture back-sampling.

use super::{
    paint_raster::{self, PreparedMesh},
    paint_reliability::ReliabilityMask,
};
use anyhow::{ensure, Result};

pub struct CameraGeometry {
    pub size: usize,
    pub depth: Vec<f32>,
    pub normal: Vec<[f32; 3]>,
    pub visible: Vec<bool>,
}

impl CameraGeometry {
    /// Tencent MeshRender.back_project:1141-1180: face normals are computed
    /// AFTER transforming vertices to camera space, and depth is interpolated
    /// from camera-space Z. Neither is the gallery raster's normal/depth field.
    pub fn render(
        mesh: &PreparedMesh,
        elevation: f32,
        azimuth: f32,
        size: usize,
        checkpoint: &mut dyn FnMut() -> Result<()>,
    ) -> Result<Self> {
        ensure!((1..=2048).contains(&size), "invalid paint camera size");
        checkpoint()?;
        mesh.mesh.validate()?;
        let mut buffers = paint_raster::render_with_checkpoint(
            mesh,
            elevation,
            azimuth,
            size as u32,
            checkpoint,
        )?;
        let matrix = paint_raster::view_matrix(elevation, azimuth);
        checkpoint()?;
        let mut camera = Vec::with_capacity(mesh.mesh.vertices.len());
        for (index, p) in mesh.mesh.vertices.iter().enumerate() {
            if index.is_multiple_of(4096) {
                checkpoint()?;
            }
            camera.push([0, 1, 2].map(|row| {
                matrix[row][0] * p[0]
                    + matrix[row][1] * p[1]
                    + matrix[row][2] * p[2]
                    + matrix[row][3]
            }));
        }
        checkpoint()?;
        let mut normals = Vec::with_capacity(mesh.mesh.faces.len());
        for (index, face) in mesh.mesh.faces.iter().enumerate() {
            if index.is_multiple_of(4096) {
                checkpoint()?;
            }
            let [a, b, c] = face.map(|v| camera[v as usize]);
            let ab = [0, 1, 2].map(|i| b[i] - a[i]);
            let ac = [0, 1, 2].map(|i| c[i] - a[i]);
            let n = [
                ab[1] * ac[2] - ab[2] * ac[1],
                ab[2] * ac[0] - ab[0] * ac[2],
                ab[0] * ac[1] - ab[1] * ac[0],
            ];
            let norm = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(1e-12);
            normals.push(n.map(|v| v / norm));
        }
        for index in 0..buffers.mask.len() {
            if index.is_multiple_of(4096) {
                checkpoint()?;
            }
            if buffers.mask[index] {
                let face_index = buffers.face_ids[index] as usize;
                let [a, b, c] = mesh.mesh.faces[face_index].map(|v| camera[v as usize][2]);
                let [x, y, z] = buffers.barycentric[index];
                // custom_rasterizer/render.py:30-31 multiplies separately,
                // then Torch's three-element CUDA reduction sums 0 + 2 + 1.
                // A different order moves normalized depth bytes and Canny.
                buffers.depth[index] = (x * a + z * c) + y * b;
                buffers.normal[index] = normals[face_index];
            } else {
                buffers.depth[index] = 0.;
                buffers.normal[index] = [0.; 3];
            }
        }
        checkpoint()?;
        Ok(Self {
            size,
            depth: buffers.depth,
            normal: buffers.normal,
            visible: buffers.mask,
        })
    }

    pub fn reliability(
        &self,
        checkpoint: &mut dyn FnMut() -> Result<()>,
    ) -> Result<ReliabilityMask> {
        ReliabilityMask::from_geometry(
            &self.depth,
            &self.normal,
            &self.visible,
            self.size,
            self.size / 256,
            checkpoint,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::super::{mesh::Mesh, paint_raster::prepare_mesh, paint_views::candidate_views};
    use candle_core::{Device, Tensor};
    use std::collections::HashMap;

    fn mesh_from_fixture(fixture: &HashMap<String, Tensor>) -> anyhow::Result<Mesh> {
        Ok(Mesh {
            vertices: fixture["vertices"]
                .flatten_all()?
                .to_vec1::<f32>()?
                .chunks_exact(3)
                .map(|v| [v[0], v[1], v[2]])
                .collect(),
            faces: fixture["faces"]
                .flatten_all()?
                .to_vec1::<i32>()?
                .chunks_exact(3)
                .map(|v| [v[0] as u32, v[1] as u32, v[2] as u32])
                .collect(),
            uvs: Some(
                fixture["uv"]
                    .flatten_all()?
                    .to_vec1::<f32>()?
                    .chunks_exact(2)
                    .map(|v| [v[0], v[1]])
                    .collect(),
            ),
            ..Mesh::default()
        })
    }

    #[test]
    fn camera_geometry_and_reliability_match_tencent() -> anyhow::Result<()> {
        let fixture = candle_core::safetensors::load_buffer(
            include_bytes!("../../../../tests/fixtures/hunyuan3d/paint-camera.safetensors"),
            &Device::Cpu,
        )?;
        let mesh = prepare_mesh(&mesh_from_fixture(&fixture)?)?;
        for (index, view) in candidate_views().iter().take(6).enumerate() {
            let geometry = super::CameraGeometry::render(
                &mesh,
                view.elevation,
                view.azimuth,
                64,
                &mut || Ok(()),
            )?;
            let floats = |name: &str| {
                fixture[&format!("view.{index}.{name}")]
                    .flatten_all()?
                    .to_vec1::<f32>()
            };
            let bools = |name: &str| -> candle_core::Result<Vec<bool>> {
                Ok(fixture[&format!("view.{index}.{name}")]
                    .flatten_all()?
                    .to_vec1::<u8>()?
                    .iter()
                    .map(|v| *v != 0)
                    .collect())
            };
            assert_eq!(geometry.visible, bools("visible")?, "view{index}: coverage");
            for (name, values) in [
                ("depth", geometry.depth.clone()),
                (
                    "normal",
                    geometry.normal.iter().flatten().copied().collect(),
                ),
            ] {
                let max = values
                    .iter()
                    .zip(floats(name)?)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                assert!(max <= 2e-6, "view{index} {name}: {max}");
            }
            let mask = geometry.reliability(&mut || Ok(()))?;
            assert_eq!(mask.visible, bools("reliable")?, "view{index}: reliable");
            assert_eq!(mask.boundary, bools("boundary")?, "view{index}: boundary");
            let max = mask
                .cosine
                .iter()
                .zip(floats("cosine")?)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(max <= 2e-6, "view{index} cosine: {max}");
        }
        Ok(())
    }

    #[test]
    fn camera_render_checks_bounds_and_cancels() -> anyhow::Result<()> {
        let fixture = candle_core::safetensors::load_buffer(
            include_bytes!("../../../../tests/fixtures/hunyuan3d/paint-camera.safetensors"),
            &Device::Cpu,
        )?;
        let mesh = prepare_mesh(&mesh_from_fixture(&fixture)?)?;
        assert!(super::CameraGeometry::render(&mesh, 0., 0., 0, &mut || Ok(())).is_err());
        assert!(super::CameraGeometry::render(&mesh, 0., 0., 2049, &mut || Ok(())).is_err());
        assert!(super::CameraGeometry::render(&mesh, f32::NAN, 0., 64, &mut || Ok(())).is_err());
        let mut calls = 0;
        let expected = super::CameraGeometry::render(&mesh, 0., 0., 64, &mut || {
            calls += 1;
            Ok(())
        })?;
        for cancel_at in 0..calls {
            let mut remaining = cancel_at;
            let result = super::CameraGeometry::render(&mesh, 0., 0., 64, &mut || {
                anyhow::ensure!(remaining > 0, "cancelled");
                remaining -= 1;
                Ok(())
            });
            assert_eq!(result.err().unwrap().to_string(), "cancelled");
        }
        let actual = super::CameraGeometry::render(&mesh, 0., 0., 64, &mut || Ok(()))?;
        assert_eq!(actual.depth, expected.depth);
        assert_eq!(actual.normal, expected.normal);
        assert_eq!(actual.visible, expected.visible);
        Ok(())
    }

    #[test]
    #[ignore = "requires retained Tencent real-mesh camera oracle; CPU only"]
    fn real_mesh_camera_and_reliability_match_tencent() -> anyhow::Result<()> {
        use std::path::PathBuf;
        let oracle = PathBuf::from(std::env::var("MOLD_PAINT_CAMERA_ORACLE")?);
        let output = PathBuf::from(std::env::var("MOLD_PAINT_CAMERA_OUTPUT")?);
        std::fs::create_dir(&output)?;
        let metadata: serde_json::Value =
            serde_json::from_slice(&std::fs::read(oracle.join("completed.json"))?)?;
        let size = metadata["size"].as_u64().unwrap() as usize;
        let mesh = prepare_mesh(&mesh_from_fixture(&candle_core::safetensors::load(
            oracle.join("mesh.safetensors"),
            &Device::Cpu,
        )?)?)?;
        let mut reports = Vec::new();
        for (index, view) in candidate_views().iter().take(6).enumerate() {
            let fixture = candle_core::safetensors::load(
                oracle.join(format!("view.{index}.safetensors")),
                &Device::Cpu,
            )?;
            let floats = |name: &str| fixture[name].flatten_all()?.to_vec1::<f32>();
            let bools = |name: &str| -> candle_core::Result<Vec<bool>> {
                Ok(fixture[name]
                    .flatten_all()?
                    .to_vec1::<u8>()?
                    .iter()
                    .map(|v| *v != 0)
                    .collect())
            };
            let start = std::time::Instant::now();
            let geometry = super::CameraGeometry::render(
                &mesh,
                view.elevation,
                view.azimuth,
                size,
                &mut || Ok(()),
            )?;
            let mask = geometry.reliability(&mut || Ok(()))?;
            let bits = |values: &[bool]| {
                Tensor::from_vec(
                    values.iter().map(|v| u8::from(*v)).collect::<Vec<_>>(),
                    (size, size, 1),
                    &Device::Cpu,
                )
            };
            let values = HashMap::from([
                (
                    "depth".to_string(),
                    Tensor::from_vec(geometry.depth.clone(), (size, size, 1), &Device::Cpu)?,
                ),
                (
                    "normal".to_string(),
                    Tensor::from_vec(
                        geometry
                            .normal
                            .iter()
                            .flatten()
                            .copied()
                            .collect::<Vec<_>>(),
                        (size, size, 3),
                        &Device::Cpu,
                    )?,
                ),
                ("visible".to_string(), bits(&geometry.visible)?),
                ("reliable".to_string(), bits(&mask.visible)?),
                ("boundary".to_string(), bits(&mask.boundary)?),
                (
                    "cosine".to_string(),
                    Tensor::from_vec(mask.cosine.clone(), (size, size, 1), &Device::Cpu)?,
                ),
            ]);
            candle_core::safetensors::save(
                &values,
                output.join(format!("view.{index}.safetensors")),
            )?;
            let mut report =
                serde_json::json!({"view":index,"seconds":start.elapsed().as_secs_f64()});
            for field in ["depth", "normal", "cosine"] {
                let max = values[field]
                    .flatten_all()?
                    .to_vec1::<f32>()?
                    .iter()
                    .zip(floats(field)?)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                report[format!("{field}_max")] = max.into();
            }
            for (field, actual) in [
                ("visible", &geometry.visible),
                ("reliable", &mask.visible),
                ("boundary", &mask.boundary),
            ] {
                report[format!("{field}_mismatch")] = actual
                    .iter()
                    .zip(bools(field)?)
                    .filter(|(a, b)| **a != *b)
                    .count()
                    .into();
            }
            eprintln!("{report}");
            reports.push(report);
        }
        std::fs::write(
            output.join("comparison.json"),
            serde_json::to_vec_pretty(&reports)?,
        )?;
        anyhow::ensure!(
            reports
                .iter()
                .all(|r| ["depth_max", "normal_max", "cosine_max"]
                    .iter()
                    .all(|k| r[k].as_f64().unwrap() <= 2e-6)
                    && ["visible_mismatch", "reliable_mismatch", "boundary_mismatch"]
                        .iter()
                        .all(|k| r[k] == 0)),
            "real mesh camera comparison failed; retained comparison.json"
        );
        Ok(())
    }
}
