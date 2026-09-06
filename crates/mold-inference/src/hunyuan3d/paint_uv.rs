//! UV-space surface geometry for paint back-projection and hole filling.

use super::{
    paint_raster::{view_matrix, PreparedMesh},
    raster::{render_projected_with_checkpoint, Culling, ScreenVertex},
};
use anyhow::{ensure, Result};

/// Only covered texels, in ascending row-major order. Normals are geometric
/// face normals, including zero for a degenerate face (Torch normalize eps).
pub struct UvGeometry {
    pub size: u32,
    pub texels: Vec<usize>,
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
}

fn screen_coordinate(ndc: f32, size: u32) -> f32 {
    // Tencent rasterizer_gpu.cu contracts the outer pixel mapping to FFMA in
    // both coverage and barycentric interpolation. Separate rounding changes
    // UV weights enough to cross byte boundaries before texture hole filling.
    (ndc * 0.5 + 0.5).mul_add((size - 1) as f32, 0.5)
}

impl UvGeometry {
    /// Tencent MeshRender.extract_textiles: rasterize the already-flipped UVs
    /// while interpolating paint-frame positions; keep the source face order.
    pub fn extract(
        mesh: &PreparedMesh,
        size: u32,
        checkpoint: &mut dyn FnMut() -> Result<()>,
    ) -> Result<Self> {
        ensure!((1..=4096).contains(&size), "invalid paint UV texture size");
        mesh.mesh.validate()?;
        let uv = mesh
            .mesh
            .uvs
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("paint mesh has no UV coordinates"))?;
        ensure!(
            uv.iter()
                .flatten()
                .all(|v| v.is_finite() && (0. ..=1.).contains(v)),
            "paint UV coordinates must be finite unit values"
        );
        checkpoint()?;
        let mut screen = Vec::with_capacity(uv.len());
        for (index, uv) in uv.iter().enumerate() {
            if index.is_multiple_of(4096) {
                checkpoint()?;
            }
            let ndc = uv.map(|v| v * 2. - 1.);
            screen.push(ScreenVertex {
                x: screen_coordinate(ndc[0], size),
                y: screen_coordinate(ndc[1], size),
                depth: -0.49999 + 0.5,
                inv_w: 1.,
            });
        }
        let mut normals = Vec::with_capacity(mesh.mesh.faces.len());
        for (index, face) in mesh.mesh.faces.iter().enumerate() {
            if index.is_multiple_of(4096) {
                checkpoint()?;
            }
            let [a, b, c] = face.map(|i| mesh.mesh.vertices[i as usize]);
            let ab = [0, 1, 2].map(|i| b[i] - a[i]);
            let ac = [0, 1, 2].map(|i| c[i] - a[i]);
            let cross = [
                ab[1] * ac[2] - ab[2] * ac[1],
                ab[2] * ac[0] - ab[0] * ac[2],
                ab[0] * ac[1] - ab[1] * ac[0],
            ];
            let length = cross.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
            normals.push(cross.map(|v| v / length));
        }
        let mut buffers = render_projected_with_checkpoint(
            &mesh.mesh,
            &screen,
            Culling::None,
            [size, size],
            true,
            checkpoint,
        )?;
        let mut texels = Vec::new();
        let mut covered = 0;
        // Compact in place so a 4096² raster does not allocate another pair of
        // full-size position/normal arrays while its G-buffers are resident.
        for pixel in 0..buffers.mask.len() {
            if pixel.is_multiple_of(4096) {
                checkpoint()?;
            }
            if buffers.mask[pixel] {
                texels.push(pixel);
                buffers.position[covered] = buffers.position[pixel];
                buffers.normal[covered] = normals[buffers.face_ids[pixel] as usize];
                covered += 1;
            }
        }
        buffers.position.truncate(covered);
        buffers.normal.truncate(covered);
        checkpoint()?;
        Ok(Self {
            size,
            texels,
            positions: buffers.position,
            normals: buffers.normal,
        })
    }

    /// MeshRender.back_project uses camera-space Z and orthographic XY;
    /// its image projection intentionally leaves depth unnormalized.
    pub fn project(
        &self,
        elevation: f32,
        azimuth: f32,
        checkpoint: &mut dyn FnMut() -> Result<()>,
    ) -> Result<Vec<[f32; 3]>> {
        ensure!(
            elevation.is_finite() && elevation.abs() <= 90. && azimuth.is_finite(),
            "invalid paint camera angle"
        );
        checkpoint()?;
        let matrix = view_matrix(elevation, azimuth);
        let mut output = Vec::with_capacity(self.positions.len());
        for (index, position) in self.positions.iter().enumerate() {
            if index.is_multiple_of(4096) {
                checkpoint()?;
            }
            let mut camera = [0, 1, 2].map(|row| {
                matrix[row][0] * position[0]
                    + matrix[row][1] * position[1]
                    + matrix[row][2] * position[2]
                    + matrix[row][3]
            });
            camera[0] *= 2. / 1.2;
            camera[1] *= 2. / 1.2;
            output.push(camera);
        }
        checkpoint()?;
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::super::{mesh::Mesh, paint_raster::prepare_mesh, paint_views::candidate_views};
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn uv_screen_map_retains_cuda_fused_rounding() {
        // Actual chair UV component; both installed CUDA kernels use FFMA.
        assert_eq!(
            screen_coordinate(f32::from_bits(3210749768), 1024).to_bits(),
            1115702023
        );
    }

    #[test]
    fn paint_uv_geometry_and_cameras_match_tencent() -> anyhow::Result<()> {
        let fixture = candle_core::safetensors::load_buffer(
            include_bytes!("../../../../tests/fixtures/hunyuan3d/paint-uv.safetensors"),
            &Device::Cpu,
        )?;
        let triples = |tensor: &Tensor| -> anyhow::Result<Vec<[f32; 3]>> {
            Ok(tensor
                .to_vec2::<f32>()?
                .iter()
                .map(|v| [v[0], v[1], v[2]])
                .collect())
        };
        let mesh = Mesh {
            vertices: triples(&fixture["vertices"])?,
            faces: fixture["faces"]
                .to_vec2::<i32>()?
                .iter()
                .map(|v| [v[0] as u32, v[1] as u32, v[2] as u32])
                .collect(),
            uvs: Some(
                fixture["uv"]
                    .to_vec2::<f32>()?
                    .iter()
                    .map(|v| [v[0], v[1]])
                    .collect(),
            ),
            ..Mesh::default()
        };
        let mesh = prepare_mesh(&mesh)?;
        let uv = UvGeometry::extract(&mesh, 32, &mut || Ok(()))?;
        assert_eq!(
            uv.texels,
            fixture["texels"]
                .to_vec1::<i64>()?
                .iter()
                .map(|v| *v as usize)
                .collect::<Vec<_>>()
        );
        let compare = |actual: &[[f32; 3]], name: &str| -> anyhow::Result<()> {
            let expected = triples(&fixture[name])?;
            assert_eq!(actual.len(), expected.len());
            let max = actual
                .iter()
                .flatten()
                .zip(expected.iter().flatten())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(max <= 2e-6, "{name}: max={max}");
            Ok(())
        };
        compare(&uv.positions, "positions")?;
        compare(&uv.normals, "normals")?;
        assert!(
            uv.normals.contains(&[0.; 3]),
            "degenerate face must retain its zero normal"
        );
        for (index, view) in candidate_views().iter().take(6).enumerate() {
            compare(
                &uv.project(view.elevation, view.azimuth, &mut || Ok(()))?,
                &format!("projected.{index}"),
            )?;
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires retained Tencent UV oracle; CPU only, includes production texture sizes"]
    fn real_mesh_paint_uv_matches_tencent() -> anyhow::Result<()> {
        use std::{collections::HashMap, path::PathBuf};
        let oracle = PathBuf::from(std::env::var("MOLD_PAINT_UV_ORACLE")?);
        let output = PathBuf::from(std::env::var("MOLD_PAINT_UV_OUTPUT")?);
        std::fs::create_dir(&output)?;
        let metadata: serde_json::Value =
            serde_json::from_slice(&std::fs::read(oracle.join("paint-uv.json"))?)?;
        let size = metadata["size"]
            .as_u64()
            .ok_or_else(|| anyhow::anyhow!("missing UV size"))? as u32;
        let fixture =
            candle_core::safetensors::load(oracle.join("paint-uv.safetensors"), &Device::Cpu)?;
        let mesh = Mesh {
            vertices: fixture["vertices"]
                .flatten_all()?
                .to_vec1::<f32>()?
                .as_chunks::<3>()
                .0
                .iter()
                .map(|v| [v[0], v[1], v[2]])
                .collect(),
            faces: fixture["faces"]
                .flatten_all()?
                .to_vec1::<i32>()?
                .as_chunks::<3>()
                .0
                .iter()
                .map(|v| [v[0] as u32, v[1] as u32, v[2] as u32])
                .collect(),
            uvs: Some(
                fixture["uv"]
                    .flatten_all()?
                    .to_vec1::<f32>()?
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|v| [v[0], v[1]])
                    .collect(),
            ),
            ..Mesh::default()
        };
        let start = std::time::Instant::now();
        let prepared = prepare_mesh(&mesh)?;
        let uv = UvGeometry::extract(&prepared, size, &mut || Ok(()))?;
        if std::env::var_os("MOLD_PAINT_UV_TRACE").is_some() {
            let screen: Vec<_> = prepared
                .mesh
                .uvs
                .as_ref()
                .unwrap()
                .iter()
                .map(|uv| {
                    let ndc = uv.map(|v| v * 2. - 1.);
                    ScreenVertex {
                        x: screen_coordinate(ndc[0], size),
                        y: screen_coordinate(ndc[1], size),
                        depth: -0.49999 + 0.5,
                        inv_w: 1.,
                    }
                })
                .collect();
            let buffers = render_projected_with_checkpoint(
                &prepared.mesh,
                &screen,
                Culling::None,
                [size, size],
                true,
                &mut || Ok(()),
            )?;
            let trace = HashMap::from([
                (
                    "barycentric".to_string(),
                    Tensor::from_vec(
                        buffers
                            .barycentric
                            .iter()
                            .flatten()
                            .copied()
                            .collect::<Vec<_>>(),
                        (size as usize, size as usize, 3),
                        &Device::Cpu,
                    )?,
                ),
                (
                    "face_ids".to_string(),
                    Tensor::from_vec(
                        buffers.face_ids,
                        (size as usize, size as usize),
                        &Device::Cpu,
                    )?,
                ),
                (
                    "mask".to_string(),
                    Tensor::from_vec(
                        buffers
                            .mask
                            .iter()
                            .map(|v| u8::from(*v))
                            .collect::<Vec<_>>(),
                        (size as usize, size as usize),
                        &Device::Cpu,
                    )?,
                ),
                (
                    "vertices".to_string(),
                    Tensor::from_vec(
                        prepared
                            .mesh
                            .vertices
                            .iter()
                            .flatten()
                            .copied()
                            .collect::<Vec<_>>(),
                        (prepared.mesh.vertices.len(), 3),
                        &Device::Cpu,
                    )?,
                ),
                (
                    "screen".to_string(),
                    Tensor::from_vec(
                        screen
                            .iter()
                            .flat_map(|p| [p.x, p.y, p.depth, p.inv_w])
                            .collect::<Vec<_>>(),
                        (screen.len(), 4),
                        &Device::Cpu,
                    )?,
                ),
            ]);
            candle_core::safetensors::save(&trace, output.join("raster-trace.safetensors"))?;
        }
        let tensor = |values: &[[f32; 3]]| {
            Tensor::from_vec(
                values.iter().flatten().copied().collect::<Vec<_>>(),
                (values.len(), 3),
                &Device::Cpu,
            )
        };
        candle_core::safetensors::save(
            &HashMap::from([
                ("positions".to_string(), tensor(&uv.positions)?),
                ("normals".to_string(), tensor(&uv.normals)?),
                (
                    "texels".to_string(),
                    Tensor::from_vec(
                        uv.texels.iter().map(|v| *v as i64).collect::<Vec<_>>(),
                        uv.texels.len(),
                        &Device::Cpu,
                    )?,
                ),
            ]),
            output.join("geometry.safetensors"),
        )?;
        let expected_texels = fixture["texels"].to_vec1::<i64>()?;
        ensure!(
            uv.texels.len() == expected_texels.len(),
            "UV coverage count differs: {} vs {}",
            uv.texels.len(),
            expected_texels.len()
        );
        ensure!(
            uv.texels
                .iter()
                .zip(expected_texels)
                .all(|(a, b)| *a as i64 == b),
            "UV coverage indices differ"
        );
        let mut measurements = Vec::new();
        let mut compare = |values: &[[f32; 3]], name: &str| -> anyhow::Result<()> {
            let expected = fixture[name].flatten_all()?.to_vec1::<f32>()?;
            let mut max = 0f32;
            for (&a, b) in values.iter().flatten().zip(expected) {
                ensure!(a.is_finite() && b.is_finite(), "nonfinite {name}");
                max = max.max((a - b).abs());
            }
            eprintln!("{name}: max={max}");
            measurements.push(serde_json::json!({"name":name,"max":max}));
            ensure!(max <= 2e-6, "{name} max {max} exceeds2e-6");
            Ok(())
        };
        compare(&uv.positions, "positions")?;
        compare(&uv.normals, "normals")?;
        for (index, view) in candidate_views().iter().take(6).enumerate() {
            let projected = uv.project(view.elevation, view.azimuth, &mut || Ok(()))?;
            let name = format!("projected.{index}");
            candle_core::safetensors::save(
                &HashMap::from([(name.clone(), tensor(&projected)?)]),
                output.join(format!("{name}.safetensors")),
            )?;
            compare(&projected, &name)?;
        }
        std::fs::write(
            output.join("comparison.json"),
            serde_json::to_vec_pretty(
                &serde_json::json!({"size":size,"texels":uv.texels.len(),"seconds":start.elapsed().as_secs_f64(),"measurements":measurements}),
            )?,
        )?;
        Ok(())
    }
}
