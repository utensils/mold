//! Tencent paint coordinates and CPU conditioning buffers.
use super::{
    mesh::Mesh,
    raster::{self, Culling, GBuffers, ScreenVertex},
};
use anyhow::{ensure, Result};

/// Reversible normalization in Tencent's paint frame (-x, z, -y).
/// `MeshRender.py:700-717` centers the bounds and fits the bounding sphere to
/// diameter 1.15; this is independent of the existing gallery camera framing.
pub struct PreparedMesh {
    pub mesh: Mesh,
    pub center: [f32; 3],
    pub scale: f32,
}

pub fn prepare_mesh(source: &Mesh) -> Result<PreparedMesh> {
    source.validate()?;
    ensure!(
        !source.is_empty() && source.faces.len() <= 2_000_000 && source.vertices.len() <= 6_000_000,
        "paint mesh exceeds the geometry budget or is empty"
    );
    let mut mesh = source.clone();
    for p in &mut mesh.vertices {
        *p = [-p[0], p[2], -p[1]];
    }
    // Tencent's default is face shading, independent of imported normals.
    mesh.normals = None;
    if let Some(uvs) = &mut mesh.uvs {
        for uv in uvs {
            ensure!(uv.iter().all(|v| v.is_finite()), "nonfinite mesh UV");
            uv[1] = 1. - uv[1];
        }
    }
    let (min, max) = mesh.bounds();
    let center = [0, 1, 2].map(|i| (min[i] + max[i]) * 0.5);
    let diameter = mesh
        .vertices
        .iter()
        .map(|p| {
            (0..3)
                .map(|i| (p[i] - center[i]).powi(2))
                .sum::<f32>()
                .sqrt()
        })
        .fold(0., f32::max)
        * 2.;
    ensure!(
        diameter.is_finite() && diameter > 0.,
        "mesh has no finite positive diameter"
    );
    let scale = 1.15 / diameter;
    ensure!(scale.is_finite(), "paint normalization scale is not finite");
    for p in &mut mesh.vertices {
        *p = [0, 1, 2].map(|i| (p[i] - center[i]) * scale);
    }
    Ok(PreparedMesh {
        mesh,
        center,
        scale,
    })
}

impl PreparedMesh {
    pub fn restore_position(&self, position: [f32; 3]) -> [f32; 3] {
        let p = [0, 1, 2].map(|i| position[i] / self.scale + self.center[i]);
        [-p[0], -p[2], p[1]]
    }
}

/// `camera_utils.py:34-71` uses Z-up inside the paint frame. Keep f64 for
/// NumPy-equivalent camera construction, then cast once before projection.
pub(super) fn view_matrix(elevation: f32, azimuth: f32) -> [[f32; 4]; 4] {
    let elev = -f64::from(elevation).to_radians();
    let azim = (f64::from(azimuth) + 90.).to_radians();
    let eye = [
        1.45 * elev.cos() * azim.cos(),
        1.45 * elev.cos() * azim.sin(),
        1.45 * elev.sin(),
    ];
    let unit = |v: [f64; 3]| {
        let length = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        v.map(|x| x / length)
    };
    let look = unit(eye.map(|x| -x));
    let right = unit([look[1], -look[0], 0.]);
    let up = unit([
        right[1] * look[2] - right[2] * look[1],
        right[2] * look[0] - right[0] * look[2],
        right[0] * look[1] - right[1] * look[0],
    ]);
    let basis = [right, up, look.map(|x| -x)];
    let mut matrix = [[0.; 4]; 4];
    for i in 0..3 {
        for j in 0..3 {
            matrix[i][j] = basis[i][j] as f32;
        }
        matrix[i][3] = -(0..3).map(|j| basis[i][j] * eye[j]).sum::<f64>() as f32;
    }
    matrix[3][3] = 1.;
    matrix
}

pub fn render(mesh: &PreparedMesh, elevation: f32, azimuth: f32, size: u32) -> Result<GBuffers> {
    ensure!(
        (1..=2048).contains(&size),
        "paint raster size must be 1 through 2048"
    );
    ensure!(
        elevation.is_finite() && azimuth.is_finite() && elevation.abs() <= 90.,
        "invalid paint camera angle"
    );
    let matrix = view_matrix(elevation, azimuth);
    let screen: Vec<_> = mesh
        .mesh
        .vertices
        .iter()
        .map(|p| {
            let camera = [0, 1, 2].map(|i| {
                matrix[i][0] * p[0] + matrix[i][1] * p[1] + matrix[i][2] * p[2] + matrix[i][3]
            });
            let ndc_z = (-2. / 99.9) * camera[2] - (100.1 / 99.9);
            ScreenVertex {
                // Upstream rasterizer_gpu.cu:93-95 maps BOTH axes without a Y
                // flip and places NDC endpoints at pixel centers.
                x: (camera[0] * (2. / 1.2) * 0.5 + 0.5) * (size - 1) as f32 + 0.5,
                y: (camera[1] * (2. / 1.2) * 0.5 + 0.5) * (size - 1) as f32 + 0.5,
                depth: ndc_z * 0.49999 + 0.5,
                inv_w: 1.,
            }
        })
        .collect();
    Ok(raster::render_projected(
        &mesh.mesh,
        &screen,
        Culling::None,
        size,
        size,
        true,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn matches_tencent_cuda_normalization_and_thirty_views() {
        let tensors = candle_core::safetensors::load_buffer(
            include_bytes!("../../../../tests/fixtures/hunyuan3d/paint-raster.safetensors"),
            &Device::Cpu,
        )
        .unwrap();
        let vectors = |name: &str| -> Vec<[f32; 3]> {
            tensors[name]
                .reshape(((), 3))
                .unwrap()
                .to_vec2::<f32>()
                .unwrap()
                .into_iter()
                .map(|v| v.try_into().unwrap())
                .collect()
        };
        let mesh = Mesh {
            vertices: vectors("vertices"),
            faces: tensors["faces"]
                .to_dtype(DType::U32)
                .unwrap()
                .to_vec2::<u32>()
                .unwrap()
                .into_iter()
                .map(|v| v.try_into().unwrap())
                .collect(),
            ..Mesh::default()
        };
        let prepared = prepare_mesh(&mesh).unwrap();
        for (original, normalized) in mesh.vertices.iter().zip(&prepared.mesh.vertices) {
            let restored = prepared.restore_position(*normalized);
            for axis in 0..3 {
                assert!((original[axis] - restored[axis]).abs() < 1e-6);
            }
        }
        for (a, b) in prepared.mesh.vertices.iter().zip(vectors("normalized")) {
            for i in 0..3 {
                assert!((a[i] - b[i]).abs() < 1e-6);
            }
        }
        let metadata: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../tests/fixtures/hunyuan3d/paint-raster.json"
        ))
        .unwrap();
        for (index, view) in metadata["views"].as_array().unwrap().iter().enumerate() {
            let matrix = view_matrix(
                view[0].as_f64().unwrap() as f32,
                view[1].as_f64().unwrap() as f32,
            );
            let expected = tensors[&format!("view.{index}.matrix")]
                .to_vec2::<f32>()
                .unwrap();
            for row in 0..4 {
                for col in 0..4 {
                    assert!((matrix[row][col] - expected[row][col]).abs() < 1e-6);
                }
            }
            let buffers = render(
                &prepared,
                view[0].as_f64().unwrap() as f32,
                view[1].as_f64().unwrap() as f32,
                32,
            )
            .unwrap();
            let normals = vectors(&format!("view.{index}.normal"));
            let positions = vectors(&format!("view.{index}.position"));
            let faces = tensors[&format!("view.{index}.face")]
                .flatten_all()
                .unwrap()
                .to_dtype(DType::U32)
                .unwrap()
                .to_vec1::<u32>()
                .unwrap();
            let mut boundary_differences = 0;
            for pixel in 0..1024 {
                let actual_face = if buffers.mask[pixel] {
                    buffers.face_ids[pixel] + 1
                } else {
                    0
                };
                if actual_face != faces[pixel] {
                    boundary_differences += 1;
                    continue;
                }
                if actual_face == 0 {
                    continue;
                }
                for axis in 0..3 {
                    let normal = (buffers.normal[pixel][axis] + 1.) * 0.5;
                    let position = 0.5 - buffers.position[pixel][axis] / 1.15;
                    assert!(
                        (normal - normals[pixel][axis]).abs() < 3e-5,
                        "view {index} normal at {pixel}"
                    );
                    assert!(
                        (position - positions[pixel][axis]).abs() < 3e-5,
                        "view {index} position at {pixel}: {position} vs {}",
                        positions[pixel][axis]
                    );
                }
            }
            assert!(
                boundary_differences <= 4,
                "view {index}: {boundary_differences} different face pixels"
            );
        }
    }
}
