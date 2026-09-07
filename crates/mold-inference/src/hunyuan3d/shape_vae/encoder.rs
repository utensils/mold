//! Hunyuan3D shape-VAE mesh sampling and encoder.
//!
//! Tencent's 2.1 checkpoint is the first packaged Hunyuan3D checkpoint that
//! carries the VAE encoder weights. The encoder consumes 81,920 normalized
//! surface samples (position, face normal and a sharp-edge label), selects
//! 4,096 latent queries with farthest-point sampling, then applies one point
//! cross-attention block and eight self-attention blocks. The neural portion
//! lives below the deterministic geometry preparation so oracle captures can
//! freeze the sampled point set independently from framework RNG behavior.

use candle_core::{Error, Result};
use rand::{rngs::StdRng, Rng, SeedableRng};

use crate::hunyuan3d::mesh::Mesh;

const NORMALIZED_MESH_SCALE: f32 = 0.9999;

/// One input row for the shape-VAE point encoder.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfacePoint {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub sharp_edge: bool,
}

/// Published Hunyuan3D 2.1 shape-VAE encoder geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShapeVaeEncoderConfig {
    pub num_latents: usize,
    pub embed_dim: usize,
    pub width: usize,
    pub heads: usize,
    pub num_layers: usize,
    pub pc_size: usize,
    pub pc_sharpedge_size: usize,
    pub point_feats: usize,
    pub downsample_ratio: usize,
    pub num_freqs: usize,
}

impl ShapeVaeEncoderConfig {
    /// Configuration recorded by Tencent's 2.1 YAML and tensor inventory.
    pub const fn v2_1() -> Self {
        Self {
            num_latents: 4096,
            embed_dim: 64,
            width: 1024,
            heads: 16,
            num_layers: 8,
            pc_size: 81_920,
            pc_sharpedge_size: 0,
            point_feats: 4,
            downsample_ratio: 20,
            num_freqs: 8,
        }
    }

    pub const fn input_width(&self) -> usize {
        3 * (2 * self.num_freqs + 1) + self.point_feats
    }
}

fn invalid(message: impl Into<String>) -> Error {
    Error::Msg(message.into())
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn length(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn normalize(v: [f32; 3]) -> Option<[f32; 3]> {
    let len = length(v);
    (len.is_finite() && len > 0.0).then(|| [v[0] / len, v[1] / len, v[2] / len])
}

/// Uniformly sample a mesh surface after Tencent's centered 0.9999 scaling.
///
/// The seed belongs to mold's durable execution record. Upstream leaves both
/// trimesh and NumPy sampling process-random; making it explicit here is what
/// lets restart/retry reuse the same encoder input instead of changing shape.
pub fn sample_mesh_surface(mesh: &Mesh, count: usize, seed: u64) -> Result<Vec<SurfacePoint>> {
    mesh.validate()
        .map_err(|error| invalid(error.to_string()))?;
    if count == 0 {
        return Ok(Vec::new());
    }
    if mesh.is_empty() {
        return Err(invalid(
            "shape-VAE surface sampling requires a non-empty mesh",
        ));
    }

    let (min, max) = mesh.bounds();
    let center = [
        (min[0] + max[0]) * 0.5,
        (min[1] + max[1]) * 0.5,
        (min[2] + max[2]) * 0.5,
    ];
    let extent = (max[0] - min[0]).max(max[1] - min[1]).max(max[2] - min[2]);
    if !extent.is_finite() || extent <= 0.0 {
        return Err(invalid(
            "shape-VAE surface sampling requires a positive mesh extent",
        ));
    }
    let scale = 2.0 * NORMALIZED_MESH_SCALE / extent;
    let positions: Vec<[f32; 3]> = mesh
        .vertices
        .iter()
        .map(|v| {
            [
                (v[0] - center[0]) * scale,
                (v[1] - center[1]) * scale,
                (v[2] - center[2]) * scale,
            ]
        })
        .collect();

    let mut cumulative = Vec::with_capacity(mesh.faces.len());
    let mut normals = Vec::with_capacity(mesh.faces.len());
    let mut total = 0.0f64;
    for face in &mesh.faces {
        let a = positions[face[0] as usize];
        let b = positions[face[1] as usize];
        let c = positions[face[2] as usize];
        let raw = cross(sub(b, a), sub(c, a));
        let normal = normalize(raw).ok_or_else(|| {
            invalid("shape-VAE surface sampling does not accept degenerate triangles")
        })?;
        total += 0.5 * f64::from(length(raw));
        cumulative.push(total);
        normals.push(normal);
    }
    if !total.is_finite() || total <= 0.0 {
        return Err(invalid(
            "shape-VAE surface sampling requires positive surface area",
        ));
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let mut samples = Vec::with_capacity(count);
    for _ in 0..count {
        let pick = rng.gen::<f64>() * total;
        let face_index = cumulative.partition_point(|weight| *weight < pick);
        let face_index = face_index.min(mesh.faces.len() - 1);
        let face = mesh.faces[face_index];
        let a = positions[face[0] as usize];
        let b = positions[face[1] as usize];
        let c = positions[face[2] as usize];
        // Trimesh's triangle picker reflects the unit-square sample whenever
        // u + v > 1. This is uniform and keeps the upstream operation order.
        let mut u = rng.gen::<f32>();
        let mut v = rng.gen::<f32>();
        if u + v > 1.0 {
            u = 1.0 - u;
            v = 1.0 - v;
        }
        samples.push(SurfacePoint {
            position: [
                a[0] + u * (b[0] - a[0]) + v * (c[0] - a[0]),
                a[1] + u * (b[1] - a[1]) + v * (c[1] - a[1]),
                a[2] + u * (b[2] - a[2]) + v * (c[2] - a[2]),
            ],
            normal: normals[face_index],
            sharp_edge: false,
        });
    }
    Ok(samples)
}

/// Deterministic farthest-point sampling with PyTorch-compatible earliest
/// index tie breaking.
pub fn farthest_point_indices(
    points: &[[f32; 3]],
    count: usize,
    first: usize,
) -> Result<Vec<usize>> {
    if count > points.len() {
        return Err(invalid(format!(
            "cannot select {count} farthest points from {} inputs",
            points.len()
        )));
    }
    if count == 0 {
        return Ok(Vec::new());
    }
    if first >= points.len() {
        return Err(invalid("farthest-point start index is out of range"));
    }
    if !points.iter().flatten().all(|value| value.is_finite()) {
        return Err(invalid("farthest-point inputs must be finite"));
    }

    let mut selected = Vec::with_capacity(count);
    let mut distances = vec![f32::INFINITY; points.len()];
    let mut farthest = first;
    for _ in 0..count {
        selected.push(farthest);
        let centroid = points[farthest];
        for (distance, point) in distances.iter_mut().zip(points) {
            let delta = sub(*point, centroid);
            *distance = distance.min(length(delta));
        }
        farthest = distances
            .iter()
            .enumerate()
            .max_by(|(left_index, left), (right_index, right)| {
                left.total_cmp(right)
                    .then_with(|| right_index.cmp(left_index))
            })
            .map(|(index, _)| index)
            .unwrap_or(0);
    }
    Ok(selected)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hunyuan3d::mesh::Mesh;

    fn tetrahedron() -> Mesh {
        Mesh {
            vertices: vec![
                [1.0, 1.0, 1.0],
                [-1.0, -1.0, 1.0],
                [-1.0, 1.0, -1.0],
                [1.0, -1.0, -1.0],
            ],
            faces: vec![[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]],
            ..Mesh::default()
        }
    }

    #[test]
    fn farthest_point_sampling_uses_stable_first_and_tie_order() {
        let points = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ];
        assert_eq!(farthest_point_indices(&points, 3, 0).unwrap(), [0, 3, 1]);
    }

    #[test]
    fn surface_sampling_is_seeded_normalized_and_finite() {
        let mesh = tetrahedron();
        let first = sample_mesh_surface(&mesh, 128, 42).unwrap();
        let second = sample_mesh_surface(&mesh, 128, 42).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.len(), 128);
        assert!(first.iter().all(|sample| {
            sample
                .position
                .iter()
                .all(|v| v.is_finite() && v.abs() <= 0.999_901)
                && sample.normal.iter().all(|v| v.is_finite())
        }));
    }

    #[test]
    fn published_v21_encoder_contract_matches_checkpoint() {
        let cfg = ShapeVaeEncoderConfig::v2_1();
        assert_eq!(cfg.num_latents, 4096);
        assert_eq!(cfg.pc_size, 81_920);
        assert_eq!(cfg.pc_sharpedge_size, 0);
        assert_eq!(cfg.point_feats, 4);
        assert_eq!(cfg.downsample_ratio, 20);
        assert_eq!(cfg.num_layers, 8);
        assert_eq!(cfg.input_width(), 55);
    }
}
