//! Hunyuan3D shape-VAE mesh sampling and encoder.
//!
//! Tencent's 2.1 checkpoint is the first packaged Hunyuan3D checkpoint that
//! carries the VAE encoder weights. The encoder consumes 81,920 normalized
//! surface samples (position, face normal and a sharp-edge label), selects
//! 4,096 latent queries with farthest-point sampling, then applies one point
//! cross-attention block and eight self-attention blocks. The neural portion
//! lives below the deterministic geometry preparation so oracle captures can
//! freeze the sampled point set independently from framework RNG behavior.

use candle_core::{DType, Device, Error, Result, Tensor, D};
use candle_nn::{layer_norm, linear, LayerNorm, LayerNormConfig, Linear, Module, VarBuilder};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rayon::prelude::*;

use crate::hunyuan3d::mesh::Mesh;

use super::{
    FourierEmbedder, ResidualAttentionBlock, ResidualCrossAttentionBlock, ShapeVaeConfig,
    LN_POST_EPS,
};

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

    fn network_config(&self) -> ShapeVaeConfig {
        ShapeVaeConfig {
            num_latents: self.num_latents,
            embed_dim: self.embed_dim,
            width: self.width,
            heads: self.heads,
            num_decoder_layers: self.num_layers,
            num_freqs: self.num_freqs,
            include_pi: false,
            qkv_bias: false,
            qk_norm: true,
            mlp_expand_ratio: 4,
            geo_decoder_mlp_expand_ratio: 4,
            geo_decoder_ln_post: true,
            out_channels: 1,
            scale_factor: ShapeVaeConfig::v2_1().scale_factor,
        }
    }
}

/// Posterior statistics and the selected latent value in upstream `[B,N,C]`
/// layout. [`Self::for_decoder`] returns mold's decoder-facing `[B,C,N]`.
#[derive(Debug)]
pub struct EncodedShapeLatents {
    pub mean: Tensor,
    pub logvar: Tensor,
    pub latents: Tensor,
    #[cfg(feature = "dev-bins")]
    pub cross_attention_output: Tensor,
    #[cfg(feature = "dev-bins")]
    pub normalized_hidden: Tensor,
}

impl EncodedShapeLatents {
    pub fn for_decoder(&self) -> Result<Tensor> {
        self.latents.transpose(1, 2)?.contiguous()
    }
}

/// Point-cross-attention encoder carried by the Hunyuan3D 2.1 shape VAE.
#[derive(Debug)]
pub struct ShapeVaeEncoder {
    cfg: ShapeVaeEncoderConfig,
    dtype: DType,
    device: Device,
    fourier: FourierEmbedder,
    input_proj: Linear,
    cross_attn: ResidualCrossAttentionBlock,
    self_attn: Vec<ResidualAttentionBlock>,
    ln_post: LayerNorm,
    pre_kl: Linear,
}

impl ShapeVaeEncoder {
    /// `vb` is scoped to the checkpoint's `vae.` prefix.
    pub fn new(cfg: &ShapeVaeEncoderConfig, vb: VarBuilder) -> Result<Self> {
        if cfg.num_latents == 0
            || cfg.width == 0
            || cfg.heads == 0
            || !cfg.width.is_multiple_of(cfg.heads)
            || cfg.pc_size + cfg.pc_sharpedge_size == 0
            || cfg.downsample_ratio == 0
        {
            return Err(invalid("invalid shape-VAE encoder configuration"));
        }
        let network = cfg.network_config();
        let dtype = vb.dtype();
        let device = vb.device().clone();
        let encoder = vb.pp("encoder");
        let self_blocks = encoder.pp("self_attn").pp("resblocks");
        let mut self_attn = Vec::with_capacity(cfg.num_layers);
        for index in 0..cfg.num_layers {
            self_attn.push(ResidualAttentionBlock::new(
                &network,
                self_blocks.pp(index),
            )?);
        }
        Ok(Self {
            cfg: *cfg,
            dtype,
            device,
            fourier: FourierEmbedder::new(cfg.num_freqs, 3, true, false),
            input_proj: linear(cfg.input_width(), cfg.width, encoder.pp("input_proj"))?,
            cross_attn: ResidualCrossAttentionBlock::new(&network, encoder.pp("cross_attn"))?,
            self_attn,
            ln_post: layer_norm(
                cfg.width,
                LayerNormConfig {
                    eps: LN_POST_EPS,
                    remove_mean: true,
                    affine: true,
                },
                encoder.pp("ln_post"),
            )?,
            pre_kl: linear(cfg.width, cfg.embed_dim * 2, vb.pp("pre_kl"))?,
        })
    }

    pub fn config(&self) -> &ShapeVaeEncoderConfig {
        &self.cfg
    }

    /// Sample and encode a mesh into the deterministic posterior mode.
    ///
    /// The seed freezes both geometry sampling and the FPS start points so a
    /// durable retry produces the same latent sequence. The returned latent
    /// layout remains `[B,N,C]`; use [`EncodedShapeLatents::for_decoder`] for
    /// the shape decoder's `[B,C,N]` input.
    pub fn encode_mesh_mode(
        &self,
        mesh: &Mesh,
        seed: u64,
        query_chunk: usize,
    ) -> Result<EncodedShapeLatents> {
        self.encode_mesh(mesh, seed, query_chunk, false)
    }

    /// Sample the posterior with mold's cross-device deterministic RNG.
    /// Tencent's published minimal round-trip samples rather than taking the
    /// mode; keeping the noise seed explicit makes that behavior restartable.
    pub fn encode_mesh_sampled(
        &self,
        mesh: &Mesh,
        seed: u64,
        query_chunk: usize,
    ) -> Result<EncodedShapeLatents> {
        self.encode_mesh(mesh, seed, query_chunk, true)
    }

    fn encode_mesh(
        &self,
        mesh: &Mesh,
        seed: u64,
        query_chunk: usize,
        sample_posterior: bool,
    ) -> Result<EncodedShapeLatents> {
        if self.cfg.point_feats != 4 {
            return Err(invalid(
                "mesh encoding requires normal xyz plus the sharp-edge label",
            ));
        }
        if self.cfg.pc_size == 0 {
            return Err(invalid(
                "mesh encoding requires at least one regular surface point",
            ));
        }
        let mut samples = sample_mesh_surface(mesh, self.cfg.pc_size, seed)?;
        let sharp_seed = seed ^ 0x4859_3353_4841_5250;
        if self.cfg.pc_sharpedge_size > 0 {
            samples.extend(sample_mesh_sharp_edges(
                mesh,
                self.cfg.pc_sharpedge_size,
                sharp_seed,
            )?);
        }

        let total_points = self.cfg.pc_size + self.cfg.pc_sharpedge_size;
        let regular_latents = self.cfg.pc_size * self.cfg.num_latents / total_points;
        let sharp_latents = self.cfg.num_latents - regular_latents;
        let mut rng = StdRng::seed_from_u64(seed ^ 0x4650_5353_5441_5254);
        let positions: Vec<[f32; 3]> = samples.iter().map(|sample| sample.position).collect();
        let mut selected = farthest_point_indices(
            &positions[..self.cfg.pc_size],
            regular_latents,
            rng.gen_range(0..self.cfg.pc_size),
        )?;
        if sharp_latents > 0 {
            let sharp = farthest_point_indices(
                &positions[self.cfg.pc_size..],
                sharp_latents,
                rng.gen_range(0..self.cfg.pc_sharpedge_size),
            )?;
            selected.extend(sharp.into_iter().map(|index| index + self.cfg.pc_size));
        }

        let positions: Vec<f32> = samples.iter().flat_map(|sample| sample.position).collect();
        let features: Vec<f32> = samples
            .iter()
            .flat_map(|sample| {
                [
                    sample.normal[0],
                    sample.normal[1],
                    sample.normal[2],
                    if sample.sharp_edge { 1.0 } else { 0.0 },
                ]
            })
            .collect();
        let points = Tensor::from_vec(positions, (1, total_points, 3), &self.device)?
            .to_dtype(self.dtype)?;
        let features = Tensor::from_vec(
            features,
            (1, total_points, self.cfg.point_feats),
            &self.device,
        )?
        .to_dtype(self.dtype)?;
        let posterior_noise = sample_posterior
            .then(|| {
                crate::engine::seeded_randn(
                    seed ^ 0x504f_5354_4552_494f,
                    &[1, self.cfg.num_latents, self.cfg.embed_dim],
                    &self.device,
                    self.dtype,
                )
                .map_err(|error| invalid(error.to_string()))
            })
            .transpose()?;
        self.encode_preselected(
            &points,
            &features,
            &selected,
            posterior_noise.as_ref(),
            query_chunk,
        )
    }

    /// Encode caller-frozen points and FPS indices.
    ///
    /// `points` and `features` are `[B, pc_size + pc_sharpedge_size, 3/point_feats]`.
    /// `selected` identifies the latent query points in that same input. Passing
    /// indices explicitly makes the neural comparison independent of PyTorch's
    /// random permutation and torch-cluster FPS implementation. A supplied
    /// `posterior_noise` must be `[B, num_latents, embed_dim]`; absent noise uses
    /// the posterior mode for deterministic mesh round trips.
    pub fn encode_preselected(
        &self,
        points: &Tensor,
        features: &Tensor,
        selected: &[usize],
        posterior_noise: Option<&Tensor>,
        query_chunk: usize,
    ) -> Result<EncodedShapeLatents> {
        let (batch, point_count, point_width) = points.dims3()?;
        let (feature_batch, feature_count, feature_width) = features.dims3()?;
        let expected_points = self.cfg.pc_size + self.cfg.pc_sharpedge_size;
        if point_width != 3
            || feature_batch != batch
            || feature_count != point_count
            || feature_width != self.cfg.point_feats
            || point_count != expected_points
        {
            return Err(invalid(format!(
                "shape-VAE encoder expects points/features [B,{expected_points},3/{}]",
                self.cfg.point_feats
            )));
        }
        if selected.len() != self.cfg.num_latents
            || selected.iter().any(|index| *index >= point_count)
        {
            return Err(invalid(format!(
                "shape-VAE encoder expects {} in-range FPS indices",
                self.cfg.num_latents
            )));
        }
        if query_chunk == 0 {
            return Err(invalid("shape-VAE encoder query chunk must be positive"));
        }

        let selected_u32: Vec<u32> = selected
            .iter()
            .map(|index| {
                u32::try_from(*index)
                    .map_err(|_| invalid("shape-VAE FPS index exceeds tensor index range"))
            })
            .collect::<Result<_>>()?;
        let ids = Tensor::from_vec(selected_u32, selected.len(), points.device())?;
        let query_points = points.index_select(&ids, 1)?;
        let query_features = features.index_select(&ids, 1)?;

        let data = Tensor::cat(&[&self.fourier.forward(points)?, features], D::Minus1)?;
        let data = self.input_proj.forward(&data)?;
        let kv = self.cross_attn.project_kv(&data)?;

        let query = Tensor::cat(
            &[&self.fourier.forward(&query_points)?, &query_features],
            D::Minus1,
        )?;
        let mut chunks = Vec::with_capacity(self.cfg.num_latents.div_ceil(query_chunk));
        for start in (0..self.cfg.num_latents).step_by(query_chunk) {
            let len = query_chunk.min(self.cfg.num_latents - start);
            let projected = self.input_proj.forward(&query.narrow(1, start, len)?)?;
            chunks.push(self.cross_attn.forward_with_kv_upcast(&projected, &kv)?);
        }
        let chunk_refs: Vec<&Tensor> = chunks.iter().collect();
        let mut latents = Tensor::cat(&chunk_refs, 1)?;
        #[cfg(feature = "dev-bins")]
        let cross_attention_output = latents.clone();
        for block in &self.self_attn {
            latents = block.forward_upcast(&latents)?;
        }
        latents = self.ln_post.forward(&latents)?;
        #[cfg(feature = "dev-bins")]
        let normalized_hidden = latents.clone();

        let moments = self.pre_kl.forward(&latents)?;
        let mean = moments
            .narrow(D::Minus1, 0, self.cfg.embed_dim)?
            .contiguous()?;
        let logvar = moments
            .narrow(D::Minus1, self.cfg.embed_dim, self.cfg.embed_dim)?
            .clamp(-30.0f32, 20.0f32)?
            .contiguous()?;
        let latents = match posterior_noise {
            Some(noise) => {
                if noise.dims() != mean.dims() {
                    return Err(invalid(format!(
                        "shape-VAE posterior noise must have shape {:?}",
                        mean.dims()
                    )));
                }
                let std = (&logvar * 0.5)?.exp()?;
                (&mean + noise.broadcast_mul(&std)?)?
            }
            None => mean.clone(),
        };
        Ok(EncodedShapeLatents {
            mean,
            logvar,
            latents,
            #[cfg(feature = "dev-bins")]
            cross_attention_output,
            #[cfg(feature = "dev-bins")]
            normalized_hidden,
        })
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

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn normalized_positions(mesh: &Mesh) -> Result<Vec<[f32; 3]>> {
    mesh.validate()
        .map_err(|error| invalid(error.to_string()))?;
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
    Ok(mesh
        .vertices
        .iter()
        .map(|v| {
            [
                (v[0] - center[0]) * scale,
                (v[1] - center[1]) * scale,
                (v[2] - center[2]) * scale,
            ]
        })
        .collect())
}

/// Uniformly sample a mesh surface after Tencent's centered 0.9999 scaling.
///
/// The seed belongs to mold's durable execution record. Upstream leaves both
/// trimesh and NumPy sampling process-random; making it explicit here is what
/// lets restart/retry reuse the same encoder input instead of changing shape.
pub fn sample_mesh_surface(mesh: &Mesh, count: usize, seed: u64) -> Result<Vec<SurfacePoint>> {
    if count == 0 {
        return Ok(Vec::new());
    }
    let positions = normalized_positions(mesh)?;

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

/// Sample Tencent's sharp-edge point set after angle-weighted vertex normals.
///
/// Upstream first labels a vertex sharp when any incident face normal has a
/// dot product below `0.985` with its trimesh vertex normal. It then retains
/// every directed triangle edge whose two endpoints are sharp, including a
/// triangulation diagonal, and samples those segments by length.
pub fn sample_mesh_sharp_edges(mesh: &Mesh, count: usize, seed: u64) -> Result<Vec<SurfacePoint>> {
    if count == 0 {
        return Ok(Vec::new());
    }
    let positions = normalized_positions(mesh)?;
    let mut face_normals = Vec::with_capacity(mesh.faces.len());
    let mut vertex_normal_sums = vec![[0.0f32; 3]; positions.len()];
    for face in &mesh.faces {
        let indices = [face[0] as usize, face[1] as usize, face[2] as usize];
        let vertices = [
            positions[indices[0]],
            positions[indices[1]],
            positions[indices[2]],
        ];
        let normal = normalize(cross(
            sub(vertices[1], vertices[0]),
            sub(vertices[2], vertices[0]),
        ))
        .ok_or_else(|| invalid("shape-VAE sharp-edge sampling rejects degenerate triangles"))?;
        face_normals.push(normal);
        for corner in 0..3 {
            let origin = vertices[corner];
            let left = normalize(sub(vertices[(corner + 1) % 3], origin)).ok_or_else(|| {
                invalid("shape-VAE sharp-edge sampling rejects zero-length edges")
            })?;
            let right = normalize(sub(vertices[(corner + 2) % 3], origin)).ok_or_else(|| {
                invalid("shape-VAE sharp-edge sampling rejects zero-length edges")
            })?;
            let angle = dot(left, right).clamp(-1.0, 1.0).acos();
            for axis in 0..3 {
                vertex_normal_sums[indices[corner]][axis] += normal[axis] * angle;
            }
        }
    }
    let vertex_normals = vertex_normal_sums
        .into_iter()
        .map(|normal| {
            normalize(normal).ok_or_else(|| {
                invalid("shape-VAE sharp-edge sampling found an invalid vertex normal")
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let mut minimum_dot = vec![1.0f32; positions.len()];
    for (face, normal) in mesh.faces.iter().zip(&face_normals) {
        for index in face {
            let index = *index as usize;
            minimum_dot[index] = minimum_dot[index].min(dot(vertex_normals[index], *normal));
        }
    }
    let sharp: Vec<bool> = minimum_dot.into_iter().map(|value| value < 0.985).collect();

    let mut edges = Vec::new();
    let mut cumulative = Vec::new();
    let mut total = 0.0f64;
    for face in &mesh.faces {
        for (a, b) in [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])] {
            let a = a as usize;
            let b = b as usize;
            if sharp[a] && sharp[b] {
                let weight = f64::from(length(sub(positions[b], positions[a])));
                if weight.is_finite() && weight > 0.0 {
                    total += weight;
                    edges.push((a, b));
                    cumulative.push(total);
                }
            }
        }
    }
    if edges.is_empty() || !total.is_finite() || total <= 0.0 {
        return Err(invalid("shape-VAE mesh has no sharp edges to sample"));
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let mut samples = Vec::with_capacity(count);
    for _ in 0..count {
        let pick = rng.gen::<f64>() * total;
        let edge_index = cumulative
            .partition_point(|weight| *weight < pick)
            .min(edges.len() - 1);
        let (a, b) = edges[edge_index];
        let weight = rng.gen::<f32>();
        let inverse = 1.0 - weight;
        samples.push(SurfacePoint {
            position: [
                weight * positions[a][0] + inverse * positions[b][0],
                weight * positions[a][1] + inverse * positions[b][1],
                weight * positions[a][2] + inverse * positions[b][2],
            ],
            normal: [
                weight * vertex_normals[a][0] + inverse * vertex_normals[b][0],
                weight * vertex_normals[a][1] + inverse * vertex_normals[b][1],
                weight * vertex_normals[a][2] + inverse * vertex_normals[b][2],
            ],
            sharp_edge: true,
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
        distances
            .par_iter_mut()
            .zip(points.par_iter())
            .for_each(|(distance, point)| {
                let delta = sub(*point, centroid);
                let squared = dot(delta, delta);
                *distance = distance.min(squared);
            });
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
fn synthetic_encoder_weights_impl(
    cfg: &ShapeVaeEncoderConfig,
    device: &candle_core::Device,
) -> std::collections::HashMap<String, Tensor> {
    use candle_core::DType;
    use std::collections::HashMap;

    let mut map = HashMap::new();
    macro_rules! linear {
        ($prefix:expr, $out:expr, $input:expr, $bias:expr $(,)?) => {{
            map.insert(
                format!("{}.weight", $prefix),
                Tensor::zeros(($out, $input), DType::F32, device).unwrap(),
            );
            if $bias {
                map.insert(
                    format!("{}.bias", $prefix),
                    Tensor::zeros($out, DType::F32, device).unwrap(),
                );
            }
        }};
    }
    macro_rules! norm {
        ($prefix:expr, $width:expr $(,)?) => {{
            map.insert(
                format!("{}.weight", $prefix),
                Tensor::ones($width, DType::F32, device).unwrap(),
            );
            map.insert(
                format!("{}.bias", $prefix),
                Tensor::zeros($width, DType::F32, device).unwrap(),
            );
        }};
    }

    linear!("encoder.input_proj", cfg.width, cfg.input_width(), true);
    let cross = "encoder.cross_attn";
    norm!(&format!("{cross}.ln_1"), cfg.width);
    norm!(&format!("{cross}.ln_2"), cfg.width);
    norm!(&format!("{cross}.ln_3"), cfg.width);
    linear!(&format!("{cross}.attn.c_q"), cfg.width, cfg.width, false);
    linear!(
        &format!("{cross}.attn.c_kv"),
        cfg.width * 2,
        cfg.width,
        false,
    );
    linear!(&format!("{cross}.attn.c_proj"), cfg.width, cfg.width, true);
    norm!(
        &format!("{cross}.attn.attention.q_norm"),
        cfg.width / cfg.heads,
    );
    norm!(
        &format!("{cross}.attn.attention.k_norm"),
        cfg.width / cfg.heads,
    );
    linear!(&format!("{cross}.mlp.c_fc"), cfg.width * 4, cfg.width, true);
    linear!(
        &format!("{cross}.mlp.c_proj"),
        cfg.width,
        cfg.width * 4,
        true,
    );
    for index in 0..cfg.num_layers {
        let block = format!("encoder.self_attn.resblocks.{index}");
        norm!(&format!("{block}.ln_1"), cfg.width);
        norm!(&format!("{block}.ln_2"), cfg.width);
        linear!(
            &format!("{block}.attn.c_qkv"),
            cfg.width * 3,
            cfg.width,
            false,
        );
        linear!(&format!("{block}.attn.c_proj"), cfg.width, cfg.width, true);
        norm!(
            &format!("{block}.attn.attention.q_norm"),
            cfg.width / cfg.heads,
        );
        norm!(
            &format!("{block}.attn.attention.k_norm"),
            cfg.width / cfg.heads,
        );
        linear!(&format!("{block}.mlp.c_fc"), cfg.width * 4, cfg.width, true);
        linear!(
            &format!("{block}.mlp.c_proj"),
            cfg.width,
            cfg.width * 4,
            true,
        );
    }
    norm!("encoder.ln_post", cfg.width);
    linear!("pre_kl", cfg.embed_dim * 2, cfg.width, true);
    map
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hunyuan3d::mesh::Mesh;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;
    use std::collections::HashMap;

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
    fn sharp_edge_sampling_is_seeded_and_marks_cube_edges() {
        let mesh = Mesh {
            vertices: vec![
                [-1.0, -1.0, -1.0],
                [1.0, -1.0, -1.0],
                [1.0, 1.0, -1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, -1.0, 1.0],
                [1.0, -1.0, 1.0],
                [1.0, 1.0, 1.0],
                [-1.0, 1.0, 1.0],
            ],
            faces: vec![
                [0, 2, 1],
                [0, 3, 2],
                [4, 5, 6],
                [4, 6, 7],
                [0, 1, 5],
                [0, 5, 4],
                [1, 2, 6],
                [1, 6, 5],
                [2, 3, 7],
                [2, 7, 6],
                [3, 0, 4],
                [3, 4, 7],
            ],
            ..Mesh::default()
        };
        let first = sample_mesh_sharp_edges(&mesh, 128, 9001).unwrap();
        let second = sample_mesh_sharp_edges(&mesh, 128, 9001).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.len(), 128);
        assert!(first.iter().all(|sample| sample.sharp_edge));
        assert!(first.iter().all(|sample| {
            let boundary_axes = sample
                .position
                .iter()
                .filter(|value| (value.abs() - NORMALIZED_MESH_SCALE).abs() < 1e-5)
                .count();
            boundary_axes >= 1
        }));
    }

    #[test]
    fn sharp_edge_sampling_refuses_a_coplanar_surface() {
        let mesh = Mesh {
            vertices: vec![
                [-1.0, -1.0, 0.0],
                [1.0, -1.0, 0.0],
                [1.0, 1.0, 0.0],
                [-1.0, 1.0, 0.0],
            ],
            faces: vec![[0, 1, 2], [0, 2, 3]],
            ..Mesh::default()
        };
        let error = sample_mesh_sharp_edges(&mesh, 8, 1).unwrap_err();
        assert!(error.to_string().contains("has no sharp edges"));
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

    #[test]
    fn tiny_encoder_returns_mode_and_seeded_posterior_in_decoder_layout() {
        let device = Device::Cpu;
        let cfg = ShapeVaeEncoderConfig {
            num_latents: 2,
            embed_dim: 2,
            width: 4,
            heads: 2,
            num_layers: 1,
            pc_size: 4,
            pc_sharpedge_size: 0,
            point_feats: 4,
            downsample_ratio: 2,
            num_freqs: 1,
        };
        let weights = synthetic_encoder_weights(&cfg, &device);
        let encoder =
            ShapeVaeEncoder::new(&cfg, VarBuilder::from_tensors(weights, DType::F32, &device))
                .unwrap();
        let points = Tensor::from_vec(
            vec![
                -0.5f32, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, -0.5, 0.0, 0.0, 0.5, 0.0,
            ],
            (1, 4, 3),
            &device,
        )
        .unwrap();
        let features = Tensor::from_vec(
            vec![
                0.0f32, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
            ],
            (1, 4, 4),
            &device,
        )
        .unwrap();
        let selected = [0usize, 1];
        let mode = encoder
            .encode_preselected(&points, &features, &selected, None, 1)
            .unwrap();
        assert_eq!(mode.mean.dims(), &[1, 2, 2]);
        assert_eq!(mode.latents.dims(), &[1, 2, 2]);
        assert_eq!(mode.for_decoder().unwrap().dims(), &[1, 2, 2]);
        assert_eq!(
            mode.mean.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            mode.latents
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        );

        let noise = Tensor::ones((1, 2, 2), DType::F32, &device).unwrap();
        let sampled = encoder
            .encode_preselected(&points, &features, &selected, Some(&noise), 2)
            .unwrap();
        assert_ne!(
            sampled
                .mean
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            sampled
                .latents
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        );

        let mesh_mode = encoder.encode_mesh_mode(&tetrahedron(), 42, 1).unwrap();
        let mesh_mode_second = encoder.encode_mesh_mode(&tetrahedron(), 42, 2).unwrap();
        let mesh_sampled = encoder.encode_mesh_sampled(&tetrahedron(), 42, 1).unwrap();
        assert_eq!(mesh_mode.mean.dims(), &[1, 2, 2]);
        assert_eq!(mesh_mode.for_decoder().unwrap().dims(), &[1, 2, 2]);
        assert_eq!(
            mesh_mode
                .mean
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            mesh_mode_second
                .mean
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            "query chunking must not change a deterministic mesh encoding"
        );
        assert_ne!(
            mesh_mode
                .latents
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            mesh_sampled
                .latents
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
        );
    }

    fn synthetic_encoder_weights(
        cfg: &ShapeVaeEncoderConfig,
        device: &Device,
    ) -> HashMap<String, Tensor> {
        synthetic_encoder_weights_impl(cfg, device)
    }
}
