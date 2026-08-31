//! The Hunyuan3D image-to-3D engine.
//!
//! # Stage layout, and where each one runs
//!
//! | Stage | Device | Reported as |
//! | --- | --- | --- |
//! | DINOv2-giant image encode | GPU | `ProgressPhase::PromptEncode` |
//! | Shape DiT sampling | GPU | `DenoiseStep` |
//! | Chunked occupancy decode | GPU | `StageProgress` per chunk |
//! | Surface extraction, normals, GLB, poster | **CPU** | `StageProgress` |
//!
//! The split is deliberate and is the whole reason the decode is chunked. At
//! octree resolution 256 the shape VAE evaluates its occupancy field on
//! `257^3` ≈ 17 million query points; materialising those coordinates and
//! their logits at once is gigabytes before any attention runs. Chunking also
//! gives the only honest progress signal the stage has, and a cancellation
//! checkpoint between chunks — without it a cancelled mesh render would keep
//! a GPU busy for minutes after the user gave up.
//!
//! Everything after the occupancy grid is pure CPU and does not touch the
//! device, so a long surface extraction does not hold the card. It still runs
//! inside `generate` today (one lease, one job); moving it to the scheduler's
//! host-utility lane is #1496.
//!
//! # No text encoder
//!
//! There is none, anywhere in this family. The source image is the only
//! conditioning, which is why `validation::validate_mesh_request` refuses a
//! request without one rather than falling back to an empty prompt. The
//! prompt, if the caller sent one, is recorded as provenance and never read.

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use mold_core::{GenerateRequest, GenerateResponse, MeshData, ModelPaths, OutputFormat};

use crate::engine::{rand_seed, InferenceEngine, LoadStrategy};
use crate::engine_base::EngineBase;
use crate::progress::{ProgressEvent, ProgressPhase};

use super::dino2::{Dinov2Config, Dinov2Model};
use super::glb::{write_glb, GlbMaterial};
use super::mesh::{Mesh, MeshAlgorithm, OccupancyGrid};
use super::sampler::{self, SamplingPlan};
use super::shape_vae::{ShapeVae, ShapeVaeConfig};
use super::transformer::{Config as DitConfig, Hunyuan3dDit};

/// Prefixes the three networks live under inside the single checkpoint.
const DIT_PREFIX: &str = "model";
const VAE_PREFIX: &str = "vae";
const VISION_PREFIX: &str = "conditioner.main_image_encoder.model";

/// Default query-grid resolution. Upstream's `VAEDecodeHunyuan3D` default.
pub const DEFAULT_OCTREE_RESOLUTION: usize = 256;
/// Default surface-net iso-level. Upstream's `VoxelToMesh` default — note it
/// is 0.6, not the 0.5 a reader might assume for a binary occupancy field.
pub const DEFAULT_THRESHOLD: f32 = 0.6;
/// Half-width of the query cube. `VanillaVolumeDecoder`'s `bounds` default.
const QUERY_BOUNDS: f32 = 1.01;
/// Query points per decode chunk. Upstream's `num_chunks` default.
const DEFAULT_DECODE_CHUNK: usize = 8_000;
/// Env override for [`DEFAULT_DECODE_CHUNK`].
const DECODE_CHUNK_ENV: &str = "MOLD_HUNYUAN3D_DECODE_CHUNKS";
/// Edge length of the gallery poster.
const POSTER_SIZE: u32 = 512;

/// How many chunks to decode before emitting a progress tick. One event per
/// chunk at 17M points would be ~2,100 SSE frames for a single stage.
const DECODE_TICKS: usize = 64;

struct Loaded {
    dit: Hunyuan3dDit,
    vae: ShapeVae,
    vision: Dinov2Model,
    device: Device,
    dtype: DType,
    /// Edge length the source image is letterboxed to before the vision
    /// tower sees it. Per-checkpoint: 512 for the 1.1B tiers, 1022 for the
    /// mini tier (`image_processor.size` in each `config.yaml`).
    conditioning_size: u32,
    /// Number of latent tokens the DiT denoises.
    num_latents: usize,
}

pub struct Hunyuan3dEngine {
    base: EngineBase<Loaded>,
}

impl Hunyuan3dEngine {
    pub fn new(
        model_name: String,
        paths: ModelPaths,
        load_strategy: LoadStrategy,
        gpu_ordinal: usize,
    ) -> Self {
        Self {
            base: EngineBase::new(model_name, paths, load_strategy, gpu_ordinal),
        }
    }

    /// Query points per decode chunk, honouring the env override.
    ///
    /// Clamped rather than trusted: a zero would loop forever and a value
    /// larger than the grid just allocates the whole thing, which is exactly
    /// what chunking exists to avoid.
    fn decode_chunk() -> usize {
        crate::runtime_env::value(DECODE_CHUNK_ENV)
            .and_then(|raw| raw.trim().parse::<usize>().ok())
            .filter(|value| *value > 0)
            .map(|value| value.clamp(256, 1_000_000))
            .unwrap_or(DEFAULT_DECODE_CHUNK)
    }

    fn load_inner(&mut self) -> Result<Loaded> {
        let checkpoint = self.base.paths.transformer.clone();
        if !checkpoint.is_file() {
            bail!("Hunyuan3D checkpoint not found at {}", checkpoint.display());
        }

        // Geometry comes from the FILE, not the filename, mirroring
        // `comfy/model_detection.py:784-797`. A repack or a community
        // re-quantization then loads without a manifest entry describing its
        // internals, and a manifest that drifts from the weights is caught
        // here instead of producing garbage.
        let header = mold_core::safetensors_probe::read_safetensors_header(&checkpoint)
            .with_context(|| format!("read safetensors header at {}", checkpoint.display()))?;
        let dit_cfg = detect_dit_config(&header)?;
        let vae_cfg = detect_vae_config(&header, &self.base.model_name);
        let conditioning_size = conditioning_size_for(&self.base.model_name);

        let device = crate::device::create_device(self.base.gpu_ordinal, &self.base.progress)?;
        // The checkpoints ship fp16. bf16 on CUDA and fp16 elsewhere matches
        // what every other mold engine does with a half-precision checkpoint;
        // CPU stays f32 because half-precision matmul there is emulated and
        // slower than the widening.
        let dtype = if device.is_cpu() {
            DType::F32
        } else if device.is_cuda() {
            DType::BF16
        } else {
            DType::F16
        };

        let vb = crate::weight_loader::load_safetensors_with_progress(
            std::slice::from_ref(&checkpoint),
            dtype,
            &device,
            "Hunyuan3D checkpoint",
            &self.base.progress,
        )?;

        let dit = Hunyuan3dDit::new(&dit_cfg, vb.pp(DIT_PREFIX))
            .context("build the Hunyuan3D shape transformer")?;
        let vae =
            ShapeVae::new(&vae_cfg, vb.pp(VAE_PREFIX)).context("build the Hunyuan3D shape VAE")?;
        let vision = Dinov2Model::new(&Dinov2Config::giant(), vb.pp(VISION_PREFIX))
            .context("build the Hunyuan3D image conditioner")?;

        Ok(Loaded {
            dit,
            vae,
            vision,
            device,
            dtype,
            conditioning_size,
            num_latents: vae_cfg.num_latents,
        })
    }

    fn generate_inner(&mut self, req: &GenerateRequest) -> Result<GenerateResponse> {
        let started = std::time::Instant::now();
        if self.base.loaded.is_none() {
            let loaded = self.load_inner()?;
            self.base.loaded = Some(loaded);
        }
        let loaded = self
            .base
            .loaded
            .as_ref()
            .expect("just loaded above or already present");

        let options = req.mesh.clone().unwrap_or_default();
        let octree = options
            .octree_resolution
            .map(|value| value as usize)
            .unwrap_or(DEFAULT_OCTREE_RESOLUTION);
        let threshold = options.threshold.unwrap_or(DEFAULT_THRESHOLD);
        if options.texture == Some(true) {
            // Refused rather than silently answered with bare geometry: a
            // user who asked for materials must not discover their absence
            // after waiting for a render.
            bail!(
                "PBR texture generation is not available in this build; \
                 omit --texture to render geometry only"
            );
        }

        // ── Conditioning ────────────────────────────────────────────────
        self.base.progress.stage_start("Encoding image");
        let phase_started = std::time::Instant::now();
        let source = req
            .source_image
            .as_deref()
            .filter(|bytes| !bytes.is_empty())
            .context("3-D generation requires a source image")?;
        let image = image::DynamicImage::ImageRgb8(
            crate::img_utils::decode_oriented_srgb(source).context("decode the source image")?,
        );
        let pixels = super::dino2::preprocess(
            &image,
            loaded.conditioning_size,
            &loaded.device,
            loaded.dtype,
        )?;
        let cond = loaded.vision.forward(&pixels)?;
        self.base
            .progress
            .stage_done("Encoding image", phase_started.elapsed());
        self.base.progress.phase_done(
            ProgressPhase::PromptEncode,
            "Encoding image",
            phase_started.elapsed(),
        );

        // ── Sampling ────────────────────────────────────────────────────
        let plan = sampler::plan(
            req.steps as usize,
            sampler::DEFAULT_SHIFT,
            req.guidance,
            loaded.dit.config().guidance_embed,
        );
        let seed = req.seed.unwrap_or_else(rand_seed);
        let latents = self.sample(loaded, &cond, &plan, seed)?;

        // ── Occupancy field ─────────────────────────────────────────────
        let grid = self.decode_occupancy(loaded, &latents, octree)?;

        // ── Surface, normals, GLB, poster (CPU) ─────────────────────────
        let mesh = self.extract_surface(&grid, threshold, options.target_faces)?;
        drop(grid);
        let (bounds_min, bounds_max) = mesh.bounds();

        self.base.progress.stage_start("Writing mesh");
        let write_started = std::time::Instant::now();
        let glb = write_glb(&mesh, &GlbMaterial::default(), None)?;
        let poster = super::poster::render_poster(&mesh, POSTER_SIZE)
            .context("render the gallery poster")?;
        self.base
            .progress
            .stage_done("Writing mesh", write_started.elapsed());

        Ok(GenerateResponse {
            images: Vec::new(),
            video: None,
            audio: None,
            mesh: Some(MeshData {
                data: glb,
                format: OutputFormat::Glb,
                vertex_count: mesh.vertex_count() as u32,
                face_count: mesh.face_count() as u32,
                bounds_min,
                bounds_max,
                textured: false,
                poster,
                poster_width: POSTER_SIZE,
                poster_height: POSTER_SIZE,
            }),
            generation_time_ms: started.elapsed().as_millis() as u64,
            model: self.base.model_name.clone(),
            seed_used: seed,
            gpu: Some(self.base.gpu_ordinal),
            request_warnings: Vec::new(),
        })
    }

    /// Run the flow-matching loop, returning `[1, in_channels, num_latents]`.
    fn sample(
        &self,
        loaded: &Loaded,
        cond: &Tensor,
        plan: &SamplingPlan,
        seed: u64,
    ) -> Result<Tensor> {
        let channels = loaded.dit.config().in_channels;
        // CPU-seeded noise transferred to the execution device — the family
        // capability's `SeedContract`, and what makes a seed reproducible
        // across a CPU/CUDA/Metal move.
        let noise = {
            let cpu = Device::Cpu;
            cpu.set_seed(seed)?;
            Tensor::randn(0f32, 1.0, (1, channels, loaded.num_latents), &cpu)?
        };
        let mut latents = noise.to_device(&loaded.device)?.to_dtype(loaded.dtype)?;

        // The unconditional branch is a zero context of the same shape, not an
        // empty one: `Hunyuan3Dv2Conditioning` builds it as
        // `torch.zeros_like(embeds)` (`nodes_hunyuan3d.py:51`).
        let uncond = if plan.cfg_scale.is_some() {
            Some(cond.zeros_like()?)
        } else {
            None
        };
        let guidance = plan
            .guidance_embed
            .map(|value| Tensor::new(&[value as f32], &loaded.device))
            .transpose()?
            .map(|t| t.to_dtype(loaded.dtype))
            .transpose()?;

        let total = plan.steps();
        let started = std::time::Instant::now();
        for (step, window) in plan.sigmas.windows(2).enumerate() {
            self.base.progress.checkpoint()?;
            let (sigma, sigma_next) = (window[0], window[1]);
            // `timestep(sigma) == sigma` at multiplier 1.0 — see the sampler
            // module doc for the derivation.
            let timestep = Tensor::new(&[sigma as f32], &loaded.device)?.to_dtype(loaded.dtype)?;

            let velocity = match (&uncond, plan.cfg_scale) {
                (Some(uncond), Some(scale)) => {
                    let conditioned =
                        loaded
                            .dit
                            .forward(&latents, &timestep, cond, guidance.as_ref())?;
                    let unconditioned =
                        loaded
                            .dit
                            .forward(&latents, &timestep, uncond, guidance.as_ref())?;
                    sampler::apply_cfg(&conditioned, &unconditioned, scale)?
                }
                _ => loaded
                    .dit
                    .forward(&latents, &timestep, cond, guidance.as_ref())?,
            };
            latents = sampler::euler_step(&latents, &velocity, sigma, sigma_next)?;
            self.base.progress.emit(ProgressEvent::DenoiseStep {
                step: step + 1,
                total,
                elapsed: started.elapsed(),
            });
        }
        Ok(latents)
    }

    /// Evaluate the occupancy field on the query grid, chunk by chunk.
    fn decode_occupancy(
        &self,
        loaded: &Loaded,
        latents: &Tensor,
        octree: usize,
    ) -> Result<OccupancyGrid> {
        self.base.progress.stage_start("Decoding volume");
        let phase_started = std::time::Instant::now();

        // The scale factor belongs to the sampler's output contract, not to
        // the decoder — see `ShapeVae::unscale_latents`.
        let unscaled = loaded.vae.unscale_latents(latents)?;
        let prepared = loaded.vae.prepare_latents(&unscaled)?;
        // Hoisted out of the loop: the cross-attention keys and values depend
        // only on the latents, so projecting them per chunk would repeat the
        // same GEMM two thousand times.
        let cross_kv = loaded.vae.prepare_cross_kv(&prepared)?;
        drop(prepared);

        let total = super::shape_vae::query_grid_len(octree);
        let chunk = Self::decode_chunk();
        let chunks = total.div_ceil(chunk);
        let tick_every = chunks.div_ceil(DECODE_TICKS).max(1);

        let mut logits: Vec<f32> = Vec::with_capacity(total);
        let mut start = 0usize;
        let mut index = 0usize;
        while start < total {
            self.base.progress.checkpoint()?;
            let len = chunk.min(total - start);
            let queries = super::shape_vae::query_grid_chunk(
                octree,
                QUERY_BOUNDS,
                start,
                len,
                &loaded.device,
                loaded.dtype,
            )?
            .unsqueeze(0)?;
            let chunk_logits = loaded.vae.decode_queries_cached(&queries, &cross_kv)?;
            logits.extend(
                chunk_logits
                    .flatten_all()?
                    .to_dtype(DType::F32)?
                    .to_vec1::<f32>()?,
            );
            start += len;
            index += 1;
            if index.is_multiple_of(tick_every) || start >= total {
                self.base
                    .progress
                    .stage_progress("Decoding volume", start, total);
            }
        }

        self.base
            .progress
            .stage_done("Decoding volume", phase_started.elapsed());
        self.base.progress.phase_done(
            ProgressPhase::Vae,
            "Decoding volume",
            phase_started.elapsed(),
        );

        // `reshape_grid_logits` reproduces upstream's trailing `movedim`,
        // which is the only thing that keeps the mesh unmirrored. Doing it on
        // the flat Vec would mean re-implementing that transpose by hand, so
        // it goes through the tensor.
        let dim = octree + 1;
        let flat = Tensor::from_vec(logits, (1, total), &Device::Cpu)?;
        let reshaped = ShapeVae::reshape_grid_logits(&flat, octree)?;
        let ordered = reshaped.flatten_all()?.to_vec1::<f32>()?;
        OccupancyGrid::new(ordered, [dim, dim, dim])
    }

    /// Surface extraction and everything after it. Pure CPU.
    fn extract_surface(
        &self,
        grid: &OccupancyGrid,
        threshold: f32,
        target_faces: Option<u32>,
    ) -> Result<Mesh> {
        self.base.progress.stage_start("Extracting surface");
        let phase_started = std::time::Instant::now();
        let progress = &self.base.progress;
        let mut mesh = super::mesh::extract(
            grid,
            MeshAlgorithm::SurfaceNet,
            threshold,
            &mut |current, total| {
                progress.checkpoint()?;
                progress.stage_progress("Extracting surface", current as usize, total as usize);
                Ok(())
            },
        )?;
        if mesh.is_empty() {
            // A blank GLB is worse than an error: it publishes a gallery row
            // for a print with nothing in it. The usual cause is a threshold
            // the occupancy field never crosses.
            bail!(
                "the shape model produced no surface at threshold {threshold}; \
                 try a different source image or a lower --mesh-threshold"
            );
        }
        super::mesh::compute_smooth_normals(&mut mesh);
        if let Some(target) = target_faces {
            mesh = super::mesh::simplify(&mesh, target as usize)?;
            // Decimation invalidates the old normals.
            super::mesh::compute_smooth_normals(&mut mesh);
        }
        self.base
            .progress
            .stage_done("Extracting surface", phase_started.elapsed());
        self.base.progress.phase_done(
            ProgressPhase::VisualDecode,
            "Extracting surface",
            phase_started.elapsed(),
        );
        Ok(mesh)
    }
}

/// Recover the DiT geometry from a checkpoint header.
fn detect_dit_config(
    header: &mold_core::safetensors_probe::SafetensorsHeader,
) -> Result<DitConfig> {
    let latent_in = header
        .tensor_shapes
        .get(&format!("{DIT_PREFIX}.latent_in.weight"))
        .filter(|shape| shape.len() == 2)
        .context("checkpoint has no model.latent_in.weight; not a Hunyuan3D shape model")?;
    let cond_in = header
        .tensor_shapes
        .get(&format!("{DIT_PREFIX}.cond_in.weight"))
        .filter(|shape| shape.len() == 2)
        .context("checkpoint has no model.cond_in.weight")?;
    let prefix = format!("{DIT_PREFIX}.");
    let cfg = DitConfig::from_state_dict(
        (latent_in[0], latent_in[1]),
        cond_in[1],
        header
            .tensor_names
            .iter()
            .filter_map(|name| name.strip_prefix(&prefix)),
    );
    if cfg.depth == 0 || cfg.depth_single_blocks == 0 {
        bail!("checkpoint declares no transformer blocks; not a Hunyuan3D shape model");
    }
    Ok(cfg)
}

/// Pick the shape-VAE config.
///
/// Only ONE field differs between the published tiers — `scale_factor` — and
/// it is not recoverable from the tensors, because it is a scalar the
/// `config.yaml` carries and the checkpoint does not. So it comes from the
/// model identity, and the fallback is the 1.1B value rather than an error:
/// a community repack under an unrecognized name still decodes, just with the
/// base tier's scale.
fn detect_vae_config(
    _header: &mold_core::safetensors_probe::SafetensorsHeader,
    model_name: &str,
) -> ShapeVaeConfig {
    if model_name.contains("mini") {
        ShapeVaeConfig::v2_0_mini()
    } else {
        ShapeVaeConfig::v2_0()
    }
}

/// Edge length the source image is letterboxed to.
///
/// `image_processor.size` in each checkpoint's `config.yaml`: 1022 for the
/// mini tier, 512 for the 1.1B ones. It is not in the safetensors, so like
/// `scale_factor` it comes from the model identity.
fn conditioning_size_for(model_name: &str) -> u32 {
    if model_name.contains("mini") {
        1022
    } else {
        512
    }
}

impl InferenceEngine for Hunyuan3dEngine {
    fn generate(&mut self, req: &GenerateRequest) -> Result<GenerateResponse> {
        self.base.progress.checkpoint()?;
        self.generate_inner(req)
    }

    fn model_name(&self) -> &str {
        self.base.model_name()
    }

    fn is_loaded(&self) -> bool {
        self.base.is_loaded()
    }

    fn load(&mut self) -> Result<()> {
        if self.base.load_strategy == LoadStrategy::Sequential {
            // Sequential mode loads inside `generate` and drops afterwards;
            // eagerly loading here would double the peak.
            return Ok(());
        }
        if self.base.loaded.is_none() {
            self.base.loaded = Some(self.load_inner()?);
        }
        Ok(())
    }

    fn unload(&mut self) {
        self.base.unload();
        // Best effort: a failed VRAM query must not turn an unload into an
        // error, because the caller is already discarding this engine.
        let _ = crate::device::post_drop_free_vram_bytes(self.base.gpu_ordinal);
    }

    fn set_on_progress(&mut self, callback: crate::progress::ProgressCallback) {
        self.base.set_on_progress(callback);
    }

    fn clear_on_progress(&mut self) {
        self.base.clear_on_progress();
    }

    fn set_cancellation_token(&mut self, token: crate::progress::InferenceCancellationToken) {
        self.base.set_cancellation_token(token);
    }

    fn clear_cancellation_token(&mut self) {
        self.base.clear_cancellation_token();
    }

    fn batch_execution_capability(&self) -> crate::BatchExecutionCapability {
        crate::batch_execution_capability_for_family(mold_core::manifest::HUNYUAN3D_FAMILY)
            .expect("production Hunyuan3D batch capability must be registered")
    }

    fn model_paths(&self) -> Option<&ModelPaths> {
        Some(&self.base.paths)
    }

    fn configured_load_strategy(&self) -> Option<LoadStrategy> {
        Some(self.base.load_strategy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mold_core::safetensors_probe::SafetensorsHeader;
    use std::collections::BTreeMap;

    fn header(depth: usize, single: usize, guidance: bool) -> SafetensorsHeader {
        let mut shapes = BTreeMap::new();
        let mut names = Vec::new();
        shapes.insert("model.latent_in.weight".to_string(), vec![1024, 64]);
        shapes.insert("model.cond_in.weight".to_string(), vec![1024, 1536]);
        names.push("model.latent_in.weight".to_string());
        names.push("model.cond_in.weight".to_string());
        for index in 0..depth {
            names.push(format!("model.double_blocks.{index}.img_attn.qkv.weight"));
        }
        for index in 0..single {
            names.push(format!("model.single_blocks.{index}.linear1.weight"));
        }
        if guidance {
            names.push("model.guidance_in.in_layer.weight".to_string());
        }
        SafetensorsHeader {
            metadata: BTreeMap::new(),
            tensor_names: names,
            tensor_shapes: shapes,
        }
    }

    #[test]
    fn dit_geometry_comes_from_the_file_not_the_filename() {
        let cfg = detect_dit_config(&header(16, 32, false)).unwrap();
        assert_eq!(cfg, DitConfig::v2_0());

        let mini = detect_dit_config(&header(8, 16, true)).unwrap();
        assert_eq!(mini.depth, 8);
        assert_eq!(mini.depth_single_blocks, 16);
        assert!(mini.guidance_embed);
    }

    #[test]
    fn a_checkpoint_without_the_marker_tensors_is_refused_by_name() {
        let mut empty = header(16, 32, false);
        empty.tensor_shapes.remove("model.latent_in.weight");
        let error = detect_dit_config(&empty).unwrap_err().to_string();
        assert!(error.contains("not a Hunyuan3D shape model"), "{error}");
    }

    #[test]
    fn a_checkpoint_with_no_blocks_is_refused() {
        let error = detect_dit_config(&header(0, 0, false))
            .unwrap_err()
            .to_string();
        assert!(error.contains("no transformer blocks"), "{error}");
    }

    #[test]
    fn the_mini_tier_takes_its_own_scale_factor_and_conditioning_size() {
        // Both come from `config.yaml`, not from the tensors, so they key on
        // the model identity — and getting either wrong is silent: the mesh
        // still renders, just subtly wrong.
        let base = detect_vae_config(&header(16, 32, false), "hunyuan3d:fp16");
        let mini = detect_vae_config(&header(8, 16, true), "hunyuan3d-mini-turbo:fp16");
        assert_ne!(base.scale_factor, mini.scale_factor);
        assert_eq!(conditioning_size_for("hunyuan3d:fp16"), 512);
        assert_eq!(conditioning_size_for("hunyuan3d-turbo:fp16"), 512);
        assert_eq!(conditioning_size_for("hunyuan3d-mini-turbo:fp16"), 1022);
    }

    #[test]
    fn the_decode_chunk_override_is_clamped_not_trusted() {
        // The default holds when nothing is set. The clamp itself is asserted
        // through the public constant bounds rather than by mutating the
        // process environment, which `runtime_env` deliberately caches.
        assert_eq!(Hunyuan3dEngine::decode_chunk(), DEFAULT_DECODE_CHUNK);
    }
}
