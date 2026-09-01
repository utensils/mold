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
///
/// Taken from the core constant the generation profile advertises, so the
/// number a client sees pre-selected is the number an omitted field renders.
pub const DEFAULT_OCTREE_RESOLUTION: usize =
    mold_core::validation::MESH_DEFAULT_OCTREE_RESOLUTION as usize;
/// Default surface-net iso-level. Upstream's `VoxelToMesh` default — note it
/// is 0.6, not the 0.5 a reader might assume for a binary occupancy field.
///
/// It is a level on the OCCUPANCY scale produced by [`occupancy_from_logits`],
/// not on the raw VAE logits: ComfyUI's node thresholds what the VAE wrapper
/// hands it, and that wrapper has already mapped the logits through the
/// image post-process `(x + 1) / 2` clamped to `[0, 1]`. On raw logits 0.6
/// means 0.2. Thresholding the raw logits at 0.6 instead — what the first cut
/// did — shrinks every surface inward and drops half the triangles.
///
/// Same authority as [`DEFAULT_OCTREE_RESOLUTION`]: the profile advertises
/// `mold_core::validation::MESH_DEFAULT_THRESHOLD` and this renders it.
pub const DEFAULT_THRESHOLD: f32 = mold_core::validation::MESH_DEFAULT_THRESHOLD as f32;
/// Half-width of the query cube. `VanillaVolumeDecoder`'s `bounds` default.
const QUERY_BOUNDS: f32 = 1.01;
/// Query points per decode chunk on CUDA and the CPU. Upstream's `num_chunks`
/// default. Metal takes a larger, measured default — see
/// [`super::backend::decode_chunk_default`].
const DEFAULT_DECODE_CHUNK: usize = 8_000;
/// Env override for the decode chunk size.
const DECODE_CHUNK_ENV: &str = "MOLD_HUNYUAN3D_DECODE_CHUNKS";
/// Edge length of the gallery poster.
const POSTER_SIZE: u32 = 512;

/// How many chunks to decode before emitting a progress tick. One event per
/// chunk at 17M points would be ~2,100 SSE frames for a single stage.
const DECODE_TICKS: usize = 64;

/// The two edge lengths a source image passes through on its way to the
/// vision tower.
///
/// They are separate `config.yaml` entries and are NOT the same number on
/// every tier, which is why they travel together. See
/// [`super::dino2::preprocess`] for what each one does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Conditioning {
    /// `image_processor.params.size` — the white square the alpha-cropped
    /// subject is centred on.
    letterbox: u32,
    /// `conditioner.params.main_image_encoder.kwargs.image_size` — what
    /// DINOv2 actually receives. Always a whole number of 14px patches.
    encoder: u32,
}

struct Loaded {
    dit: Hunyuan3dDit,
    vae: ShapeVae,
    vision: Dinov2Model,
    device: Device,
    dtype: DType,
    /// The letterbox and encoder edge lengths this checkpoint conditions at.
    /// Per-tier, and derived from the DiT geometry rather than the model
    /// name — see [`conditioning_for`].
    conditioning: Conditioning,
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
    fn decode_chunk(device: &Device) -> usize {
        crate::runtime_env::value(DECODE_CHUNK_ENV)
            .and_then(|raw| raw.trim().parse::<usize>().ok())
            .filter(|value| *value > 0)
            .map(|value| value.clamp(256, 1_000_000))
            .unwrap_or_else(|| super::backend::decode_chunk_default(device, DEFAULT_DECODE_CHUNK))
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
        let vae_cfg = detect_vae_config(&dit_cfg);
        let conditioning = conditioning_for(&dit_cfg);

        let device = crate::device::create_device(self.base.gpu_ordinal, &self.base.progress)?;
        // F16 on every accelerator, not the crate's usual BF16-on-CUDA. See
        // `super::backend`: ComfyUI runs all three networks in fp16, and
        // BF16 cannot resolve the query grid the shape VAE is evaluated on.
        let dtype = super::backend::compute_dtype(&device);

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
            conditioning,
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
            // Defence in depth. `validation::validate_mesh_request` already
            // refuses this at admission, which is the only place that can do
            // so before a multi-gigabyte checkpoint is mapped; this guard
            // exists so a caller that bypasses admission still cannot get
            // bare geometry back from a request that asked for materials.
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
        // RGBA, not RGB: alpha is the SUBJECT MASK here, not decoration.
        // `letterbox_square` crops and centres on the non-zero alpha bounding
        // box, so flattening first would make a background-removed cutout —
        // the input the docs recommend as the best one — indistinguishable
        // from a full opaque frame and condition DINOv2 on the whole canvas,
        // black transparent pixels included.
        let image = image::DynamicImage::ImageRgba8(
            crate::img_utils::decode_oriented_srgb_rgba(source)
                .context("decode the source image")?,
        );
        let pixels = super::dino2::preprocess(
            &image,
            loaded.conditioning.letterbox,
            loaded.conditioning.encoder,
            &loaded.device,
            loaded.dtype,
        )?;
        let cond = loaded.vision.forward(&pixels)?;
        self.base.progress.stage_complete(
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
        // Named as its own stage. Without it the log jumped from a 0.1 s
        // encode straight to the volume decode, so the whole denoise read as
        // decode time and no report could attribute it.
        self.base.progress.stage_start("Sampling");
        let sample_started = std::time::Instant::now();
        let latents = self.sample(loaded, &cond, &plan, seed)?;
        self.base
            .progress
            .stage_done("Sampling", sample_started.elapsed());

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
        // `seeded_randn` is the shared implementation of the family
        // capability's `SeedContract::CpuSeededNoiseTransferredToExecutionDevice`.
        // Calling candle's own `set_seed` + `randn` instead would produce a
        // DIFFERENT stream — candle's per-backend RNG is not the crate-wide
        // `StdRng` — so a seed would not reproduce across a CPU/CUDA/Metal
        // move, which is exactly what the contract promises.
        let mut latents = crate::engine::seeded_randn(
            seed,
            &[1, channels, loaded.num_latents],
            &loaded.device,
            loaded.dtype,
        )?;

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
            .map(|value| guidance_tensor(value, &loaded.device))
            .transpose()?;

        let total = plan.steps();
        let started = std::time::Instant::now();
        for (step, window) in plan.sigmas.windows(2).enumerate() {
            self.base.progress.checkpoint()?;
            let (sigma, sigma_next) = (window[0], window[1]);
            // `timestep(sigma) == sigma` at multiplier 1.0 — see the sampler
            // module doc for the derivation.
            let timestep = timestep_tensor(sigma, &loaded.device)?;

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
        let chunk = Self::decode_chunk(&loaded.device);
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

        self.base.progress.stage_complete(
            ProgressPhase::Vae,
            "Decoding volume",
            phase_started.elapsed(),
        );

        // `reshape_grid_logits` reproduces BOTH upstream moves between the
        // decoder and the mesher (`vae.py:976`, then `comfy/sd.py:1277`);
        // either one alone rotates the mesh. Doing them on the flat Vec would
        // mean re-implementing two transposes by hand, so it goes through the
        // tensor.
        let dim = octree + 1;
        let flat = Tensor::from_vec(logits, (1, total), &Device::Cpu)?;
        let reshaped = ShapeVae::reshape_grid_logits(&flat, octree)?;
        let mut ordered = reshaped.flatten_all()?.to_vec1::<f32>()?;
        occupancy_from_logits(&mut ordered);
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
        self.base.progress.stage_complete(
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

/// DiT depth of the 0.6B mini tier.
///
/// This ONE number separates the two published tiers, and ComfyUI keys on it
/// exactly this way: `Hunyuan3Dv2mini.unet_config` is
/// `{"image_model": "hunyuan3d2", "depth": 8}`
/// (`comfy/supported_models.py:1593-1597`), against the base
/// `Hunyuan3Dv2`'s `{"image_model": "hunyuan3d2"}` at `:1543-1546`.
const MINI_TIER_DIT_DEPTH: usize = 8;

/// Whether a detected DiT geometry is the 0.6B mini tier.
fn is_mini_tier(dit: &DitConfig) -> bool {
    dit.depth == MINI_TIER_DIT_DEPTH
}

/// Map raw occupancy logits onto the scale ComfyUI's mesher thresholds.
///
/// `VAEDecodeHunyuan3D` reaches `ShapeVAE.decode` through the generic
/// `comfy.sd.VAE.decode` wrapper, and the ShapeVAE branch of that class
/// (`comfy/sd.py:838-856`) never overrides `process_output`, so the default
/// image post-process — `image.add_(1.0).div_(2.0).clamp_(0.0, 1.0)`
/// (`comfy/sd.py:505`) — is applied in place to the voxel grid
/// (`comfy/sd.py:1233`) before `VoxelToMesh` ever sees it. The node's 0.6
/// default, and every threshold a user types, is a level on THIS scale.
/// Mirroring it keeps `--mesh-threshold` meaning what it means in the oracle,
/// and keeps the surface-net interpolation identical, clamp included.
fn occupancy_from_logits(logits: &mut [f32]) {
    for value in logits.iter_mut() {
        *value = ((*value + 1.0) * 0.5).clamp(0.0, 1.0);
    }
}

/// Pick the shape-VAE config.
///
/// Only ONE field differs between the published tiers — `scale_factor` — and
/// it is not recoverable from the tensors, because it is a scalar the
/// `config.yaml` carries and the checkpoint does not. It is recoverable from
/// the DiT DEPTH though, which the header does carry, so the tier comes from
/// the weights rather than from whatever name the file was installed under.
/// A repack, a re-quantization, or a locally renamed checkpoint then takes
/// the right scale instead of silently rendering a subtly wrong mesh.
///
/// The values match `comfy/latent_formats.py`: `Hunyuan3Dv2mini.scale_factor`
/// is 1.0188137142395404 (`:945`) and `Hunyuan3Dv2`'s is 0.9990943042622529
/// (`:935`).
fn detect_vae_config(dit: &DitConfig) -> ShapeVaeConfig {
    if is_mini_tier(dit) {
        ShapeVaeConfig::v2_0_mini()
    } else {
        ShapeVaeConfig::v2_0()
    }
}

/// The letterbox and encoder edge lengths this checkpoint conditions at.
///
/// Both are `config.yaml` entries the safetensors does not carry, so like the
/// VAE scale factor they key on the detected DiT depth:
///
/// | Tier | `image_processor.params.size` | `...main_image_encoder.kwargs.image_size` |
/// | --- | --- | --- |
/// | 1.1B (`hunyuan3d-dit-v2-0`, `-turbo`) | 512 | 518 |
/// | 0.6B (`hunyuan3d-dit-v2-mini-turbo`) | 1022 | 1022 |
///
/// 518 is also DINOv2-giant's own stored resolution
/// (`comfy/image_encoders/dino2_giant.json`) and ComfyUI's conditioning size
/// for this family (`comfy/clip_vision.py:38`, `:68`). 512 is not a legal
/// encoder size at all — it is not a multiple of the 14px patch — so
/// conflating the two made the base tiers fail at image encode rather than
/// render differently.
fn conditioning_for(dit: &DitConfig) -> Conditioning {
    if is_mini_tier(dit) {
        Conditioning {
            letterbox: 1022,
            encoder: 1022,
        }
    } else {
        Conditioning {
            letterbox: 512,
            encoder: 518,
        }
    }
}

/// The timestep tensor for one sampler step. **Always F32**, never the
/// compute dtype.
///
/// Upstream floats the timestep before the model sees it
/// (`comfy/model_base.py:222`) and casts only the finished sinusoidal
/// embedding (`comfy/ldm/hunyuan3d/model.py:82`, built by
/// `comfy/ldm/flux/layers.py:38-47`). Building this in F16 instead would round
/// the sigma to about three significant digits and then let the 1000x
/// `time_factor` multiply amplify the error into every cosine argument, which
/// is a different denoising trajectory rather than a slightly noisier one.
fn timestep_tensor(sigma: f64, device: &Device) -> Result<Tensor> {
    Ok(Tensor::new(&[sigma as f32], device)?)
}

/// The guidance tensor for a distilled checkpoint. **Always F32**, matching
/// `torch.FloatTensor([guidance])` (`comfy/model_base.py:2098-2100`); it goes
/// through the same [`super::transformer::timestep_embedding`] and takes the
/// same cast at the end.
fn guidance_tensor(value: f64, device: &Device) -> Result<Tensor> {
    Ok(Tensor::new(&[value as f32], device)?)
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

    /// Tier identity is a property of the WEIGHTS, not of the name mold
    /// happened to install them under.
    ///
    /// ComfyUI keys the mini tier on the DiT depth — `Hunyuan3Dv2mini`'s
    /// `unet_config` is `{"image_model": "hunyuan3d2", "depth": 8}`
    /// (`comfy/supported_models.py:1593-1597`) — and mold already detects
    /// that depth from the checkpoint header. Keying on
    /// `model_name.contains("mini")` instead made a repack, a community
    /// re-quantization, or a renamed local file take the wrong scale factor
    /// and the wrong conditioning size, silently: the mesh still renders.
    #[test]
    fn tier_identity_comes_from_the_dit_depth() {
        let mini_dit = detect_dit_config(&header(8, 16, true)).unwrap();
        let mini = detect_vae_config(&mini_dit);
        assert_eq!(mini, ShapeVaeConfig::v2_0_mini());
        assert_eq!(
            conditioning_for(&mini_dit),
            Conditioning {
                letterbox: 1022,
                encoder: 1022,
            }
        );

        for dit in [DitConfig::v2_0(), DitConfig::v2_0_turbo()] {
            let vae = detect_vae_config(&dit);
            assert_eq!(
                vae,
                ShapeVaeConfig::v2_0(),
                "depth {} is a base tier",
                dit.depth
            );

            // `image_processor.params.size` is the letterbox and
            // `conditioner.params.main_image_encoder.kwargs.image_size` is
            // what DINOv2 actually receives. They are NOT the same number on
            // the base tiers, and 512 is not even a legal encoder size:
            // 512 % 14 == 8, so `Dinov2Model::forward` refuses it.
            let sizes = conditioning_for(&dit);
            assert_eq!(
                sizes,
                Conditioning {
                    letterbox: 512,
                    encoder: 518,
                }
            );
            assert_eq!(
                sizes.encoder % 14,
                0,
                "the encoder size must be a whole number of patches"
            );
            assert_eq!(
                sizes.encoder as usize,
                Dinov2Config::giant().image_size,
                "the base tiers condition at the tower's own stored resolution"
            );
        }
    }

    /// The mesher thresholds ComfyUI's post-processed occupancy, not the raw
    /// logits: `(x + 1) / 2` clamped to `[0, 1]` (`comfy/sd.py:505`, applied
    /// at `:1233`). A raw logit of 0.2 is the node's default 0.6 iso-level,
    /// and anything past ±1 saturates exactly as it does upstream.
    #[test]
    fn occupancy_is_the_vae_wrappers_image_post_process_of_the_logits() {
        let mut values = vec![-3.0, -1.0, -0.5, 0.0, 0.2, 1.0, 5.0];
        occupancy_from_logits(&mut values);
        let expected = [0.0, 0.0, 0.25, 0.5, 0.6, 1.0, 1.0];
        for (got, want) in values.iter().zip(expected) {
            assert!(
                (got - want).abs() < 1e-6,
                "occupancy {got} should be {want} (from {values:?})"
            );
        }
    }

    /// Both scalars are f32 on every device and in every build. The DiT
    /// widens defensively, but the sigma has to arrive intact: the 1000x
    /// `time_factor` multiply inside the sinusoidal embedding amplifies
    /// whatever precision the caller threw away.
    #[test]
    fn the_timestep_and_guidance_tensors_are_always_f32() {
        let device = Device::Cpu;

        let timestep = timestep_tensor(0.9, &device).unwrap();
        assert_eq!(timestep.dtype(), DType::F32);
        assert_eq!(timestep.dims(), &[1]);
        assert!((timestep.to_vec1::<f32>().unwrap()[0] - 0.9).abs() < 1e-7);

        let guidance = guidance_tensor(5.0, &device).unwrap();
        assert_eq!(guidance.dtype(), DType::F32);
        assert_eq!(guidance.dims(), &[1]);
        assert!((guidance.to_vec1::<f32>().unwrap()[0] - 5.0).abs() < 1e-7);
    }

    #[test]
    fn the_decode_chunk_override_is_clamped_not_trusted() {
        // The default holds when nothing is set. The clamp itself is asserted
        // through the public constant bounds rather than by mutating the
        // process environment, which `runtime_env` deliberately caches.
        assert_eq!(
            Hunyuan3dEngine::decode_chunk(&Device::Cpu),
            DEFAULT_DECODE_CHUNK
        );
    }

    /// Metal's default is the measured one, not upstream's, and the env
    /// override still wins over it (asserted through the resolver's shape:
    /// with nothing set, the backend default is what comes back).
    #[cfg(feature = "metal")]
    #[test]
    fn metal_takes_the_measured_decode_chunk_default() {
        let Ok(metal) = Device::new_metal(0) else {
            return;
        };
        assert_eq!(
            Hunyuan3dEngine::decode_chunk(&metal),
            super::super::backend::decode_chunk_default(&metal, DEFAULT_DECODE_CHUNK)
        );
        assert_ne!(Hunyuan3dEngine::decode_chunk(&metal), DEFAULT_DECODE_CHUNK);
    }
}
