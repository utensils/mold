//! End-to-end Hunyuan3D Paint orchestration around the individually qualified stages.

use anyhow::{ensure, Context, Result};
use candle_core::{DType, Device, Tensor};
use image::{Rgb, RgbImage, RgbaImage};
use mold_core::hunyuan3d_paint_assets::Hunyuan3dPaintPaths;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, StandardNormal};

use super::{
    mesh::Mesh,
    paint_images::PaintImages,
    paint_materials::{bake_materials, encode_textured_glb, finish_materials},
    paint_pipeline::{PaintCheckpoints, PaintEvent, PaintRequest, PaintStage},
    paint_raster::{prepare_mesh, render_with_checkpoint, PreparedMesh},
    paint_upscale::upscale_materials,
    paint_views::{candidate_views, select_views, PaintView},
    uv,
};
use crate::{
    progress::ProgressReporter,
    upscaler::{resolve_upscale_execution_plan, ExactUpscalePlacement},
};

const VIEW_SIZE: u32 = 512;
const SELECTION_SIZE: u32 = 1024;

pub struct TexturedMesh {
    pub mesh: Mesh,
    pub glb: Vec<u8>,
}

pub struct PaintRuntime<'a> {
    pub assets: &'a Hunyuan3dPaintPaths,
    pub device: &'a Device,
    pub gpu_ordinal: usize,
    pub progress: &'a ProgressReporter,
}

fn face_areas(mesh: &Mesh) -> Vec<f32> {
    mesh.faces
        .iter()
        .map(|face| {
            let [a, b, c] = face.map(|index| mesh.vertices[index as usize]);
            let ab = [0, 1, 2].map(|axis| b[axis] - a[axis]);
            let ac = [0, 1, 2].map(|axis| c[axis] - a[axis]);
            let cross = [
                ab[1] * ac[2] - ab[2] * ac[1],
                ab[2] * ac[0] - ab[0] * ac[2],
                ab[0] * ac[1] - ab[1] * ac[0],
            ];
            0.5 * cross.iter().map(|value| value * value).sum::<f32>().sqrt()
        })
        .collect()
}

fn selected_views(
    mesh: &PreparedMesh,
    checkpoint: &mut dyn FnMut() -> Result<()>,
) -> Result<Vec<PaintView>> {
    let candidates = candidate_views();
    let mut visible = Vec::with_capacity(candidates.len());
    for view in &candidates {
        checkpoint()?;
        let buffers = render_with_checkpoint(
            mesh,
            view.elevation,
            view.azimuth,
            SELECTION_SIZE,
            checkpoint,
        )?;
        let mut faces = buffers
            .mask
            .iter()
            .zip(&buffers.face_ids)
            .filter_map(|(visible, face)| visible.then_some(*face))
            .collect::<Vec<_>>();
        faces.sort_unstable();
        faces.dedup();
        visible.push(faces);
    }
    let selected = select_views(&face_areas(&mesh.mesh), &visible, 6)?;
    Ok(selected
        .into_iter()
        .map(|index| candidates[index])
        .collect())
}

fn unit_byte(value: f32) -> u8 {
    (value.clamp(0., 1.) * 255.) as u8
}

fn condition_images(
    mesh: &PreparedMesh,
    views: &[PaintView],
    checkpoint: &mut dyn FnMut() -> Result<()>,
) -> Result<(Vec<RgbImage>, Vec<RgbImage>)> {
    let mut normals = Vec::with_capacity(views.len());
    let mut positions = Vec::with_capacity(views.len());
    for view in views {
        checkpoint()?;
        let buffers =
            render_with_checkpoint(mesh, view.elevation, view.azimuth, VIEW_SIZE, checkpoint)?;
        let mut normal = RgbImage::from_pixel(VIEW_SIZE, VIEW_SIZE, Rgb([255; 3]));
        let mut position = normal.clone();
        for pixel in 0..buffers.mask.len() {
            if pixel.is_multiple_of(4096) {
                checkpoint()?;
            }
            if !buffers.mask[pixel] {
                continue;
            }
            let x = pixel as u32 % VIEW_SIZE;
            let y = pixel as u32 / VIEW_SIZE;
            normal.put_pixel(
                x,
                y,
                Rgb(buffers.normal[pixel].map(|value| unit_byte((value + 1.) * 0.5))),
            );
            position.put_pixel(
                x,
                y,
                Rgb(buffers.position[pixel].map(|value| unit_byte(0.5 - value / 1.15))),
            );
        }
        normals.push(normal);
        positions.push(position);
    }
    Ok((normals, positions))
}

fn randn(rng: &mut StdRng, shape: &[usize]) -> Result<Tensor> {
    let values = (0..shape.iter().product())
        .map(|_| StandardNormal.sample(rng))
        .collect::<Vec<f32>>();
    Ok(Tensor::from_vec(values, shape, &Device::Cpu)?)
}

fn request(images: PaintImages, views: usize, seed: u64) -> Result<PaintRequest> {
    let latent = VIEW_SIZE as usize / 8;
    let mut rng = StdRng::seed_from_u64(seed);
    Ok(PaintRequest {
        appearance: images.appearance,
        reference: images.reference,
        normal: images.normal,
        position: images.position,
        reference_noise: randn(&mut rng, &[1, 1, 4, latent, latent])?,
        normal_noise: randn(&mut rng, &[1, views, 4, latent, latent])?,
        position_noise: randn(&mut rng, &[1, views, 4, latent, latent])?,
        initial_noise: randn(&mut rng, &[2 * views, 4, latent, latent])?,
    })
}

fn stage_name(stage: PaintStage) -> &'static str {
    match stage {
        PaintStage::Appearance => "Encoding paint appearance",
        PaintStage::Reference => "Encoding paint reference",
        PaintStage::Normal => "Encoding paint normals",
        PaintStage::Position => "Encoding paint positions",
        PaintStage::Denoise => "Generating PBR views",
        PaintStage::Decode => "Decoding PBR views",
    }
}

fn report(progress: &ProgressReporter, event: PaintEvent<'_>) -> Result<()> {
    progress.checkpoint()?;
    let name = stage_name(event.stage);
    if event.step == 0 {
        progress.stage_start(name);
    } else {
        progress.stage_progress(name, event.step, event.total);
    }
    Ok(())
}

impl PaintRuntime<'_> {
    pub fn generate(
        &self,
        source: &Mesh,
        appearance: &RgbaImage,
        texture_size: u32,
        seed: u64,
    ) -> Result<TexturedMesh> {
        let assets = self.assets;
        let device = self.device;
        let gpu_ordinal = self.gpu_ordinal;
        let progress = self.progress;
        ensure!(
            [1024, 2048, 4096].contains(&texture_size),
            "invalid paint texture size"
        );
        ensure!(
            device.is_cuda() || device.is_metal(),
            "Hunyuan3D Paint requires an accelerator"
        );
        let token = progress.cancellation_token();
        progress.stage_start("Unwrapping mesh");
        let unwrapped = uv::unwrap(source, token.flag()).context("unwrap the generated mesh")?;
        let prepared = prepare_mesh(&unwrapped)?;
        progress.stage_start("Selecting paint views");
        let mut checkpoint = || -> Result<()> { Ok(progress.checkpoint()?) };
        let views = selected_views(&prepared, &mut checkpoint)?;
        let (normals, positions) = condition_images(&prepared, &views, &mut checkpoint)?;
        let images = PaintImages::prepare(
            appearance,
            &normals,
            &positions,
            VIEW_SIZE,
            DType::F16,
            &mut checkpoint,
        )?;
        let input = request(images, views.len(), seed)?;
        let _scope = crate::conv_policy::ConvScope::apply(crate::conv_policy::resolve_for(
            crate::conv_policy::ConvPolicy::Paint,
        ));
        let materials = PaintCheckpoints {
            dino: &assets.dino,
            vae: &assets.vae,
            unet: &assets.unet,
        }
        .run(&input, DType::F16, device, |event| report(progress, event))?;
        let placement = ExactUpscalePlacement::Device {
            backend: if device.is_cuda() {
                mold_core::GpuBackend::Cuda
            } else {
                mold_core::GpuBackend::Metal
            },
            ordinal: gpu_ordinal,
        };
        let upscale = resolve_upscale_execution_plan(
            "real-esrgan-x4plus:fp16",
            &assets.upscaler,
            None,
            placement,
        )?
        .for_paint_materials()?;
        let materials =
            upscale_materials(upscale, &materials, token, || Ok(progress.checkpoint()?))?;
        progress.stage_start("Baking PBR textures");
        let baked = bake_materials(
            &prepared,
            &materials,
            &views,
            texture_size,
            device,
            &mut checkpoint,
        )?;
        let textures = finish_materials(&prepared, baked, &mut checkpoint)?;
        let (mesh, glb) = encode_textured_glb(&prepared, &textures, None, &mut checkpoint)?;
        Ok(TexturedMesh { mesh, glb })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn four_noise_draws_share_one_deterministic_stream() -> Result<()> {
        let image = Tensor::zeros((1, 1, 3, 512, 512), DType::F16, &Device::Cpu)?;
        let images = || -> Result<PaintImages> {
            Ok(PaintImages {
                appearance: Tensor::zeros((1, 3, 224, 224), DType::F32, &Device::Cpu)?,
                reference: image.clone(),
                normal: image.clone(),
                position: image.clone(),
            })
        };
        let a = request(images()?, 1, 42)?;
        let b = request(images()?, 1, 42)?;
        assert_eq!(
            a.initial_noise.flatten_all()?.to_vec1::<f32>()?,
            b.initial_noise.flatten_all()?.to_vec1::<f32>()?
        );
        assert_ne!(
            a.reference_noise.flatten_all()?.to_vec1::<f32>()?,
            a.normal_noise.flatten_all()?.to_vec1::<f32>()?
        );
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires CUDA, installed Hunyuan3D Paint assets, and a retained source mesh"]
    fn installed_runtime_writes_a_self_contained_textured_glb() -> Result<()> {
        use std::path::PathBuf;

        let mesh_path = PathBuf::from(std::env::var("MOLD_PAINT_RUNTIME_MESH")?);
        let appearance_path = PathBuf::from(std::env::var("MOLD_PAINT_RUNTIME_APPEARANCE")?);
        let output = PathBuf::from(std::env::var("MOLD_PAINT_RUNTIME_OUTPUT")?);
        std::fs::create_dir(&output)?;
        let mesh = super::super::glb::read_glb(&std::fs::read(mesh_path)?)?;
        let appearance = image::open(appearance_path)?.to_rgba8();
        let assets = Hunyuan3dPaintPaths {
            unet: PathBuf::from(std::env::var("MOLD_PAINT_UNET_WEIGHTS")?),
            vae: PathBuf::from(std::env::var("MOLD_PAINT_VAE_CHECKPOINT")?),
            dino: PathBuf::from(std::env::var("MOLD_PAINT_DINO_CHECKPOINT")?),
            upscaler: PathBuf::from(std::env::var("MOLD_PAINT_UPSCALER_WEIGHTS")?),
        };
        let device = Device::new_cuda(0)?;
        let started = std::time::Instant::now();
        let result = PaintRuntime {
            assets: &assets,
            device: &device,
            gpu_ordinal: 0,
            progress: &ProgressReporter::default(),
        }
        .generate(&mesh, &appearance, 1024, 42)?;
        std::fs::write(output.join("textured.glb"), &result.glb)?;
        let summary = mold_core::glb_summary::summarize_glb(&result.glb)
            .context("runtime output is not a valid summarized GLB")?;
        ensure!(
            summary.textured,
            "runtime GLB has no embedded base color texture"
        );
        ensure!(result.mesh.uvs.is_some(), "runtime mesh lost its UVs");
        std::fs::write(
            output.join("completed.json"),
            serde_json::to_vec_pretty(&serde_json::json!({
                "seconds": started.elapsed().as_secs_f64(),
                "bytes": result.glb.len(),
                "vertices": result.mesh.vertex_count(),
                "faces": result.mesh.face_count(),
                "textured": summary.textured,
            }))?,
        )?;
        Ok(())
    }
}
