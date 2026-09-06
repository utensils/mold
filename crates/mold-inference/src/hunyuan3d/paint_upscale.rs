//! Untiled RealESRGAN stage for ordered paint material views.

use anyhow::{ensure, Result};
use image::RgbImage;
use mold_core::{OutputFormat, UpscaleRequest};

use super::paint_pixels::PaintMaterials;
use crate::engine::LoadStrategy;
use crate::progress::InferenceCancellationToken;
use crate::upscaler::{
    create_upscale_engine_from_resolved_plan, with_upscale_cancellation, ExactUpscalePlacement,
    ResolvedUpscaleExecutionPlan, UpscaleEngine,
};

const MODEL: &str = "real-esrgan-x4plus:fp16";

/// Run both material streams on the planned accelerator, then release the
/// upscaler before baking. The engine never falls back to another device.
pub fn upscale_materials(
    plan: ResolvedUpscaleExecutionPlan,
    materials: &PaintMaterials,
    token: InferenceCancellationToken,
    mut checkpoint: impl FnMut() -> Result<()>,
) -> Result<PaintMaterials> {
    validate_materials(materials)?;
    ensure!(plan.model_name == MODEL, "paint requires {MODEL}");
    ensure!(
        matches!(plan.placement, ExactUpscalePlacement::Device { .. }),
        "paint upscaling requires an accelerator for its FP16 recipe"
    );
    checkpoint()?;
    token.checkpoint()?;
    let engine = create_upscale_engine_from_resolved_plan(plan, LoadStrategy::Eager)?;
    let _scope = crate::conv_policy::ConvScope::apply(crate::conv_policy::resolve_for(
        crate::conv_policy::ConvPolicy::Paint,
    ));
    run_materials(engine, materials, token, checkpoint)
}

struct StageEngine(Box<dyn UpscaleEngine>);
impl Drop for StageEngine {
    fn drop(&mut self) {
        self.0.unload();
    }
}

fn validate_materials(materials: &PaintMaterials) -> Result<()> {
    ensure!(
        (1..=6).contains(&materials.albedo.len())
            && materials.albedo.len() == materials.metallic_roughness.len(),
        "paint requires matching nonempty material view sets, at most six views"
    );
    ensure!(
        materials
            .albedo
            .iter()
            .chain(&materials.metallic_roughness)
            .all(|image| image.dimensions() == (512, 512)),
        "paint upscaling requires 512x512 material views"
    );
    Ok(())
}

fn reverse_channels(image: &RgbImage) -> RgbImage {
    let mut image = image.clone();
    for pixel in image.pixels_mut() {
        pixel.0.swap(0, 2);
    }
    image
}

fn run_materials(
    engine: Box<dyn UpscaleEngine>,
    materials: &PaintMaterials,
    token: InferenceCancellationToken,
    mut checkpoint: impl FnMut() -> Result<()>,
) -> Result<PaintMaterials> {
    let mut engine = StageEngine(engine);
    validate_materials(materials)?;
    ensure!(engine.0.model_name() == MODEL, "paint requires {MODEL}");
    let check_token = token.clone();
    let mut check = || -> Result<()> {
        checkpoint()?;
        Ok(check_token.checkpoint()?)
    };
    check()?;
    with_upscale_cancellation(engine.0.as_mut(), token, |engine| {
        engine.load()?;
        check()?;
        ensure!(engine.scale_factor() == 4, "paint requires a 4x upscaler");
        let mut images = Vec::with_capacity(materials.albedo.len() * 2);
        for image in materials.albedo.iter().chain(&materials.metallic_roughness) {
            check()?;
            // Tencent passes PIL RGB directly into the BGR-oriented enhance
            // wrapper. Preserve both reversals; MR channels remain raw data.
            let input = reverse_channels(image);
            let response = engine.upscale(&UpscaleRequest {
                model: MODEL.to_string(),
                image: crate::image::encode_rgb_image(&input, OutputFormat::Png, None)?,
                output_format: OutputFormat::Png,
                tile_size: Some(0),
                metadata: None,
            })?;
            check()?;
            ensure!(
                response.model == MODEL
                    && response.scale_factor == 4
                    && response.original_width == 512
                    && response.original_height == 512
                    && response.image.width == 2048
                    && response.image.height == 2048
                    && response.image.format == OutputFormat::Png,
                "paint upscaler returned an incompatible response"
            );
            let image =
                image::load_from_memory_with_format(&response.image.data, image::ImageFormat::Png)?
                    .to_rgb8();
            ensure!(
                image.dimensions() == (2048, 2048),
                "paint upscaler returned invalid image dimensions"
            );
            images.push(reverse_channels(&image));
        }
        check()?;
        let metallic_roughness = images.split_off(materials.albedo.len());
        Ok(PaintMaterials {
            albedo: images,
            metallic_roughness,
        })
    })
}

#[cfg(test)]
mod tests {
    use super::super::paint_pixels::PaintMaterials;
    use super::*;
    use crate::progress::{InferenceCancellationToken, ProgressCallback};
    use crate::upscaler::UpscaleEngine;
    use anyhow::Result;
    use image::{Rgb, RgbImage};
    use mold_core::{ImageData, OutputFormat, UpscaleRequest, UpscaleResponse};
    use std::sync::{Arc, Mutex};

    #[test]
    #[cfg(feature = "cuda")]
    #[ignore = "requires CUDA, installed upscaler and retained original material oracles"]
    fn installed_material_adapter_matches_tencent_pixels() -> Result<()> {
        use std::path::PathBuf;
        let albedo = PathBuf::from(std::env::var("MOLD_PAINT_UPSCALE_ALBEDO_ORACLE")?);
        let mr = PathBuf::from(std::env::var("MOLD_PAINT_UPSCALE_MR_ORACLE")?);
        let output = PathBuf::from(std::env::var("MOLD_PAINT_UPSCALE_OUTPUT")?);
        let weights = PathBuf::from(std::env::var("MOLD_PAINT_UPSCALER_WEIGHTS")?);
        std::fs::create_dir(&output)?;
        for oracle in [&albedo, &mr] {
            let metadata: serde_json::Value =
                serde_json::from_slice(&std::fs::read(oracle.join("completed.json"))?)?;
            ensure!(
                metadata.get("diagnostic_first_features").is_none(),
                "diagnostic oracle cannot qualify material adapter"
            );
        }
        let materials = PaintMaterials {
            albedo: vec![image::open(albedo.join("input.png"))?.to_rgb8()],
            metallic_roughness: vec![image::open(mr.join("input.png"))?.to_rgb8()],
        };
        let plan = crate::upscaler::resolve_upscale_execution_plan(
            MODEL,
            &weights,
            None,
            ExactUpscalePlacement::Device {
                backend: mold_core::GpuBackend::Cuda,
                ordinal: 0,
            },
        )?;
        let start = std::time::Instant::now();
        let result = upscale_materials(
            plan,
            &materials,
            InferenceCancellationToken::default(),
            || Ok(()),
        )?;
        let seconds = start.elapsed().as_secs_f64();
        let mut comparison = serde_json::Map::new();
        let mut failures = Vec::new();
        for (name, actual, oracle) in [
            ("albedo", &result.albedo[0], albedo),
            ("mr", &result.metallic_roughness[0], mr),
        ] {
            actual.save(output.join(format!("{name}.png")))?;
            let expected = image::open(oracle.join("expected.png"))?.to_rgb8();
            ensure!(
                actual.dimensions() == expected.dimensions(),
                "oracle dimensions differ"
            );
            let max = actual
                .as_raw()
                .iter()
                .zip(expected.as_raw())
                .map(|(a, b)| a.abs_diff(*b))
                .max()
                .unwrap();
            comparison.insert(name.into(), serde_json::json!({"max_byte":max}));
            if max != 0 {
                failures.push(name);
            }
        }
        comparison.insert("seconds".into(), serde_json::json!(seconds));
        std::fs::write(
            output.join("comparison.json"),
            serde_json::to_vec_pretty(&comparison)?,
        )?;
        ensure!(
            failures.is_empty(),
            "material adapter pixel differences: {failures:?}"
        );
        Ok(())
    }

    #[derive(Default)]
    struct State {
        seen: Vec<[u8; 3]>,
        loaded: usize,
        unloaded: usize,
        cleared: usize,
    }
    struct Fake {
        state: Arc<Mutex<State>>,
        token: Option<InferenceCancellationToken>,
        cancel: bool,
    }
    impl UpscaleEngine for Fake {
        fn upscale(&mut self, request: &UpscaleRequest) -> Result<UpscaleResponse> {
            assert_eq!(request.tile_size, Some(0));
            assert_eq!(request.output_format, OutputFormat::Png);
            assert_eq!(request.model, "real-esrgan-x4plus:fp16");
            let source = image::load_from_memory(&request.image)?.to_rgb8();
            assert_eq!(source.dimensions(), (512, 512));
            let pixel = source.get_pixel(0, 0).0;
            self.state.lock().unwrap().seen.push(pixel);
            if self.cancel {
                self.token.as_ref().unwrap().cancel();
            }
            let image = RgbImage::from_pixel(2048, 2048, Rgb(pixel));
            Ok(UpscaleResponse {
                image: ImageData {
                    data: crate::image::encode_rgb_image(&image, OutputFormat::Png, None)?,
                    format: OutputFormat::Png,
                    width: 2048,
                    height: 2048,
                    index: 0,
                },
                upscale_time_ms: 0,
                model: request.model.clone(),
                scale_factor: 4,
                original_width: 512,
                original_height: 512,
            })
        }
        fn model_name(&self) -> &str {
            "real-esrgan-x4plus:fp16"
        }
        fn is_loaded(&self) -> bool {
            self.state.lock().unwrap().loaded != 0
        }
        fn load(&mut self) -> Result<()> {
            self.state.lock().unwrap().loaded += 1;
            Ok(())
        }
        fn unload(&mut self) {
            self.state.lock().unwrap().unloaded += 1;
        }
        fn scale_factor(&self) -> u32 {
            4
        }
        fn set_on_progress(&mut self, _: ProgressCallback) {}
        fn clear_on_progress(&mut self) {}
        fn set_cancellation_token(&mut self, token: InferenceCancellationToken) {
            self.token = Some(token);
        }
        fn clear_cancellation_token(&mut self) {
            self.token = None;
            self.state.lock().unwrap().cleared += 1;
        }
    }
    fn materials() -> PaintMaterials {
        PaintMaterials {
            albedo: vec![
                RgbImage::from_pixel(512, 512, Rgb([11, 22, 33])),
                RgbImage::from_pixel(512, 512, Rgb([44, 55, 66])),
            ],
            metallic_roughness: vec![
                RgbImage::from_pixel(512, 512, Rgb([77, 88, 99])),
                RgbImage::from_pixel(512, 512, Rgb([101, 112, 123])),
            ],
        }
    }
    #[test]
    fn ordered_materials_reverse_channels_and_share_one_loaded_engine() -> Result<()> {
        let state = Arc::new(Mutex::new(State::default()));
        let output = run_materials(
            Box::new(Fake {
                state: state.clone(),
                token: None,
                cancel: false,
            }),
            &materials(),
            InferenceCancellationToken::default(),
            || Ok(()),
        )?;
        assert_eq!(
            output
                .albedo
                .iter()
                .map(|x| x.get_pixel(0, 0).0)
                .collect::<Vec<_>>(),
            [[11, 22, 33], [44, 55, 66]]
        );
        assert_eq!(
            output
                .metallic_roughness
                .iter()
                .map(|x| x.get_pixel(0, 0).0)
                .collect::<Vec<_>>(),
            [[77, 88, 99], [101, 112, 123]]
        );
        let state = state.lock().unwrap();
        assert_eq!(
            state.seen,
            [[33, 22, 11], [66, 55, 44], [99, 88, 77], [123, 112, 101]]
        );
        assert_eq!((state.loaded, state.unloaded, state.cleared), (1, 1, 1));
        Ok(())
    }
    #[test]
    fn cancellation_drops_the_stage_without_returning_partial_materials() {
        let state = Arc::new(Mutex::new(State::default()));
        let result = run_materials(
            Box::new(Fake {
                state: state.clone(),
                token: None,
                cancel: true,
            }),
            &materials(),
            InferenceCancellationToken::default(),
            || Ok(()),
        );
        assert!(crate::progress::is_inference_cancelled(
            &result.err().unwrap()
        ));
        let state = state.lock().unwrap();
        assert_eq!(state.seen.len(), 1);
        assert_eq!((state.unloaded, state.cleared), (1, 1));
    }
    #[test]
    fn malformed_materials_are_refused_before_loading() {
        let state = Arc::new(Mutex::new(State::default()));
        let mut input = materials();
        input.metallic_roughness.pop();
        assert!(run_materials(
            Box::new(Fake {
                state: state.clone(),
                token: None,
                cancel: false
            }),
            &input,
            InferenceCancellationToken::default(),
            || Ok(())
        )
        .is_err());
        assert_eq!(state.lock().unwrap().loaded, 0);
    }
}
