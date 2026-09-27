//! Weight-gated end-to-end renders: the Qwen Image 2.1 ENGINE, loaded through
//! the header probe, renders one 1024² text-to-image per transformer tier.
//!
//! This is the engine's own path — `QwenImage21Engine::new` with the tier's
//! file as the transformer and the shared VAE / Qwen3-VL shards / tokenizer —
//! so it exercises exactly what a manifest tier will load, before the tier
//! manifests exist. Each PNG is written for visual review.
//!
//! Environment:
//! - `MOLD_QWEN_IMAGE21_TIERS_DIR`: the staged tier checkpoints
//! - `MOLD_QWEN_IMAGE21_BF16_DIR`: the BF16 model dir (`transformer/` shards)
//! - `MOLD_QWEN_IMAGE21_SHARED_DIR`: `shared/qwen-image21` (vae, text_encoder,
//!   processor)
//! - `MOLD_QWEN_IMAGE21_TIER_RENDER_DIR`: where the PNGs go
//! - `MOLD_QWEN_IMAGE21_TIERS` (optional): comma-separated tier names
//! - `MOLD_QWEN_IMAGE21_TIER_RENDER_SIZE` (optional): `WxH`, default 1024x1024

use std::path::PathBuf;

use mold_core::{GenerateRequest, ModelPaths};

use super::QwenImage21Engine;
use crate::engine::{InferenceEngine, LoadStrategy};

fn env_dir(name: &str) -> PathBuf {
    PathBuf::from(std::env::var(name).unwrap_or_else(|_| panic!("{name} must be set")))
}

/// `(tier, transformer files)`.
fn tiers() -> Vec<(&'static str, Vec<PathBuf>)> {
    let staged = env_dir("MOLD_QWEN_IMAGE21_TIERS_DIR");
    let bf16 = env_dir("MOLD_QWEN_IMAGE21_BF16_DIR").join("transformer");
    let one = |file: &str| vec![staged.join(file)];
    vec![
        (
            "bf16",
            vec![
                bf16.join("diffusion_pytorch_model-00001-of-00002.safetensors"),
                bf16.join("diffusion_pytorch_model-00002-of-00002.safetensors"),
            ],
        ),
        ("int8-conv", one("qwen_image_2.1_int8_convrot.safetensors")),
        ("fp8", one("Qwen-Image-2.1-FP8.safetensors")),
        ("q8", one("qwen_image_2.1-Q8_0.gguf")),
        ("q6", one("qwen_image_2.1-Q6_K.gguf")),
        ("q5", one("qwen_image_2.1-Q5_0.gguf")),
        ("q4", one("qwen_image_2.1-Q4_K.gguf")),
        ("q3", one("qwen_image_2.1-Q3_K.gguf")),
        ("q2", one("qwen_image_2.1-Q2_K.gguf")),
    ]
}

fn paths(transformer: Vec<PathBuf>) -> ModelPaths {
    let shared = env_dir("MOLD_QWEN_IMAGE21_SHARED_DIR");
    let (transformer, transformer_shards) = if transformer.len() == 1 {
        (transformer[0].clone(), Vec::new())
    } else {
        (transformer[0].clone(), transformer)
    };
    ModelPaths {
        low_noise_transformer: None,
        low_noise_distilled_lora: None,
        transformer,
        transformer_shards,
        vae: shared.join("vae/diffusion_pytorch_model.safetensors"),
        spatial_upscaler: None,
        temporal_upscaler: None,
        distilled_lora: None,
        t5_encoder: None,
        clip_encoder: None,
        t5_tokenizer: None,
        clip_tokenizer: None,
        clip_encoder_2: None,
        clip_tokenizer_2: None,
        text_encoder_files: (1..=4)
            .map(|i| shared.join(format!("text_encoder/model-0000{i}-of-00004.safetensors")))
            .collect(),
        text_tokenizer: Some(shared.join("processor/tokenizer.json")),
        decoder: None,
    }
}

#[test]
#[ignore = "needs the staged Qwen Image 2.1 tiers, the BF16 shards and a GPU"]
fn every_tier_renders_through_the_engine() {
    let out = env_dir("MOLD_QWEN_IMAGE21_TIER_RENDER_DIR");
    std::fs::create_dir_all(&out).unwrap();
    let selected = std::env::var("MOLD_QWEN_IMAGE21_TIERS").ok();
    let (width, height) = std::env::var("MOLD_QWEN_IMAGE21_TIER_RENDER_SIZE")
        .ok()
        .and_then(|size| {
            let (w, h) = size.split_once('x')?;
            Some((w.parse::<u32>().ok()?, h.parse::<u32>().ok()?))
        })
        .unwrap_or((1024, 1024));
    for (tier, transformer) in tiers() {
        if selected
            .as_deref()
            .is_some_and(|list| !list.split(',').any(|t| t.trim() == tier))
        {
            continue;
        }
        let model = format!("qwen-image-2.1:{tier}");
        let mut engine =
            QwenImage21Engine::new(model.clone(), paths(transformer), LoadStrategy::Eager, 0);
        let request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "A red fox curled asleep on a mossy stone in a misty pine forest at dawn, \
                       soft golden light, a hand-painted wooden sign reading \"MOON CAFE\"",
            "model": model,
            "width": width,
            "height": height,
            "steps": 40,
            "guidance": 1.0,
            "seed": 210001,
            "output_format": "png"
        }))
        .unwrap();
        let load_started = std::time::Instant::now();
        engine
            .load()
            .unwrap_or_else(|error| panic!("{tier} load: {error:#}"));
        let load_secs = load_started.elapsed().as_secs_f64();
        let started = std::time::Instant::now();
        let response = engine
            .generate(&request)
            .unwrap_or_else(|error| panic!("{tier}: {error:#}"));
        let image = &response.images[0];
        assert_eq!((image.width, image.height), (width, height), "{tier}");
        let path = out.join(format!("qwen-image-2.1-{tier}-{width}x{height}.png"));
        std::fs::write(&path, &image.data).unwrap();
        eprintln!(
            "TIER-RENDER {tier}: load {load_secs:.1}s, render {:.1}s -> {}",
            started.elapsed().as_secs_f64(),
            path.display()
        );
        engine.unload();
    }
}
