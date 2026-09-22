use super::*;

use candle_core::IndexOp;
use candle_transformers::models::z_image::postprocess_image;
use safetensors::tensor::{serialize_to_file, Dtype as SafeDtype, TensorView};
use serde_json::{json, Map, Value};
use std::path::{Path, PathBuf};
use std::time::Instant;

struct OwnedF32 {
    name: String,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

fn capture(name: impl Into<String>, tensor: &Tensor) -> Result<(OwnedF32, Value)> {
    let tensor = tensor.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
    let values = tensor.flatten_all()?.to_vec1::<f32>()?;
    anyhow::ensure!(
        values.iter().all(|value| value.is_finite()),
        "benchmark tensor contains a non-finite value"
    );
    let count = values.len() as f64;
    let min = values.iter().copied().fold(f32::INFINITY, f32::min);
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mean = values.iter().map(|&value| f64::from(value)).sum::<f64>() / count;
    let rms = (values
        .iter()
        .map(|&value| f64::from(value) * f64::from(value))
        .sum::<f64>()
        / count)
        .sqrt();
    let bytes = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    Ok((
        OwnedF32 {
            name: name.into(),
            shape: tensor.dims().to_vec(),
            bytes,
        },
        json!({ "finite": true, "min": min, "max": max, "mean": mean, "rms": rms }),
    ))
}

fn save_tensors(path: &Path, tensors: &[OwnedF32]) -> Result<()> {
    let views = tensors
        .iter()
        .map(|tensor| {
            Ok((
                tensor.name.as_str(),
                TensorView::new(SafeDtype::F32, tensor.shape.clone(), &tensor.bytes)?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    serialize_to_file(views, &None, path)?;
    Ok(())
}

/// Opt-in, real-checkpoint Metal qualification harness. It writes benchmark
/// artifacts only beneath QWEN_IMAGE21_BENCH_OUTPUT and never downloads data.
#[test]
#[ignore = "requires installed Qwen Image 2.1 weights and an idle Metal GPU"]
fn official_metal_mode_benchmark() -> Result<()> {
    use crate::progress::ProgressReporter;

    let root = PathBuf::from(std::env::var("QWEN_IMAGE21_MODEL_ROOT")?);
    let output = PathBuf::from(std::env::var("QWEN_IMAGE21_BENCH_OUTPUT")?);
    std::fs::create_dir_all(&output)?;
    let mode = std::env::var("QWEN_IMAGE21_BENCH_MODE").unwrap_or_else(|_| "math".into());
    let (dtype, fused_target, fused_ops) = match mode.as_str() {
        "math" => (DType::F32, false, false),
        "fused" => (DType::F32, true, false),
        "bf16" => (DType::BF16, true, false),
        "optimized" => (DType::F32, true, true),
        "bf16-optimized" => (DType::BF16, true, true),
        _ => anyhow::bail!(
            "QWEN_IMAGE21_BENCH_MODE must be math, fused, bf16, optimized, or bf16-optimized"
        ),
    };
    let steps =
        std::env::var("QWEN_IMAGE21_BENCH_STEPS").map_or(Ok(40usize), |value| value.parse())?;
    let limit =
        std::env::var("QWEN_IMAGE21_BENCH_LIMIT").map_or(Ok(steps), |value| value.parse())?;
    anyhow::ensure!(
        steps > 0 && limit > 0,
        "benchmark steps and limit must be positive"
    );
    let prompt = std::env::var("QWEN_IMAGE21_BENCH_PROMPT").unwrap_or_else(|_| {
        "A small red ceramic teapot on a sunlit wooden windowsill, editorial product photograph, soft morning shadows".into()
    });
    let seed =
        std::env::var("QWEN_IMAGE21_BENCH_SEED").map_or(Ok(210001u64), |value| value.parse())?;
    let device = Device::new_metal(0)?;
    let progress = ProgressReporter::default();
    let started = Instant::now();
    let mut phases = Map::new();

    let shared = root.join("shared/qwen-image21");
    let text_paths = (1..=4)
        .map(|i| shared.join(format!("text_encoder/model-{i:05}-of-00004.safetensors")))
        .collect::<Vec<_>>();
    device.synchronize()?;
    let phase = Instant::now();
    let mut encoder = crate::encoders::qwen3::Qwen3Encoder::load_bf16(
        &text_paths,
        &shared.join("processor/tokenizer.json"),
        &device,
        DType::F32,
        &crate::encoders::qwen3_bf16::Qwen3BF16Config::qwen3_image_21_text_encoder(),
        &progress,
    )?;
    device.synchronize()?;
    let seconds = phase.elapsed().as_secs_f64();
    eprintln!("mode={mode} phase=encoder_load seconds={seconds:.4}");
    phases.insert("encoder_load_seconds".into(), json!(seconds));
    device.synchronize()?;
    let phase = Instant::now();
    let conditioning =
        super::super::encode_t2i_prompts(&mut encoder, std::slice::from_ref(&prompt))?
            .to_device_dtype(&device, dtype)?;
    device.synchronize()?;
    let seconds = phase.elapsed().as_secs_f64();
    eprintln!("mode={mode} phase=prompt_encode seconds={seconds:.4}");
    phases.insert("prompt_encode_seconds".into(), json!(seconds));
    let prefix_tokens = conditioning.sequence_length();
    drop(encoder);
    device.synchronize()?;

    let transformer_paths = (1..=2)
        .map(|i| root.join(format!("qwen-image-2.1-bf16/transformer/diffusion_pytorch_model-{i:05}-of-00002.safetensors")))
        .collect::<Vec<_>>();
    device.synchronize()?;
    let phase = Instant::now();
    let mut transformer =
        QwenImage21Transformer::load(&transformer_paths, &device, dtype, &progress)?;
    transformer.compact_modulation = fused_ops;
    for block in &mut transformer.blocks {
        block.attn.fused_target = fused_target;
        block.attn.fused_ops = fused_ops;
    }
    device.synchronize()?;
    let seconds = phase.elapsed().as_secs_f64();
    eprintln!("mode={mode} phase=transformer_load seconds={seconds:.4}");
    phases.insert("transformer_load_seconds".into(), json!(seconds));

    let mut scheduler = super::super::scheduler::QwenImage21Scheduler::new(
        steps,
        4096,
        super::super::scheduler::QwenShiftPolicy::DynamicResolution,
    );
    let noise =
        crate::engine::seeded_randn(seed, &[1, 4096, 64], &device, DType::F32)?.to_dtype(dtype)?;
    let mut latents = (noise * scheduler.initial_sigma())?;
    let total_steps = scheduler.num_steps();
    let executed_steps = limit.min(total_steps);
    let mut prepared = transformer.prepare_t2i(&conditioning, 64, 64);
    let mut predictions = Vec::new();
    let mut step_receipts = Vec::with_capacity(executed_steps);
    device.synchronize()?;
    let denoise_started = Instant::now();
    for step in 0..executed_steps {
        let timestep = scheduler.current_timestep() / 1000.0;
        device.synchronize()?;
        let step_started = Instant::now();
        let prediction = prepared.forward(&latents, timestep)?;
        device.synchronize()?;
        let seconds = step_started.elapsed().as_secs_f64();
        eprintln!(
            "mode={mode} step={}/{} seconds={seconds:.4}",
            step + 1,
            executed_steps
        );
        let (prediction_copy, prediction_stats) =
            capture(format!("prediction_{step:03}"), &prediction)?;
        if step < 4 {
            predictions.push(prediction_copy);
        }
        latents = scheduler.step(&prediction, &latents)?;
        let (_, latent_stats) = capture("discard", &latents)?;
        step_receipts.push(json!({
            "step": step + 1,
            "timestep": timestep,
            "seconds": seconds,
            "metal_allocated_bytes": device.as_metal_device()?.device().current_allocated_size(),
            "prediction": prediction_stats,
            "latent": latent_stats,
        }));
    }
    device.synchronize()?;
    phases.insert(
        "denoise_seconds".into(),
        json!(denoise_started.elapsed().as_secs_f64()),
    );
    save_tensors(
        &output.join(format!("predictions-{mode}.safetensors")),
        &predictions,
    )?;
    drop(prepared);
    drop(transformer);
    drop(conditioning);
    device.synchronize()?;

    let complete = executed_steps == total_steps;
    let (_, final_latent_stats) = capture("final_latent", &latents)?;
    if complete {
        let (final_latent, _) = capture("latent", &latents)?;
        save_tensors(
            &output.join(format!("final-latent-{mode}.safetensors")),
            &[final_latent],
        )?;
        let vae_path = shared.join("vae/diffusion_pytorch_model.safetensors");
        device.synchronize()?;
        let phase = Instant::now();
        let vae =
            super::super::vae::QwenImage21Vae::load(&vae_path, &device, DType::F32, &progress)?;
        device.synchronize()?;
        let seconds = phase.elapsed().as_secs_f64();
        eprintln!("mode={mode} phase=vae_load seconds={seconds:.4}");
        phases.insert("vae_load_seconds".into(), json!(seconds));
        device.synchronize()?;
        let phase = Instant::now();
        let decoded = vae.decode_packed(&latents.to_dtype(DType::F32)?, 64, 64)?;
        let (_, decoded_stats) = capture("decoded", &decoded)?;
        phases.insert("decoded_stats".into(), decoded_stats);
        let image = postprocess_image(&decoded)?.narrow(1, 0, 3)?.i(0)?;
        device.synchronize()?;
        let seconds = phase.elapsed().as_secs_f64();
        eprintln!("mode={mode} phase=vae_decode seconds={seconds:.4}");
        phases.insert("vae_decode_seconds".into(), json!(seconds));
        let png =
            crate::image::encode_image(&image, mold_core::OutputFormat::Png, 1024, 1024, None)?;
        std::fs::write(output.join(format!("image-{mode}.png")), png)?;
    }
    device.synchronize()?;
    phases.insert(
        "total_seconds".into(),
        json!(started.elapsed().as_secs_f64()),
    );
    let receipt = json!({
        "mode": mode,
        "width": 1024,
        "height": 1024,
        "guidance": 1.0,
        "device": format!("{device:?}"),
        "transformer_dtype": format!("{dtype:?}"),
        "encoder_dtype": "F32",
        "vae_dtype": "F32",
        "fused_target": fused_target,
        "fused_ops": fused_ops,
        "prompt": prompt,
        "seed": seed,
        "prefix_tokens": prefix_tokens,
        "schedule_steps": total_steps,
        "executed_steps": executed_steps,
        "complete": complete,
        "finite": true,
        "final_latent": final_latent_stats,
        "phases": phases,
        "steady_forward_seconds": step_receipts.iter().skip(1).map(|s| s["seconds"].as_f64().unwrap()).sum::<f64>(),
        "steps": step_receipts,
    });
    std::fs::write(
        output.join(format!("receipt-{mode}.json")),
        serde_json::to_vec_pretty(&receipt)?,
    )?;
    Ok(())
}
