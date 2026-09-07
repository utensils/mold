use std::{collections::HashMap, env, fs, path::PathBuf};

use anyhow::{ensure, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use mold_inference::hunyuan3d::shape_vae::{ShapeVaeEncoder, ShapeVaeEncoderConfig};
use serde_json::json;

const MEAN_MAX_ABS_TOLERANCE: f32 = 0.02;
const LOGVAR_MAX_ABS_TOLERANCE: f32 = 0.04;

fn difference(actual: &Tensor, expected: &Tensor) -> Result<(f32, f32)> {
    let delta = (actual.to_dtype(DType::F32)? - expected.to_dtype(DType::F32)?)?;
    let maximum = delta.abs()?.max_all()?.to_scalar::<f32>()?;
    let count = delta.elem_count() as f64;
    let rms = (delta.sqr()?.sum_all()?.to_scalar::<f32>()? as f64 / count).sqrt() as f32;
    Ok((maximum, rms))
}

fn main() -> Result<()> {
    let mut args = env::args_os().skip(1);
    let checkpoint = PathBuf::from(
        args.next()
            .context("usage: probe CHECKPOINT ORACLE.safetensors OUTPUT_DIR")?,
    );
    let oracle_path = PathBuf::from(
        args.next()
            .context("usage: probe CHECKPOINT ORACLE.safetensors OUTPUT_DIR")?,
    );
    let output = PathBuf::from(
        args.next()
            .context("usage: probe CHECKPOINT ORACLE.safetensors OUTPUT_DIR")?,
    );
    ensure!(
        args.next().is_none(),
        "usage: probe CHECKPOINT ORACLE.safetensors OUTPUT_DIR"
    );
    ensure!(checkpoint.is_file(), "checkpoint does not exist");
    ensure!(oracle_path.is_file(), "oracle does not exist");
    fs::create_dir(&output).with_context(|| format!("create {}", output.display()))?;

    let cpu = Device::Cpu;
    let oracle = candle_core::safetensors::load(&oracle_path, &cpu)?;
    let points = oracle["points"].to_dtype(DType::F16)?;
    let features = oracle["features"].to_dtype(DType::F16)?;
    let selected_i64 = oracle["selected"].to_vec1::<i64>()?;
    let selected: Vec<usize> = selected_i64
        .into_iter()
        .map(|value| usize::try_from(value).context("negative oracle FPS index"))
        .collect::<Result<_>>()?;

    let mut cfg = ShapeVaeEncoderConfig::v2_1();
    cfg.pc_size = points.dim(1)?;
    cfg.num_latents = selected.len();
    cfg.downsample_ratio = cfg.pc_size / cfg.num_latents;
    let device = Device::new_cuda(0)?;
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(std::slice::from_ref(&checkpoint), DType::F16, &device)?
    };
    let encoder = ShapeVaeEncoder::new(&cfg, vb.pp("vae"))?;
    let actual = encoder.encode_preselected(
        &points.to_device(&device)?,
        &features.to_device(&device)?,
        &selected,
        None,
        selected.len(),
    )?;
    let actual_mean = actual.mean.to_dtype(DType::F32)?.to_device(&cpu)?;
    let actual_logvar = actual.logvar.to_dtype(DType::F32)?.to_device(&cpu)?;
    let (mean_max, mean_rms) = difference(&actual_mean, &oracle["mean"])?;
    let (logvar_max, logvar_rms) = difference(&actual_logvar, &oracle["logvar"])?;
    candle_core::safetensors::save(
        &HashMap::from([
            ("mean".to_owned(), actual_mean),
            ("logvar".to_owned(), actual_logvar),
        ]),
        output.join("encoder-candle.safetensors"),
    )?;
    let report = json!({
        "schema": "mold.hunyuan3d.shape-vae-encoder-comparison.v1",
        "checkpoint": checkpoint,
        "oracle": oracle_path,
        "device": format!("{device:?}"),
        "dtype": "F16",
        "points": cfg.pc_size,
        "latents": cfg.num_latents,
        "mean_max_abs": mean_max,
        "mean_rms": mean_rms,
        "logvar_max_abs": logvar_max,
        "logvar_rms": logvar_rms,
        "mean_max_abs_tolerance": MEAN_MAX_ABS_TOLERANCE,
        "logvar_max_abs_tolerance": LOGVAR_MAX_ABS_TOLERANCE,
    });
    fs::write(
        output.join("comparison.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    ensure!(
        mean_max.is_finite() && mean_max <= MEAN_MAX_ABS_TOLERANCE,
        "encoder mean parity failed"
    );
    ensure!(
        logvar_max.is_finite() && logvar_max <= LOGVAR_MAX_ABS_TOLERANCE,
        "encoder logvar parity failed"
    );
    Ok(())
}
