use std::{env, fs, path::PathBuf, time::Instant};

use anyhow::{ensure, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use mold_inference::hunyuan3d::{
    glb::{read_glb, write_glb, GlbMaterial},
    mesh::{self, MeshAlgorithm, OccupancyGrid},
    shape_vae::{
        query_grid_chunk, query_grid_len, ShapeVae, ShapeVaeConfig, ShapeVaeEncoder,
        ShapeVaeEncoderConfig,
    },
};
use serde_json::json;

const QUERY_BOUNDS: f32 = 1.01;
const ENCODER_QUERY_CHUNK: usize = 64;
const DECODER_QUERY_CHUNK: usize = 8_000;
// Tencent's minimal VAE round-trip uses raw `mc_level=0.0`. Mold's mesher
// receives `(logit + 1) / 2`, so the equivalent level is exactly 0.5.
const THRESHOLD: f32 = 0.5;

fn main() -> Result<()> {
    let mut args = env::args_os().skip(1);
    let checkpoint = PathBuf::from(
        args.next()
            .context("usage: roundtrip CHECKPOINT INPUT.glb OUTPUT_DIR [RESOLUTION]")?,
    );
    let input = PathBuf::from(
        args.next()
            .context("usage: roundtrip CHECKPOINT INPUT.glb OUTPUT_DIR [RESOLUTION]")?,
    );
    let output = PathBuf::from(
        args.next()
            .context("usage: roundtrip CHECKPOINT INPUT.glb OUTPUT_DIR [RESOLUTION]")?,
    );
    let resolution = args
        .next()
        .map(|value| value.to_string_lossy().parse::<usize>())
        .transpose()
        .context("resolution must be an integer")?
        .unwrap_or(128);
    ensure!(
        args.next().is_none(),
        "usage: roundtrip CHECKPOINT INPUT.glb OUTPUT_DIR [RESOLUTION]"
    );
    ensure!(checkpoint.is_file(), "checkpoint does not exist");
    ensure!(input.is_file(), "input GLB does not exist");
    ensure!(resolution >= 32, "resolution must be at least 32");
    fs::create_dir(&output).with_context(|| format!("create {}", output.display()))?;

    let source = read_glb(&fs::read(&input)?).context("read input GLB")?;
    let device = Device::new_cuda(0)?;
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(std::slice::from_ref(&checkpoint), DType::F16, &device)?
    };
    let encoder = ShapeVaeEncoder::new(&ShapeVaeEncoderConfig::v2_1(), vb.pp("vae"))?;
    let decoder = ShapeVae::new(&ShapeVaeConfig::v2_1(), vb.pp("vae"))?;

    let encode_started = Instant::now();
    let encoded = encoder.encode_mesh_sampled(&source, 130_013, ENCODER_QUERY_CHUNK)?;
    let encode_seconds = encode_started.elapsed().as_secs_f64();
    let latent_path = output.join("latents.safetensors");
    candle_core::safetensors::save(
        &std::collections::HashMap::from([
            ("mean".to_owned(), encoded.mean.to_device(&Device::Cpu)?),
            ("logvar".to_owned(), encoded.logvar.to_device(&Device::Cpu)?),
        ]),
        &latent_path,
    )?;

    let decode_started = Instant::now();
    let prepared = decoder.prepare_latents(&encoded.for_decoder()?)?;
    let cross_kv = decoder.prepare_cross_kv(&prepared)?;
    drop((prepared, encoded));
    let total = query_grid_len(resolution);
    let mut logits = Vec::with_capacity(total);
    for start in (0..total).step_by(DECODER_QUERY_CHUNK) {
        let len = DECODER_QUERY_CHUNK.min(total - start);
        let queries = query_grid_chunk(resolution, QUERY_BOUNDS, start, len, &device, DType::F16)?
            .unsqueeze(0)?;
        let chunk = decoder.decode_queries_cached(&queries, &cross_kv)?;
        logits.extend(
            chunk
                .flatten_all()?
                .to_dtype(DType::F32)?
                .to_vec1::<f32>()?,
        );
    }
    let dim = resolution + 1;
    let flat = Tensor::from_vec(logits, (1, total), &Device::Cpu)?;
    let mut ordered = ShapeVae::reshape_grid_logits(&flat, resolution)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    for value in &mut ordered {
        *value = ((*value + 1.0) * 0.5).clamp(0.0, 1.0);
    }
    let grid = OccupancyGrid::new(ordered, [dim, dim, dim])?;
    let mut roundtrip = mesh::extract(&grid, MeshAlgorithm::SurfaceNet, THRESHOLD, &mut |_, _| {
        Ok(())
    })?;
    ensure!(!roundtrip.is_empty(), "round trip produced an empty mesh");
    mesh::compute_smooth_normals(&mut roundtrip);
    let decode_seconds = decode_started.elapsed().as_secs_f64();

    let glb_path = output.join("roundtrip.glb");
    fs::write(
        &glb_path,
        write_glb(&roundtrip, &GlbMaterial::default(), None)?,
    )?;
    let report = json!({
        "schema": "mold.hunyuan3d.shape-vae-roundtrip.v1",
        "checkpoint": checkpoint,
        "input": input,
        "device": format!("{device:?}"),
        "dtype": "F16",
        "seed": 130013,
        "posterior": "sampled_deterministically",
        "resolution": resolution,
        "threshold": THRESHOLD,
        "encoder_query_chunk": ENCODER_QUERY_CHUNK,
        "decoder_query_chunk": DECODER_QUERY_CHUNK,
        "source_vertices": source.vertices.len(),
        "source_faces": source.faces.len(),
        "roundtrip_vertices": roundtrip.vertices.len(),
        "roundtrip_faces": roundtrip.faces.len(),
        "encode_seconds": encode_seconds,
        "decode_seconds": decode_seconds,
        "latent_path": latent_path,
        "glb_path": glb_path,
    });
    fs::write(output.join("run.json"), serde_json::to_vec_pretty(&report)?)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
