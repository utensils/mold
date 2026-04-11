#![allow(dead_code)]

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use image::{imageops::FilterType, GenericImage, Rgb, RgbImage};
use mold_inference::device::create_device;
use mold_inference::ltx2::media::write_contact_sheet;
use mold_inference::progress::ProgressReporter;

#[path = "../ltx_video/video_enc.rs"]
mod video_enc_impl;
#[path = "../ltx2/model/video_vae.rs"]
mod video_vae_impl;

use video_enc_impl::encode_gif;
use video_vae_impl::{AutoencoderKLLtx2Video, AutoencoderKLLtx2VideoConfig};

fn main() -> Result<()> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    if args.len() < 3 {
        bail!(
            "usage: cargo run -p mold-ai-inference --features cuda,mp4 --bin ltx2_vae_probe -- <checkpoint.safetensors> <input-image> <output.mp4> [width] [height] [frames] [fps]"
        );
    }

    let checkpoint_path = PathBuf::from(&args[0]);
    let input_path = PathBuf::from(&args[1]);
    let output_path = PathBuf::from(&args[2]);
    let width = parse_or_default(args.get(3), 576u32)?;
    let height = parse_or_default(args.get(4), 320u32)?;
    let frames = parse_or_default(args.get(5), 9usize)?;
    let fps = parse_or_default(args.get(6), 12u32)?;

    roundtrip_vae(
        &checkpoint_path,
        &input_path,
        &output_path,
        width,
        height,
        frames,
        fps,
    )
}

fn parse_or_default<T>(value: Option<&String>, default: T) -> Result<T>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    match value {
        Some(value) => value
            .parse::<T>()
            .map_err(|err| anyhow::anyhow!("failed to parse '{value}': {err}")),
        None => Ok(default),
    }
}

fn roundtrip_vae(
    checkpoint_path: &Path,
    input_path: &Path,
    output_path: &Path,
    width: u32,
    height: u32,
    frames: usize,
    fps: u32,
) -> Result<()> {
    if !checkpoint_path.is_file() {
        bail!("checkpoint not found: {}", checkpoint_path.display());
    }
    if !input_path.is_file() {
        bail!("input image not found: {}", input_path.display());
    }
    if output_path.extension().and_then(|ext| ext.to_str()) != Some("mp4") {
        bail!("output must be an .mp4 path: {}", output_path.display());
    }
    if !width.is_multiple_of(32) || !height.is_multiple_of(32) {
        bail!("width and height must be divisible by 32, got {width}x{height}");
    }
    if frames == 0 {
        bail!("frames must be positive");
    }

    let device = create_device(&ProgressReporter::default())?;
    if device.is_metal() {
        bail!("Metal is not supported for native LTX-2 VAE probing");
    }
    let dtype = if device.is_cuda() {
        DType::BF16
    } else {
        DType::F32
    };

    let input_frame = load_and_resize_image(input_path, width, height)?;
    let input_frames = vec![input_frame.clone(); frames];
    let input_tensor = frames_to_video_tensor(&input_frames, &device, dtype)?;

    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(std::slice::from_ref(&checkpoint_path), dtype, &device)?
    };
    let vae_config = infer_vae_config(checkpoint_path);
    let vae = AutoencoderKLLtx2Video::new(vae_config, vb.pp("vae")).with_context(|| {
        format!(
            "failed to load native LTX-2 VAE from '{}'",
            checkpoint_path.display()
        )
    })?;

    println!(
        "device={:?} dtype={:?} mean_of_means_rms={:.6} std_of_means_rms={:.6}",
        device,
        dtype,
        tensor_rms(vae.latents_mean())?,
        tensor_rms(vae.latents_std())?,
    );
    println!(
        "input_video: shape={:?} mean={:.6} rms={:.6}",
        input_tensor.dims(),
        tensor_mean(&input_tensor)?,
        tensor_rms(&input_tensor)?,
    );

    let latents = vae.encode(&input_tensor)?;
    println!(
        "encoded_latents: shape={:?} mean={:.6} rms={:.6}",
        latents.dims(),
        tensor_mean(&latents)?,
        tensor_rms(&latents)?,
    );

    let (_decoder_output, decoded) = vae.decode(&latents, None, false, false)?;
    println!(
        "decoded_video: shape={:?} mean={:.6} rms={:.6}",
        decoded.dims(),
        tensor_mean(&decoded)?,
        tensor_rms(&decoded)?,
    );

    let decoded_frames = tensor_to_frames(&decoded)?;
    let mp4_bytes = encode_mp4_bytes(&decoded_frames, fps)?;
    fs::write(output_path, mp4_bytes)
        .with_context(|| format!("failed to write {}", output_path.display()))?;

    let stem = output_path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .with_context(|| {
            format!(
                "failed to derive output stem from {}",
                output_path.display()
            )
        })?;
    let output_dir = output_path.parent().unwrap_or_else(|| Path::new("."));
    let gif_path = output_dir.join(format!("{stem}.gif"));
    let contact_sheet_path = output_dir.join(format!("{stem}-contact-sheet.png"));
    let input_png_path = output_dir.join(format!("{stem}-input.png"));
    let recon_png_path = output_dir.join(format!("{stem}-recon-first-frame.png"));
    let comparison_png_path = output_dir.join(format!("{stem}-comparison.png"));

    fs::write(&gif_path, encode_gif(&decoded_frames, fps)?)
        .with_context(|| format!("failed to write {}", gif_path.display()))?;
    write_contact_sheet(output_path, &contact_sheet_path)?;
    input_frame
        .save(&input_png_path)
        .with_context(|| format!("failed to write {}", input_png_path.display()))?;
    decoded_frames[0]
        .save(&recon_png_path)
        .with_context(|| format!("failed to write {}", recon_png_path.display()))?;
    build_comparison(&input_frame, &decoded_frames[0])?
        .save(&comparison_png_path)
        .with_context(|| format!("failed to write {}", comparison_png_path.display()))?;

    println!(
        "saved mp4={} gif={} contact_sheet={} input={} recon={} comparison={}",
        output_path.display(),
        gif_path.display(),
        contact_sheet_path.display(),
        input_png_path.display(),
        recon_png_path.display(),
        comparison_png_path.display(),
    );

    Ok(())
}

fn infer_vae_config(checkpoint_path: &Path) -> AutoencoderKLLtx2VideoConfig {
    let checkpoint_name = checkpoint_path.to_string_lossy().to_ascii_lowercase();
    if checkpoint_name.contains("ltx-2.3-22b") {
        AutoencoderKLLtx2VideoConfig::ltx2_22b()
    } else {
        AutoencoderKLLtx2VideoConfig::default()
    }
}

#[cfg(feature = "mp4")]
fn encode_mp4_bytes(frames: &[RgbImage], fps: u32) -> Result<Vec<u8>> {
    video_enc_impl::encode_mp4(frames, fps)
}

#[cfg(not(feature = "mp4"))]
fn encode_mp4_bytes(_frames: &[RgbImage], _fps: u32) -> Result<Vec<u8>> {
    bail!("ltx2_vae_probe requires the `mp4` feature")
}

fn load_and_resize_image(path: &Path, width: u32, height: u32) -> Result<RgbImage> {
    let image = image::open(path)
        .with_context(|| format!("failed to load {}", path.display()))?
        .to_rgb8();
    Ok(image::imageops::resize(
        &image,
        width,
        height,
        FilterType::Lanczos3,
    ))
}

fn frames_to_video_tensor(frames: &[RgbImage], device: &Device, dtype: DType) -> Result<Tensor> {
    let first = frames
        .first()
        .ok_or_else(|| anyhow::anyhow!("no frames for tensor conversion"))?;
    let width = first.width() as usize;
    let height = first.height() as usize;
    let frame_count = frames.len();
    let mut data = Vec::with_capacity(frame_count * width * height * 3);

    for channel in 0..3usize {
        for frame in frames {
            if frame.width() as usize != width || frame.height() as usize != height {
                bail!("all frames must have the same dimensions");
            }
            for pixel in frame.pixels() {
                data.push((pixel[channel] as f32 / 127.5) - 1.0);
            }
        }
    }

    Ok(Tensor::from_vec(data, (1, 3, frame_count, height, width), device)?.to_dtype(dtype)?)
}

fn tensor_to_frames(video: &Tensor) -> Result<Vec<RgbImage>> {
    let video = ((video
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .clamp(-1f32, 1f32)?
        + 1.0)?
        * 127.5)?
        .to_dtype(DType::U8)?;
    let video = video.i(0)?;
    let (_, frames, height, width) = video.dims4()?;
    let mut out = Vec::with_capacity(frames);
    for index in 0..frames {
        let frame = video
            .i((.., index, .., ..))?
            .permute((1, 2, 0))?
            .contiguous()?;
        let data: Vec<u8> = frame.flatten_all()?.to_vec1()?;
        let image = RgbImage::from_raw(width as u32, height as u32, data)
            .ok_or_else(|| anyhow::anyhow!("failed to rebuild RGB frame {index}"))?;
        out.push(image);
    }
    Ok(out)
}

fn build_comparison(input: &RgbImage, recon: &RgbImage) -> Result<RgbImage> {
    if input.dimensions() != recon.dimensions() {
        bail!(
            "comparison images must have the same dimensions, got {:?} and {:?}",
            input.dimensions(),
            recon.dimensions()
        );
    }
    let gutter = 16u32;
    let label_band = 24u32;
    let width = input.width() * 2 + gutter;
    let height = input.height() + label_band;
    let mut canvas = RgbImage::from_pixel(width, height, Rgb([18, 18, 18]));
    canvas.copy_from(input, 0, label_band)?;
    canvas.copy_from(recon, input.width() + gutter, label_band)?;
    Ok(canvas)
}

fn tensor_mean(tensor: &Tensor) -> Result<f32> {
    Ok(tensor
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .mean_all()?
        .to_scalar::<f32>()?)
}

fn tensor_rms(tensor: &Tensor) -> Result<f32> {
    let tensor = tensor.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
    Ok(tensor
        .flatten_all()?
        .sqr()?
        .mean_all()?
        .to_scalar::<f32>()?
        .sqrt())
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::infer_vae_config;

    #[test]
    fn infer_vae_config_selects_22b_layout_from_checkpoint_name() {
        let config = infer_vae_config(Path::new("/tmp/ltx-2.3-22b-distilled-fp8.safetensors"));

        assert_eq!(config.encoder_blocks[0].num_layers, 4);
        assert_eq!(config.encoder_blocks[1].name, "compress_space_res");
        assert_eq!(config.decoder_blocks[1].name, "compress_space");
    }

    #[test]
    fn infer_vae_config_defaults_for_non_22b_checkpoint() {
        let config = infer_vae_config(Path::new("/tmp/ltx-2-19b-distilled-fp8.safetensors"));

        assert_eq!(config.encoder_blocks[1].name, "compress_space_res");
        assert!(!config.encoder_blocks[1].residual);
        assert_eq!(config.decoder_blocks[1].name, "compress_all");
    }
}
