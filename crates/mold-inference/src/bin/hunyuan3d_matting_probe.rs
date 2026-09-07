use std::{env, path::PathBuf};

use anyhow::{Context, Result};
use candle_core::Device;
use mold_inference::hunyuan3d::background_matting::U2Net;

fn main() -> Result<()> {
    let mut args = env::args_os().skip(1);
    let model = PathBuf::from(
        args.next()
            .context("usage: probe MODEL.onnx INPUT OUTPUT.png")?,
    );
    let input = PathBuf::from(
        args.next()
            .context("usage: probe MODEL.onnx INPUT OUTPUT.png")?,
    );
    let output = PathBuf::from(
        args.next()
            .context("usage: probe MODEL.onnx INPUT OUTPUT.png")?,
    );
    let backend = args.next();
    anyhow::ensure!(
        args.next().is_none(),
        "usage: probe MODEL.onnx INPUT OUTPUT.png [cuda]"
    );

    let image = image::open(&input)
        .with_context(|| format!("decode {}", input.display()))?
        .to_rgba8();
    let device = match backend.as_deref().and_then(|value| value.to_str()) {
        None | Some("cpu") => Device::Cpu,
        #[cfg(feature = "cuda")]
        Some("cuda") => Device::new_cuda(0)?,
        Some(value) => anyhow::bail!("unknown backend {value}; expected cpu or cuda"),
    };
    let network = U2Net::load(&model, &device)?;
    let matte = network.matte(&image)?;
    matte
        .save(&output)
        .with_context(|| format!("save {}", output.display()))?;
    Ok(())
}
