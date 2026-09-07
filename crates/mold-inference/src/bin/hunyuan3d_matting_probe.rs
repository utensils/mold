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
    anyhow::ensure!(
        args.next().is_none(),
        "usage: probe MODEL.onnx INPUT OUTPUT.png"
    );

    let image = image::open(&input)
        .with_context(|| format!("decode {}", input.display()))?
        .to_rgba8();
    let network = U2Net::load(&model, &Device::Cpu)?;
    let matte = network.matte(&image)?;
    matte
        .save(&output)
        .with_context(|| format!("save {}", output.display()))?;
    Ok(())
}
