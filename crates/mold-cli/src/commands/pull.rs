use anyhow::{Result, bail};
use colored::Colorize;

pub async fn run(model: &str) -> Result<()> {
    eprintln!("{} `mold pull` is not yet implemented.", "✗".red().bold());
    eprintln!();
    eprintln!(
        "To use {} manually, place the GGUF/safetensors files in your models directory",
        model.bold()
    );
    eprintln!("and configure paths in ~/.mold/config.toml or via environment variables:");
    eprintln!();
    eprintln!("  MOLD_TRANSFORMER_PATH=/path/to/transformer.gguf");
    eprintln!("  MOLD_VAE_PATH=/path/to/ae.safetensors");
    eprintln!("  MOLD_T5_PATH=/path/to/t5xxl_fp16.safetensors");
    eprintln!("  MOLD_CLIP_PATH=/path/to/clip_l.safetensors");
    eprintln!();
    eprintln!("Model files for {} are available at:", model.bold());
    match model {
        "flux-schnell" => eprintln!("  https://huggingface.co/black-forest-labs/FLUX.1-schnell"),
        "flux-dev" => eprintln!("  https://huggingface.co/black-forest-labs/FLUX.1-dev"),
        _ => eprintln!("  https://huggingface.co/black-forest-labs"),
    }
    bail!("pull not implemented");
}
