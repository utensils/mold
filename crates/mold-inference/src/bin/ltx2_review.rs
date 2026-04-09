use std::env;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use mold_inference::ltx2::media::{extract_gif_preview, probe_video, write_contact_sheet};

fn main() -> Result<()> {
    let inputs = env::args().skip(1).collect::<Vec<_>>();
    if inputs.is_empty() {
        bail!("usage: cargo run -p mold-ai-inference --features mp4 --bin ltx2_review -- <video.mp4> [more.mp4...]");
    }

    for input in inputs {
        let input = PathBuf::from(input);
        review_video(&input)?;
    }

    Ok(())
}

fn review_video(input: &Path) -> Result<()> {
    if !input.exists() {
        bail!("input video not found: {}", input.display());
    }
    if input.extension().and_then(|ext| ext.to_str()) != Some("mp4") {
        bail!("input must be an .mp4 file: {}", input.display());
    }

    let stem = input
        .file_stem()
        .and_then(|stem| stem.to_str())
        .with_context(|| format!("failed to derive output stem from {}", input.display()))?;
    let gif_path = input.with_file_name(format!("{stem}.gif"));
    let contact_sheet_path = input.with_file_name(format!("{stem}-contact-sheet.png"));
    let metadata = probe_video(input)?;

    extract_gif_preview(input, &gif_path)?;
    write_contact_sheet(input, &contact_sheet_path)?;

    println!(
        "{} -> {} frames, {}x{}, {} fps, audio={}, gif={}, contact_sheet={}",
        input.display(),
        metadata.frames.unwrap_or(0),
        metadata.width,
        metadata.height,
        metadata.fps,
        metadata.has_audio,
        gif_path.display(),
        contact_sheet_path.display()
    );

    Ok(())
}
