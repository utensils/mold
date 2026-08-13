//! `mold chain validate <path>` — parse and normalise a TOML chain script
//! without submitting it. Used to gate TOML authored by hand or by the
//! movie-maker UI before a render.

use std::path::Path;

use anyhow::Result;

pub async fn run(path: &Path) -> Result<()> {
    let toml_src = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("failed to read script {}: {e}", path.display()))?;
    let script_dir = path.parent().unwrap_or_else(|| Path::new("."));
    let script = mold_core::chain_toml::read_script_resolving_paths(&toml_src, script_dir)
        .map_err(|e| anyhow::anyhow!("invalid chain TOML in {}: {e}", path.display()))?;
    // The seam a family can honour is resolved before `normalise()`, so this
    // reports the chain that will actually render rather than the one the
    // TOML asked for. Wan carries one frame across a seam, or none on a
    // text-to-video checkpoint — never LTX-2's 17, which sits on wan's own
    // `4k+1` grid and so would otherwise validate clean (#783).
    let mut built = super::chain::build_request_from_script(&script)?;
    let substitution = super::chain::normalize_script_motion_tail(&mut built);
    let model = built.model.clone();
    let req = built.normalise()?;
    if let Some((original, applied)) = substitution {
        println!(
            "note: {model} carries {applied} frame(s) across a seam, not {original}; \
             validated with motion_tail_frames = {applied}"
        );
    }
    println!(
        "OK — {} stages, {} frames estimated",
        req.stages.len(),
        req.estimated_total_frames()
    );
    Ok(())
}
