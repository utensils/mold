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
    let req = super::chain::build_request_from_script(&script)?.normalise()?;
    println!(
        "OK — {} stages, {} frames estimated",
        req.stages.len(),
        req.estimated_total_frames()
    );
    Ok(())
}
