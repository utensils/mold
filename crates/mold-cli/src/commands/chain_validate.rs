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
    let config = mold_core::Config::load_or_default();
    let mut built = super::chain::build_request_from_script(&script)?;
    let authority = super::chain::resolve_chain_model_authority(&built.model, &config);
    let substitution = super::chain::normalize_script_motion_tail(&mut built, &authority);
    let model = built.model.clone();
    let mut req = built.normalise_with_family(authority.family_hint())?;
    // Audio is a resolved default, not an opt-in: a script that never
    // mentions `enable_audio` still renders with sound on an LTX-2 chain, so
    // validation has to report the answer the render will use.
    //
    // `mold chain validate` has no `--local`, so it plans the submission
    // route: the answer belongs to whichever host renders it, and the CLI
    // only resolves it for a run it performs itself. An explicit
    // `enable_audio` in the script still survives.
    req.enable_audio = super::chain::resolve_chain_enable_audio(&req, &config, false);
    if let Some((original, applied)) = substitution {
        println!(
            "note: {model} carries {applied} frame(s) across a seam, not {original}; \
             validated with motion_tail_frames = {applied}"
        );
    }
    println!(
        "OK — {} stages, {} frames estimated, audio {}",
        req.stages.len(),
        req.estimated_total_frames(),
        super::chain::describe_enable_audio(req.enable_audio)
    );
    Ok(())
}
