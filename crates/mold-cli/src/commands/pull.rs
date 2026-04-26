use anyhow::Result;
use colored::Colorize;
use mold_core::config::Config;
use mold_core::download::DownloadError;
use mold_core::manifest::{find_manifest, resolve_model_name, ModelManifest};
use mold_core::{classify_server_error, ServerAvailability};

use crate::control::CliContext;
use crate::output::status;
use crate::theme;
use crate::ui::print_server_fallback;
use crate::AlreadyReported;

/// Download a model and write its config. Returns the updated Config.
pub async fn pull_and_configure(
    model: &str,
    opts: &mold_core::download::PullOptions,
) -> Result<Config> {
    let canonical = resolve_model_name(model);

    // Pre-flight: print status and validate manifest exists (for CLI-specific error formatting)
    let manifest = match find_manifest(&canonical) {
        Some(m) => m,
        None => {
            print_unknown_model_error(model);
            return Err(AlreadyReported.into());
        }
    };

    let (total_bytes, remaining_bytes) = mold_core::manifest::compute_download_size(manifest);
    let total_gb = total_bytes as f64 / 1_073_741_824.0;
    let remaining_gb = remaining_bytes as f64 / 1_073_741_824.0;
    let cached_gb = total_gb - remaining_gb;
    if cached_gb > 0.1 {
        status!(
            "{} Pulling {} ({:.1}GB to download, {:.1}GB already cached)",
            theme::icon_info(),
            manifest.name.bold(),
            remaining_gb,
            cached_gb,
        );
    } else {
        status!(
            "{} Pulling {} ({:.1}GB to download)",
            theme::icon_info(),
            manifest.name.bold(),
            total_gb,
        );
    }
    status!(
        "  {}",
        crate::output::colorize_description(&manifest.description)
    );
    status!("");

    // Delegate to core pull_and_configure
    let (config, _paths) = mold_core::download::pull_and_configure(model, opts)
        .await
        .map_err(|e| -> anyhow::Error {
            match e {
                DownloadError::UnknownModel { .. } => {
                    print_unknown_model_error(model);
                }
                DownloadError::Unauthorized { repo, .. } => {
                    eprintln!();
                    eprintln!("{} Authentication required for {repo}", theme::icon_fail());
                    eprintln!();
                    eprintln!("  1. Create a token at: https://huggingface.co/settings/tokens");
                    eprintln!("     (select at least \"Read\" access)");
                    eprintln!("  2. Set: export HF_TOKEN=hf_...");
                    eprintln!("     Or run: huggingface-cli login");
                    eprintln!("  3. Retry: mold pull {}", canonical);
                    if std::env::var("HF_TOKEN").is_ok() {
                        eprintln!();
                        eprintln!(
                            "  {} HF_TOKEN is set but was rejected — it may be invalid or expired.",
                            theme::icon_alert()
                        );
                    }
                }
                DownloadError::GatedModel { .. } => {
                    eprintln!();
                    eprintln!(
                        "{} This model requires access approval on HuggingFace.",
                        theme::icon_fail()
                    );
                    eprintln!();

                    let gated_repo = manifest
                        .files
                        .iter()
                        .find(|f| f.gated)
                        .map(|f| f.hf_repo.as_str())
                        .unwrap_or("the model repository");

                    eprintln!("  1. Visit: https://huggingface.co/{gated_repo}");
                    eprintln!("  2. Accept the license agreement");
                    eprintln!("  3. Create a token at: https://huggingface.co/settings/tokens");
                    eprintln!("  4. Set: export HF_TOKEN=hf_...");
                    eprintln!("  5. Retry: mold pull {}", canonical);
                }
                DownloadError::Sha256Mismatch {
                    filename,
                    expected,
                    actual,
                    ..
                } => {
                    eprintln!();
                    eprintln!(
                        "{} SHA-256 mismatch for {}",
                        theme::icon_fail(),
                        filename.bold()
                    );
                    eprintln!("  Expected: {expected}");
                    eprintln!("  Got:      {actual}");
                    eprintln!();
                    eprintln!("The corrupted file has been removed.");
                    eprintln!("  Re-run: mold pull {}", canonical);
                    eprintln!();
                    eprintln!("If the file was intentionally updated on HuggingFace, use:");
                    eprintln!("  mold pull {} --skip-verify", canonical);
                }
                other => {
                    eprintln!();
                    eprintln!("{} Download failed: {other}", theme::icon_fail());
                }
            }
            AlreadyReported.into()
        })?;

    status!("");
    status!("{} {} is ready!", theme::icon_done(), canonical.bold());

    Ok(config)
}

fn print_unknown_model_error(model: &str) {
    eprintln!("{} Unknown model: {}", theme::icon_fail(), model.bold());
    eprintln!();
    eprintln!("Available models:");
    let visible: Vec<_> = mold_core::manifest::visible_manifests().collect();
    let nw = visible.iter().map(|m| m.name.len()).max().unwrap_or(4) + 2;
    for m in &visible {
        let total_bytes = mold_core::manifest::total_download_size(m);
        let total_gb = total_bytes as f64 / 1_073_741_824.0;
        eprintln!(
            "  {:<nw$} {:>5.1}GB  {}",
            m.name.bold(),
            total_gb,
            crate::output::colorize_description(&m.description),
            nw = nw,
        );
    }
    eprintln!();
    eprintln!("Usage: mold pull <model>");
}

pub async fn run(model: &str, opts: &mold_core::download::PullOptions) -> Result<()> {
    let canonical = resolve_model_name(model);
    let manifest = match find_manifest(&canonical) {
        Some(m) => m,
        None => {
            print_unknown_model_error(model);
            return Err(AlreadyReported.into());
        }
    };

    let ctx = CliContext::new(None);
    match pull_via_server(&ctx, manifest).await {
        Ok(()) => {}
        Err(e) => match classify_server_error(&e) {
            ServerAvailability::FallbackLocal => {
                print_server_fallback(ctx.client().host(), "pulling locally");
                pull_and_configure(model, opts).await?;
            }
            ServerAvailability::SurfaceError => return Err(e),
        },
    }

    status!("  mold run \"your prompt\"");
    Ok(())
}

/// Run a recipe-driven pull for a Civitai catalog row. Pulls each missing
/// canonical companion FIRST (so the SDXL/SD1.5 engine has clip-l, clip-g,
/// vae before it tries to load the primary), then fetches the recipe's
/// files into `MOLD_MODELS_DIR/<sanitized-id>/`.
///
/// Mirrors the manifest path's lifecycle (status prints, marker, sha-verify)
/// but doesn't try to upgrade through the manifest registry — the catalog
/// id is the canonical identifier for this download.
pub async fn run_recipe(
    row: mold_db::catalog::CatalogRow,
    opts: &mold_core::download::PullOptions,
) -> Result<()> {
    use mold_core::download::{
        civitai_auth_or_error, fetch_recipe, missing_companions_from_json, DownloadError,
        RecipeAuth, RecipeFetchFile,
    };

    let recipe: mold_catalog::entry::DownloadRecipe = serde_json::from_str(&row.download_recipe)
        .map_err(|e| {
            anyhow::anyhow!("catalog row {} has malformed download_recipe: {e}", row.id)
        })?;

    // Resolve auth before printing anything so a missing token surfaces
    // an actionable error instead of "starting download...". The
    // mold-core helper already crafts a remediation message naming the env var.
    let auth = match recipe.needs_token {
        Some(mold_catalog::entry::TokenKind::Civitai) => match civitai_auth_or_error(&row.id) {
            Ok(a) => a,
            Err(e) => {
                eprintln!();
                eprintln!("{} {e}", theme::icon_fail());
                return Err(AlreadyReported.into());
            }
        },
        _ => RecipeAuth::None,
    };

    let total_recipe_bytes: u64 = recipe.files.iter().filter_map(|f| f.size_bytes).sum();
    let total_gb = total_recipe_bytes as f64 / 1_073_741_824.0;
    status!(
        "{} Pulling {} ({:.1}GB to download)",
        theme::icon_info(),
        row.id.bold(),
        total_gb,
    );
    if let Some(desc) = row.description.as_deref() {
        if !desc.is_empty() {
            status!("  {}", crate::output::colorize_description(desc));
        }
    }
    status!("");

    // Companion-first ordering: find every canonical companion the
    // catalog row declares that isn't already on disk, then pull each
    // through the manifest path. mold-core de-dupes against in-flight
    // pulls of the same name, so concurrent requests won't double-pull.
    let models_dir = mold_core::Config::load_or_default().resolved_models_dir();
    let companions = missing_companions_from_json(row.companions.as_deref(), &models_dir);
    if !companions.is_empty() {
        status!(
            "{} {} companion file(s) needed before primary",
            theme::icon_info(),
            companions.len(),
        );
        for manifest in companions {
            pull_and_configure(&manifest.name, opts).await?;
        }
        status!("");
    }

    // Primary: fetch the recipe files. Translate from the catalog's owned
    // strings into the borrowed slice mold-core expects.
    let fetch_files: Vec<RecipeFetchFile<'_>> = recipe
        .files
        .iter()
        .map(|f| RecipeFetchFile {
            url: f.url.as_str(),
            dest: f.dest.as_str(),
            sha256: f.sha256.as_deref(),
            size_bytes: f.size_bytes,
        })
        .collect();

    fetch_recipe(&row.id, &fetch_files, auth, &models_dir, None, opts)
        .await
        .map_err(|e| -> anyhow::Error {
            match e {
                DownloadError::MissingCivitaiToken { .. } => {
                    eprintln!();
                    eprintln!("{} {e}", theme::icon_fail());
                }
                DownloadError::Sha256Mismatch {
                    filename,
                    expected,
                    actual,
                    ..
                } => {
                    eprintln!();
                    eprintln!(
                        "{} SHA-256 mismatch for {}",
                        theme::icon_fail(),
                        filename.bold()
                    );
                    eprintln!("  Expected: {expected}");
                    eprintln!("  Got:      {actual}");
                    eprintln!();
                    eprintln!(
                        "The corrupted file has been removed. Re-run: mold pull {}",
                        row.id
                    );
                }
                DownloadError::RecipeHttp { url, status, body } => {
                    eprintln!();
                    eprintln!(
                        "{} HTTP {status} for {}{}",
                        theme::icon_fail(),
                        url,
                        body.as_deref()
                            .map(|b| format!(" — {b}"))
                            .unwrap_or_default(),
                    );
                }
                other => {
                    eprintln!();
                    eprintln!("{} Download failed: {other}", theme::icon_fail());
                }
            }
            AlreadyReported.into()
        })?;

    status!("");
    status!("{} {} is ready!", theme::icon_done(), row.id.bold());
    status!("  mold run {} \"your prompt\"", row.id);
    Ok(())
}

async fn pull_via_server(ctx: &CliContext, manifest: &ModelManifest) -> Result<()> {
    status!(
        "{} Pulling {} on {}",
        theme::icon_info(),
        manifest.name.bold(),
        ctx.client().host().bold(),
    );
    status!(
        "  {}",
        crate::output::colorize_description(&manifest.description)
    );
    status!("");

    ctx.stream_server_pull(&manifest.name).await?;

    status!("");
    status!(
        "{} {} is ready on {}!",
        theme::icon_done(),
        manifest.name.bold(),
        ctx.client().host().bold(),
    );
    Ok(())
}
