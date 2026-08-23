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

fn rendered_primary_dest<'a>(
    recipe: &mold_catalog::entry::DownloadRecipe,
    rendered_dests: &'a [String],
) -> Option<&'a str> {
    recipe
        .files
        .iter()
        .zip(rendered_dests)
        .find(|(file, _)| file.role.is_none())
        .or_else(|| recipe.files.iter().zip(rendered_dests).next())
        .map(|(_, dest)| dest.as_str())
}

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
                // Not a download failure — nothing was attempted. The error
                // already carries the terms, the URL, and the exact command
                // that records acceptance, so print it as its own refusal.
                DownloadError::LicenseNotAccepted { ref message, .. } => {
                    eprintln!();
                    eprintln!("{} {message}", theme::icon_fail());
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

/// A license the user was shown, and the exact payload that must be sent as a
/// result.
pub struct AcceptedLicense {
    /// Precisely the identity that was displayed. Sending anything else would
    /// mean the recorded consent is not the consent that was given.
    pub acceptance: mold_core::LicenseAcceptance,
    /// This build's pinned license, when the displayed terms were ours.
    /// `None` when a server showed terms from a different Mold release — in
    /// which case we must not record locally, because we would be storing our
    /// own text under an agreement made to theirs.
    pub local: Option<&'static mold_core::license_acceptance::ThirdPartyLicense>,
}

fn print_known_licenses_error(id: &str, known: impl Iterator<Item = (String, String)>) {
    eprintln!(
        "{} unknown license id '{}'",
        theme::prefix_error(),
        id.bold()
    );
    eprintln!();
    eprintln!("  Known licenses:");
    for (license_id, name) in known {
        eprintln!("    {} — {}", license_id.bold(), name);
    }
}

fn show(name: &str, summary: &str, url: &str, canonical: &str) {
    status!("{} {}", theme::icon_info(), name.bold());
    status!("  {}", summary);
    status!("  Terms (pinned): {}", url);
    status!("  Project terms:  {}", canonical);
    status!("");
}

/// Read the terms from the SERVER that will record the acceptance.
///
/// Returns `Ok(None)` only when there is genuinely no such server to ask —
/// a connection failure, or an older build without `capabilities.licenses` —
/// in which case the pull will land locally (or on a server with no license
/// gate at all) and this build's own constants are the right thing to display.
/// A server that answered but could not be read (auth, 5xx, malformed body)
/// is an error naming the host: showing this build's terms in its place
/// would prompt the user to consent to text the recording machine never saw.
async fn server_license_terms(
    ctx: &CliContext,
) -> Result<Option<Vec<mold_core::ThirdPartyLicenseStatus>>> {
    let host = ctx.client().host().to_string();
    let capabilities = match ctx.client().capabilities().await {
        Ok(capabilities) => capabilities,
        Err(error) => return terms_fetch_failure(&error, &host, "capabilities"),
    };
    if !capabilities.licenses {
        return Ok(None);
    }
    match ctx.client().list_licenses().await {
        Ok(listing) => Ok(Some(listing)),
        Err(error) => terms_fetch_failure(&error, &host, "licenses"),
    }
}

/// Connection failures mean "no server to consult"; everything else is a
/// server that answered and must not be silently replaced by local terms.
fn terms_fetch_failure<T>(error: &anyhow::Error, host: &str, what: &str) -> Result<Option<T>> {
    match mold_core::classify_server_error(error) {
        mold_core::ServerAvailability::FallbackLocal => Ok(None),
        mold_core::ServerAvailability::SurfaceError => Err(anyhow::anyhow!(
            "{host} did not report its {what}, so its license terms cannot be shown for acceptance: {error}"
        )),
    }
}

/// Resolve a `--accept-license` id and SHOW the user what they are accepting.
///
/// The terms displayed are the ones held by the machine that will RECORD the
/// acceptance, and the payload sent back is byte-for-byte what was displayed.
/// Showing our own pinned revision and then letting a server on a different
/// release resolve the bare id to its own would record consent for text the
/// user never read.
///
/// Still offline in the local case: the terms are pinned to an exact upstream
/// commit whose digest was verified when the pin landed, so accepting works on
/// an air-gapped host.
pub fn resolve_and_show_license(
    id: &str,
    server_terms: Option<&[mold_core::ThirdPartyLicenseStatus]>,
) -> Result<AcceptedLicense> {
    use mold_core::license_acceptance;

    if let Some(terms) = server_terms {
        let Some(status) = terms.iter().find(|entry| entry.id == id) else {
            print_known_licenses_error(
                id,
                terms
                    .iter()
                    .map(|entry| (entry.id.clone(), entry.name.clone())),
            );
            return Err(AlreadyReported.into());
        };
        show(
            &status.name,
            &status.summary,
            &status.url,
            &status.canonical,
        );
        let local = license_acceptance::license_by_id(id)
            .filter(|ours| ours.url == status.url && ours.sha256 == status.sha256);
        return Ok(AcceptedLicense {
            acceptance: mold_core::LicenseAcceptance {
                id: status.id.clone(),
                url: status.url.clone(),
                sha256: status.sha256.clone(),
            },
            local,
        });
    }

    let Some(license) = license_acceptance::license_by_id(id) else {
        print_known_licenses_error(
            id,
            license_acceptance::THIRD_PARTY_LICENSES
                .iter()
                .map(|known| (known.id.to_string(), known.name.to_string())),
        );
        return Err(AlreadyReported.into());
    };
    show(
        license.name,
        license.summary,
        license.url,
        license.canonical,
    );
    Ok(AcceptedLicense {
        acceptance: license_acceptance::acceptance_for(license),
        local: Some(license),
    })
}

/// Write the acceptance into THIS machine's Mold data root.
///
/// Only correct when the pull itself runs locally. A pull dispatched to a
/// server must instead send the id in `accept_licenses` so the SERVER records
/// it in its own root — recording here and pulling there is exactly the bug
/// that made the documented `--accept-license` command fail against a remote
/// `MOLD_HOST`.
pub fn record_license_locally(accepted: &AcceptedLicense) -> Result<()> {
    // The displayed terms came from a server on a different release, and the
    // pull has since fallen back to this machine. Recording our own text under
    // an agreement made to theirs would be exactly the substitution this whole
    // payload shape exists to prevent.
    let Some(license) = accepted.local else {
        eprintln!(
            "{} the terms shown for '{}' came from the server, but the pull fell back to this machine.",
            theme::prefix_error(),
            accepted.acceptance.id.bold()
        );
        eprintln!("  Re-run the command to review and accept this machine's terms.");
        return Err(AlreadyReported.into());
    };
    let Some(mold_home) = Config::mold_dir() else {
        eprintln!(
            "{} could not resolve the Mold data directory to record acceptance",
            theme::prefix_error()
        );
        return Err(AlreadyReported.into());
    };

    mold_core::license_acceptance::record_acceptance(&mold_home, license).map_err(|error| {
        eprintln!(
            "{} failed to record license acceptance: {error}",
            theme::prefix_error()
        );
        anyhow::Error::from(AlreadyReported)
    })?;

    status!(
        "{} recorded acceptance of {} on this machine",
        theme::icon_done(),
        license.id.bold()
    );
    status!("");
    Ok(())
}

pub async fn run(
    model: &str,
    opts: &mold_core::download::PullOptions,
    accept_licenses: &[String],
) -> Result<()> {
    let canonical = resolve_model_name(model);
    let manifest = match find_manifest(&canonical) {
        Some(m) => m,
        None => {
            print_unknown_model_error(model);
            return Err(AlreadyReported.into());
        }
    };

    let ctx = CliContext::new(None);

    // Ask the machine that will RECORD the acceptance for its terms before
    // displaying anything — only skipped entirely when nothing is being
    // accepted, so an ordinary pull pays no extra round trip.
    let server_terms = if accept_licenses.is_empty() {
        None
    } else {
        server_license_terms(&ctx).await?
    };

    // Resolve and display before anything is dispatched: an unknown id must
    // fail here rather than after a download has been enqueued somewhere.
    let mut licenses = Vec::with_capacity(accept_licenses.len());
    for id in accept_licenses {
        licenses.push(resolve_and_show_license(id, server_terms.as_deref())?);
    }
    let payload: Vec<mold_core::LicenseAcceptance> = licenses
        .iter()
        .map(|accepted| accepted.acceptance.clone())
        .collect();

    // The server path carries the accepted terms on the wire so the SERVER
    // records them in its own root; the local path records them here. Which
    // machine runs the pull decides which machine's acceptance counts.
    match pull_via_server(&ctx, manifest, &payload).await {
        Ok(()) => {}
        Err(e) => match classify_server_error(&e) {
            ServerAvailability::FallbackLocal => {
                print_server_fallback(ctx.client().host(), "pulling locally");
                for license in &licenses {
                    record_license_locally(license)?;
                }
                pull_and_configure(model, opts).await?;
            }
            ServerAvailability::SurfaceError => return Err(e),
        },
    }

    // A download-only model (no engine arm in this build) must not be handed
    // a `mold run` hint it will refuse; say what it is instead.
    match mold_core::require_model_activation(&manifest.name, Some(&manifest.family)) {
        Ok(()) => status!("  mold run \"your prompt\""),
        Err(error) => status!("  Downloaded and verified. {error}"),
    }
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
    entry: mold_catalog::entry::CatalogEntry,
    opts: &mold_core::download::PullOptions,
) -> Result<()> {
    use mold_core::download::{
        civitai_auth_or_error, fetch_recipe, missing_companions, DownloadError, RecipeAuth,
        RecipeFetchFile,
    };

    let id_str = entry.id.as_str().to_string();
    let recipe = &entry.download_recipe;

    // Resolve auth before printing anything so a missing token surfaces
    // an actionable error instead of "starting download...". The
    // mold-core helper already crafts a remediation message naming the env var.
    let auth = match recipe.needs_token {
        Some(mold_catalog::entry::TokenKind::Civitai) => match civitai_auth_or_error(&id_str) {
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
        id_str.bold(),
        total_gb,
    );
    if let Some(desc) = entry.description.as_deref() {
        if !desc.is_empty() {
            status!("  {}", crate::output::colorize_description(desc));
        }
    }
    status!("");

    // Companion-first ordering: find every canonical companion the
    // catalog entry declares that isn't already on disk, then pull each
    // through the manifest path. mold-core de-dupes against in-flight
    // pulls of the same name, so concurrent requests won't double-pull.
    let models_dir = mold_core::Config::load_or_default().resolved_models_dir();
    let companions = missing_companions(&entry.companions, &models_dir);
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

    // Primary: fetch the recipe files. Templates ship literal `{family}`,
    // `{author}`, `{name}` placeholders; render them now so the file lands
    // under e.g. `models/sdxl/civitai/<id>/...` instead of the literal
    // `models/{family}/civitai/<id>/...` that the runner can never find.
    let (author, name) = match entry.source_id.split_once('/') {
        Some((a, n)) => (a, n),
        None => ("", entry.source_id.as_str()),
    };
    let rendered_dests: Vec<String> = recipe
        .files
        .iter()
        .map(|f| {
            mold_catalog::entry::render_recipe_dest(&f.dest, entry.family.as_str(), author, name)
        })
        .collect();
    let fetch_files: Vec<RecipeFetchFile<'_>> = recipe
        .files
        .iter()
        .zip(rendered_dests.iter())
        .map(|(f, dest)| RecipeFetchFile {
            url: f.url.as_str(),
            dest: dest.as_str(),
            sha256: f.sha256.as_deref(),
            size_bytes: f.size_bytes,
        })
        .collect();

    fetch_recipe(&id_str, &fetch_files, auth, &models_dir, None, opts)
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
                    eprintln!("The corrupted file has been removed. Re-run: mold pull {id_str}");
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

    // Refresh the sidecar even when the primary was already present. Besides
    // keeping live metadata current, this upgrades legacy sidecars from the
    // removed numeric compatibility field to the direct `supported` boolean.
    if id_str.starts_with("cv:") {
        let primary_dest = rendered_primary_dest(recipe, &rendered_dests)
            .ok_or_else(|| anyhow::anyhow!("catalog recipe {id_str} has no primary file"))?;
        let sidecar = mold_catalog::sidecar::sidecar_from_entry(&entry, primary_dest.to_string());
        let sidecar_path =
            mold_catalog::sidecar::civitai_sidecar_path(&models_dir, entry.id.as_str());
        mold_catalog::sidecar::write_sidecar(&sidecar_path, &sidecar).map_err(|error| {
            anyhow::anyhow!(
                "write catalog sidecar {} after pull: {error}",
                sidecar_path.display()
            )
        })?;
    }

    status!("");
    status!("{} {} is ready!", theme::icon_done(), id_str.bold());
    status!("  mold run {id_str} \"your prompt\"");
    Ok(())
}

async fn pull_via_server(
    ctx: &CliContext,
    manifest: &ModelManifest,
    accept_licenses: &[mold_core::LicenseAcceptance],
) -> Result<()> {
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

    ctx.stream_server_pull_accepting(&manifest.name, accept_licenses)
        .await?;

    status!("");
    status!(
        "{} {} is ready on {}!",
        theme::icon_done(),
        manifest.name.bold(),
        ctx.client().host().bold(),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_server_that_answered_but_could_not_be_read_is_an_error_naming_the_host() {
        let error = anyhow::anyhow!("401 Unauthorized");
        let outcome = super::terms_fetch_failure::<()>(&error, "http://box:7680", "licenses");
        let message = outcome
            .expect_err("a non-connection failure must surface")
            .to_string();
        assert!(message.contains("http://box:7680"), "{message}");
        assert!(message.contains("licenses"), "{message}");
        assert!(message.contains("401 Unauthorized"), "{message}");
    }

    use super::*;
    use mold_catalog::entry::{DownloadRecipe, RecipeFile, RecipeFileRole};

    fn server_status(url: &str, sha256: &str) -> mold_core::ThirdPartyLicenseStatus {
        mold_core::ThirdPartyLicenseStatus {
            id: "insightface-antelopev2".into(),
            name: "Server's name for it".into(),
            url: url.into(),
            canonical: "https://example.invalid/terms".into(),
            sha256: sha256.into(),
            summary: "Server's summary".into(),
            accepted: false,
            required_by: vec!["pulid-flux".into()],
        }
    }

    /// The consent-integrity rule at the client end: whatever identity was
    /// displayed is exactly what gets sent. A server on a different release
    /// pins a different revision, we show THAT, and we send THAT — never our
    /// own constant under the server's summary.
    #[test]
    fn the_payload_is_exactly_the_displayed_identity() {
        let their_url = "https://raw.githubusercontent.com/deepinsight/insightface/1111111111111111111111111111111111111111/README.md";
        let their_sha = "1".repeat(64);
        let terms = vec![server_status(their_url, &their_sha)];

        let accepted = resolve_and_show_license("insightface-antelopev2", Some(&terms)).unwrap();
        assert_eq!(accepted.acceptance.id, "insightface-antelopev2");
        assert_eq!(accepted.acceptance.url, their_url);
        assert_eq!(accepted.acceptance.sha256, their_sha);
        assert!(
            accepted.local.is_none(),
            "terms that are not ours must not be recordable locally"
        );
    }

    /// When the server pins what we pin, the payload still comes from the
    /// server's row, and the local handle is available for a fallback pull.
    #[test]
    fn matching_server_terms_stay_recordable_locally() {
        let ours = &mold_core::license_acceptance::INSIGHTFACE_ANTELOPEV2;
        let terms = vec![server_status(ours.url, ours.sha256)];

        let accepted = resolve_and_show_license("insightface-antelopev2", Some(&terms)).unwrap();
        assert_eq!(accepted.acceptance.url, ours.url);
        assert_eq!(accepted.acceptance.sha256, ours.sha256);
        assert_eq!(accepted.local.map(|license| license.id), Some(ours.id));
    }

    /// No server to ask (older build, or unreachable): this build's own pinned
    /// constant is displayed and sent.
    #[test]
    fn without_server_terms_the_local_pin_is_displayed_and_sent() {
        let ours = &mold_core::license_acceptance::INSIGHTFACE_ANTELOPEV2;
        let accepted = resolve_and_show_license("insightface-antelopev2", None).unwrap();
        assert_eq!(accepted.acceptance.url, ours.url);
        assert_eq!(accepted.acceptance.sha256, ours.sha256);
        assert_eq!(accepted.local.map(|license| license.id), Some(ours.id));
    }

    #[test]
    fn an_id_the_server_does_not_know_is_refused() {
        let ours = &mold_core::license_acceptance::INSIGHTFACE_ANTELOPEV2;
        let terms = vec![server_status(ours.url, ours.sha256)];
        assert!(resolve_and_show_license("not-a-license", Some(&terms)).is_err());
        assert!(resolve_and_show_license("not-a-license", None).is_err());
    }

    #[test]
    fn rendered_primary_dest_skips_explicit_companion_roles() {
        let recipe = DownloadRecipe {
            files: vec![
                RecipeFile {
                    url: "https://example/vae".into(),
                    dest: "vae.safetensors".into(),
                    sha256: None,
                    size_bytes: None,
                    role: Some(RecipeFileRole::Vae),
                },
                RecipeFile {
                    url: "https://example/model".into(),
                    dest: "model.safetensors".into(),
                    sha256: None,
                    size_bytes: None,
                    role: None,
                },
            ],
            needs_token: None,
        };
        let rendered = vec![
            "resolved/vae.safetensors".into(),
            "resolved/model.safetensors".into(),
        ];
        assert_eq!(
            rendered_primary_dest(&recipe, &rendered),
            Some("resolved/model.safetensors")
        );
    }
}
