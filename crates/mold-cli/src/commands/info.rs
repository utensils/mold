use std::collections::HashSet;
use std::time::Duration;

use anyhow::Result;
use colored::Colorize;
use mold_core::manifest::{find_manifest, resolve_model_name, ModelComponent};
use mold_core::{build_model_catalog, Config, ModelPaths, MoldClient};
use sha2::{Digest, Sha256};

use crate::theme;
use crate::ui::{col_width, format_disk_size, format_family, format_family_padded};

/// Count files and total bytes in a directory (recursive).
/// Count files and total bytes in a directory (recursive, no symlink following).
/// Avoids double-counting symlinked HF cache blobs.
fn dir_stats(path: &std::path::Path) -> (u64, u64) {
    let mut files = 0u64;
    let mut bytes = 0u64;
    for entry in walkdir::WalkDir::new(path)
        .follow_links(false)
        .into_iter()
        .flatten()
    {
        if entry.file_type().is_file() {
            files += 1;
            bytes += entry.metadata().map(|m| m.len()).unwrap_or(0);
        }
    }
    (files, bytes)
}

fn compute_sha256(path: &str) -> Result<String> {
    use std::io::Read;
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20]; // 1MB buffer
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn resolve_file_path(
    model_config: Option<&mold_core::config::ModelConfig>,
    component: &ModelComponent,
) -> Option<String> {
    let mcfg = model_config?;
    match component {
        ModelComponent::Transformer => mcfg.transformer.clone(),
        ModelComponent::Vae => mcfg.vae.clone(),
        ModelComponent::T5Encoder => mcfg.t5_encoder.clone(),
        ModelComponent::ClipEncoder => mcfg.clip_encoder.clone(),
        ModelComponent::T5Tokenizer => mcfg.t5_tokenizer.clone(),
        ModelComponent::ClipTokenizer => mcfg.clip_tokenizer.clone(),
        ModelComponent::ClipEncoder2 => mcfg.clip_encoder_2.clone(),
        ModelComponent::ClipTokenizer2 => mcfg.clip_tokenizer_2.clone(),
        ModelComponent::TextTokenizer => mcfg.text_tokenizer.clone(),
        ModelComponent::Decoder => mcfg.decoder.clone(),
        ModelComponent::TransformerShard | ModelComponent::TextEncoder => None,
    }
}

/// Resolve a file path for verification: check ModelPaths (which honors env vars)
/// first, then fall back to config-only resolution.
fn resolve_verify_path(
    resolved: Option<&ModelPaths>,
    model_config: Option<&mold_core::config::ModelConfig>,
    component: &ModelComponent,
) -> Option<String> {
    if let Some(paths) = resolved {
        let path = match component {
            ModelComponent::Transformer => Some(&paths.transformer),
            ModelComponent::Vae => Some(&paths.vae),
            ModelComponent::T5Encoder => paths.t5_encoder.as_ref(),
            ModelComponent::ClipEncoder => paths.clip_encoder.as_ref(),
            ModelComponent::T5Tokenizer => paths.t5_tokenizer.as_ref(),
            ModelComponent::ClipTokenizer => paths.clip_tokenizer.as_ref(),
            ModelComponent::ClipEncoder2 => paths.clip_encoder_2.as_ref(),
            ModelComponent::ClipTokenizer2 => paths.clip_tokenizer_2.as_ref(),
            ModelComponent::TextTokenizer => paths.text_tokenizer.as_ref(),
            ModelComponent::Decoder => paths.decoder.as_ref(),
            // Shards and text encoder files are multi-valued; fall through to config
            ModelComponent::TransformerShard | ModelComponent::TextEncoder => None,
        };
        if let Some(p) = path {
            return Some(p.to_string_lossy().to_string());
        }
    }
    // Fall back to config-only resolution for components ModelPaths doesn't cover
    resolve_file_path(model_config, component)
}

fn component_label(component: &ModelComponent) -> &'static str {
    match component {
        ModelComponent::Transformer => "Transformer",
        ModelComponent::TransformerShard => "Transformer Shard",
        ModelComponent::Vae => "VAE",
        ModelComponent::T5Encoder => "T5 Encoder",
        ModelComponent::ClipEncoder => "CLIP-L Encoder",
        ModelComponent::T5Tokenizer => "T5 Tokenizer",
        ModelComponent::ClipTokenizer => "CLIP-L Tokenizer",
        ModelComponent::ClipEncoder2 => "CLIP-G Encoder",
        ModelComponent::ClipTokenizer2 => "CLIP-G Tokenizer",
        ModelComponent::TextEncoder => "Text Encoder",
        ModelComponent::TextTokenizer => "Text Tokenizer",
        ModelComponent::Decoder => "Decoder",
    }
}

pub async fn run_overview() -> Result<()> {
    let config = Config::load_or_default();

    // Header
    println!(
        "{}",
        format!("mold v{}", mold_core::build_info::version_string()).bold()
    );
    println!("{}", "─".repeat(42).dimmed());

    // Installation section
    println!();
    println!("  {}", "Installation".bold());

    if let Some(home) = Config::mold_dir() {
        println!("  {:<18} {}", "Home:".dimmed(), home.display());
    }

    if let Some(config_path) = Config::config_path() {
        let status = if config_path.exists() {
            "exists".green().to_string()
        } else {
            "not found".dimmed().to_string()
        };
        println!(
            "  {:<18} {} ({})",
            "Config:".dimmed(),
            config_path.display(),
            status,
        );
    }

    let models_dir = config.resolved_models_dir();
    let models_dir_status = if models_dir.exists() {
        String::new()
    } else {
        format!(" ({})", "not created".dimmed())
    };
    println!(
        "  {:<18} {}{}",
        "Models dir:".dimmed(),
        models_dir.display(),
        models_dir_status,
    );

    let default_model = resolve_model_name(&config.resolved_default_model());
    let default_downloaded = config.manifest_model_is_downloaded(&default_model);
    let default_status = if default_downloaded {
        "installed".green().to_string()
    } else {
        "not installed".dimmed().to_string()
    };
    println!(
        "  {:<18} {} ({})",
        "Default model:".dimmed(),
        default_model,
        default_status,
    );

    // Installed Models section
    let catalog = build_model_catalog(&config, None, false);
    let downloaded: Vec<_> = catalog.iter().filter(|m| m.downloaded).collect();

    println!();
    println!("  {} ({})", "Installed Models".bold(), downloaded.len());

    if downloaded.is_empty() {
        println!(
            "  {} Run {} to download.",
            "No models installed.".dimmed(),
            "mold pull <model>".bold(),
        );
    } else {
        let nw = col_width(downloaded.iter().map(|m| m.name.len()), 4, 2);
        let fw = col_width(
            downloaded
                .iter()
                .map(|m| crate::ui::family_label(&m.family).len()),
            6,
            2,
        );

        let mut all_unique_paths: HashSet<String> = HashSet::new();
        for model in &downloaded {
            let mcfg = config.model_config(&model.name);
            let model_paths = mcfg.all_file_paths();
            let disk_bytes: u64 = model.disk_usage_bytes.unwrap_or_else(|| {
                model_paths
                    .iter()
                    .filter_map(|p| std::fs::metadata(p).ok())
                    .map(|m| m.len())
                    .sum()
            });
            all_unique_paths.extend(model_paths);
            println!(
                "  {:<nw$} {} {:>7}",
                model.name,
                format_family_padded(&model.family, fw),
                format_disk_size(disk_bytes),
                nw = nw,
            );
        }

        if downloaded.len() > 1 {
            let total_disk: u64 = all_unique_paths
                .iter()
                .filter_map(|p| std::fs::metadata(p).ok())
                .map(|m| m.len())
                .sum();
            if total_disk > 0 {
                println!(
                    "{}",
                    format!(
                        "{:>width$}",
                        format!("Total: {}", format_disk_size(total_disk)),
                        width = nw + fw + 9,
                    )
                    .dimmed(),
                );
            }
        }
    }

    // Storage section
    println!();
    println!("  {}", "Storage".bold());

    // Cache directory
    if let Some(home) = Config::mold_dir() {
        let cache_dir = home.join("cache");
        if cache_dir.exists() {
            let (files, bytes) = dir_stats(&cache_dir);
            println!(
                "  {:<18} {} ({}, {} files)",
                "Cache:".dimmed(),
                cache_dir.display(),
                format_disk_size(bytes),
                files,
            );
        } else {
            println!(
                "  {:<18} {} ({})",
                "Cache:".dimmed(),
                cache_dir.display(),
                "empty".dimmed(),
            );
        }
    }

    // Output directory — check config/env first, then fall back to $MOLD_HOME/output
    let output_dir = config
        .resolved_output_dir()
        .or_else(|| Config::mold_dir().map(|h| h.join("output")));
    if let Some(output_dir) = output_dir {
        if output_dir.exists() {
            let (files, bytes) = dir_stats(&output_dir);
            if files > 0 {
                println!(
                    "  {:<18} {} ({}, {} images)",
                    "Output:".dimmed(),
                    output_dir.display(),
                    format_disk_size(bytes),
                    files,
                );
            }
        }
    }

    // HuggingFace hub cache — mold uses <models_dir>/.hf-cache/
    let hf_cache = models_dir.join(".hf-cache");
    if hf_cache.exists() {
        let (files, bytes) = dir_stats(&hf_cache);
        if bytes > 0 {
            println!(
                "  {:<18} {} ({}, {} files)",
                "HF hub cache:".dimmed(),
                hf_cache.display(),
                format_disk_size(bytes),
                files,
            );
        }
    }

    // Prompt Expansion section
    let expand_settings = config.expand.clone().with_env_overrides();
    println!();
    println!("  {}", "Prompt Expansion".bold());
    println!(
        "  {:<18} {}",
        "Enabled:".dimmed(),
        if expand_settings.enabled {
            "yes".green().to_string()
        } else {
            "no".dimmed().to_string()
        },
    );
    println!(
        "  {:<18} {}",
        "Backend:".dimmed(),
        if expand_settings.is_local() {
            "local".to_string()
        } else {
            expand_settings.backend.clone()
        },
    );
    let expand_model_name = if expand_settings.is_local() {
        &expand_settings.model
    } else {
        &expand_settings.api_model
    };
    println!("  {:<18} {}", "Model:".dimmed(), expand_model_name);
    if expand_settings.system_prompt.is_some() {
        println!("  {:<18} {}", "System prompt:".dimmed(), "custom".yellow());
    }
    if expand_settings.batch_prompt.is_some() {
        println!("  {:<18} {}", "Batch prompt:".dimmed(), "custom".yellow());
    }
    if !expand_settings.families.is_empty() {
        let mut families: Vec<&String> = expand_settings.families.keys().collect();
        families.sort();
        println!(
            "  {:<18} {}",
            "Family overrides:".dimmed(),
            families
                .iter()
                .map(|f| f.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        );
    }

    // Server section
    let client = MoldClient::from_env();
    println!();
    println!("  {}", "Server".bold());
    println!("  {:<18} {}", "Host:".dimmed(), client.host());

    match tokio::time::timeout(Duration::from_secs(2), client.server_status()).await {
        Ok(Ok(status)) => {
            let loaded = if status.models_loaded.is_empty() {
                String::new()
            } else {
                format!(" ({} loaded)", status.models_loaded.join(", "))
            };
            println!(
                "  {:<18} {} (v{}){}",
                "Status:".dimmed(),
                "Running".green(),
                status.version,
                loaded,
            );
        }
        _ => {
            println!("  {:<18} {}", "Status:".dimmed(), "Not running".dimmed(),);
        }
    }

    Ok(())
}

pub fn run(name: &str, verify: bool) -> Result<()> {
    let canonical = resolve_model_name(name);
    let config = Config::load_or_default();
    let resolved_model_cfg = config.resolved_model_config(&canonical);

    let manifest = find_manifest(&canonical);
    let model_config = if manifest.is_some() {
        let cfg = config.model_config(&canonical);
        (!cfg.all_file_paths().is_empty()).then_some(cfg)
    } else {
        config
            .models
            .get(&canonical)
            .or_else(|| config.models.get(name))
            .cloned()
    };
    let is_installed = if manifest.is_some() {
        config.manifest_model_is_downloaded(&canonical)
    } else {
        model_config.is_some()
    };

    if manifest.is_none() && model_config.is_none() {
        eprintln!(
            "{} Unknown model '{}'. Use {} to see available models.",
            theme::prefix_error(),
            canonical,
            "mold list".bold()
        );
        return Err(crate::AlreadyReported.into());
    }

    // Header
    println!("{}", canonical.bold());
    println!("{}", "─".repeat(60).dimmed());

    if let Some(m) = manifest {
        println!("  {:<16} {}", "Family:".dimmed(), format_family(&m.family));
        println!("  {:<16} {}", "Description:".dimmed(), m.description);
        if m.is_gated() {
            println!(
                "  {:<16} {} — requires HF_TOKEN for download",
                "Auth:".dimmed(),
                "Gated".yellow().bold(),
            );
        }

        // Size info — compute from manifest files, accounting for cache
        let (total_bytes, remaining_bytes) = mold_core::manifest::compute_download_size(m);
        let total_gb = total_bytes as f64 / 1_073_741_824.0;
        let remaining_gb = remaining_bytes as f64 / 1_073_741_824.0;
        if remaining_bytes == 0 {
            println!(
                "  {:<16} {:.1}GB total (fully cached)",
                "Size:".dimmed(),
                total_gb,
            );
        } else if remaining_gb < total_gb - 0.1 {
            println!(
                "  {:<16} {:.1}GB total ({:.1}GB to download)",
                "Size:".dimmed(),
                total_gb,
                remaining_gb,
            );
        } else {
            println!("  {:<16} {:.1}GB total", "Size:".dimmed(), total_gb,);
        }

        // Generation defaults
        println!();
        println!("  {}", "Generation Defaults".bold());
        println!(
            "  {:<16} {}",
            "Steps:".dimmed(),
            resolved_model_cfg.effective_steps(&config)
        );
        println!(
            "  {:<16} {:.1}",
            "Guidance:".dimmed(),
            resolved_model_cfg.effective_guidance()
        );
        println!(
            "  {:<16} {}x{}",
            "Dimensions:".dimmed(),
            resolved_model_cfg.effective_width(&config),
            resolved_model_cfg.effective_height(&config)
        );
        if let Some(scheduler) = resolved_model_cfg.scheduler.or(m.defaults.scheduler) {
            println!("  {:<16} {}", "Scheduler:".dimmed(), scheduler);
        }

        // HuggingFace sources
        println!();
        println!("  {}", "HuggingFace Sources".bold());
        for file in &m.files {
            let url = format!(
                "https://huggingface.co/{}/blob/main/{}",
                file.hf_repo, file.hf_filename
            );
            let size_mb = file.size_bytes as f64 / (1024.0 * 1024.0);
            let size_str = if size_mb >= 1024.0 {
                format!("{:.1}GB", size_mb / 1024.0)
            } else {
                format!("{:.0}MB", size_mb)
            };
            let cached = mold_core::manifest::is_file_cached(m, file);
            let status_marker = if cached {
                "cached".green().to_string()
            } else {
                "not downloaded".dimmed().to_string()
            };
            let gated_marker = if file.gated { ", gated" } else { "" };
            println!(
                "  {:<16} {} ({}{}, {})",
                format!("{}:", component_label(&file.component)).dimmed(),
                url,
                size_str,
                gated_marker,
                status_marker,
            );
        }
    } else if model_config.is_some() {
        // Custom model — show what we know from config
        let family = resolved_model_cfg.family.as_deref().unwrap_or("unknown");
        println!("  {:<16} {}", "Family:".dimmed(), format_family(family));
        if let Some(ref desc) = resolved_model_cfg.description {
            println!("  {:<16} {}", "Description:".dimmed(), desc);
        }
        println!();
        println!("  {}", "Generation Defaults".bold());
        println!(
            "  {:<16} {}",
            "Steps:".dimmed(),
            resolved_model_cfg.effective_steps(&config)
        );
        println!(
            "  {:<16} {:.1}",
            "Guidance:".dimmed(),
            resolved_model_cfg.effective_guidance()
        );
        println!(
            "  {:<16} {}x{}",
            "Dimensions:".dimmed(),
            resolved_model_cfg.effective_width(&config),
            resolved_model_cfg.effective_height(&config)
        );
    }

    // Installation status + local file paths
    println!();
    if is_installed {
        println!(
            "  {:<16} {}",
            "Status:".dimmed(),
            "Installed".green().bold()
        );
        if let Some(config_path) = Config::config_path() {
            println!("  {:<16} {}", "Config:".dimmed(), config_path.display());
        }

        // Show local file paths
        println!();
        println!("  {}", "Local Files".bold());
        let mut has_files = false;
        let mcfg = model_config
            .as_ref()
            .expect("installed models should have local file paths");
        if let Some(ref p) = mcfg.transformer {
            println!("  {:<20} {}", "Transformer:".dimmed(), p);
            has_files = true;
        }
        if let Some(ref shards) = mcfg.transformer_shards {
            for (i, p) in shards.iter().enumerate() {
                let label = format!("Xformer Shard {}:", i + 1);
                println!("  {:<20} {}", label.dimmed(), p);
            }
            has_files = true;
        }
        if let Some(ref p) = mcfg.vae {
            println!("  {:<20} {}", "VAE:".dimmed(), p);
            has_files = true;
        }
        if let Some(ref p) = mcfg.t5_encoder {
            println!("  {:<20} {}", "T5 Encoder:".dimmed(), p);
            has_files = true;
        }
        if let Some(ref p) = mcfg.clip_encoder {
            println!("  {:<20} {}", "CLIP-L Encoder:".dimmed(), p);
            has_files = true;
        }
        if let Some(ref p) = mcfg.t5_tokenizer {
            println!("  {:<20} {}", "T5 Tokenizer:".dimmed(), p);
            has_files = true;
        }
        if let Some(ref p) = mcfg.clip_tokenizer {
            println!("  {:<20} {}", "CLIP-L Tokenizer:".dimmed(), p);
            has_files = true;
        }
        if let Some(ref p) = mcfg.clip_encoder_2 {
            println!("  {:<20} {}", "CLIP-G Encoder:".dimmed(), p);
            has_files = true;
        }
        if let Some(ref p) = mcfg.clip_tokenizer_2 {
            println!("  {:<20} {}", "CLIP-G Tokenizer:".dimmed(), p);
            has_files = true;
        }
        if let Some(ref files) = mcfg.text_encoder_files {
            for (i, p) in files.iter().enumerate() {
                let label = format!("Text Enc Shard {}:", i + 1);
                println!("  {:<20} {}", label.dimmed(), p);
            }
            has_files = true;
        }
        if let Some(ref p) = mcfg.text_tokenizer {
            println!("  {:<20} {}", "Text Tokenizer:".dimmed(), p);
            has_files = true;
        }
        if !has_files {
            println!("  {}", "(no paths configured)".dimmed());
        }
        // Estimated memory usage
        if let Some(paths) = ModelPaths::resolve(&canonical, &config) {
            let eager = mold_inference::device::estimate_peak_memory(
                &paths,
                mold_inference::LoadStrategy::Eager,
            );
            let sequential = mold_inference::device::estimate_peak_memory(
                &paths,
                mold_inference::LoadStrategy::Sequential,
            );
            println!();
            println!("  {}", "Estimated Peak Memory".bold());
            println!(
                "  {:<20} {:.1}GB",
                "Eager (--eager):".dimmed(),
                eager as f64 / 1_073_741_824.0
            );
            println!(
                "  {:<20} {:.1}GB",
                "Sequential:".dimmed(),
                sequential as f64 / 1_073_741_824.0
            );
        }
        // SHA-256 verification
        if verify {
            if let Some(m) = manifest {
                println!();
                println!("  {}", "Integrity Check".bold());
                let resolved_paths = ModelPaths::resolve(&canonical, &config);
                let mut all_ok = true;
                for file in &m.files {
                    let local_path =
                        resolve_verify_path(resolved_paths.as_ref(), Some(mcfg), &file.component);
                    match (local_path, file.sha256) {
                        (Some(path), Some(expected)) => match compute_sha256(&path) {
                            Ok(actual) if actual == expected => {
                                println!(
                                    "  {} {} {}",
                                    "OK".green().bold(),
                                    component_label(&file.component),
                                    path.dimmed()
                                );
                            }
                            Ok(actual) => {
                                all_ok = false;
                                println!(
                                    "  {} {} — hash mismatch",
                                    "FAIL".red().bold(),
                                    component_label(&file.component)
                                );
                                println!("    expected: {}", expected.dimmed());
                                println!("    actual:   {}", actual.dimmed());
                            }
                            Err(e) => {
                                all_ok = false;
                                println!(
                                    "  {} {} — {}",
                                    "ERR".red().bold(),
                                    component_label(&file.component),
                                    e
                                );
                            }
                        },
                        (Some(_), None) => {
                            println!(
                                "  {} {} — no checksum available",
                                "SKIP".yellow().bold(),
                                component_label(&file.component)
                            );
                        }
                        (None, _) => {
                            println!(
                                "  {} {} — file not found locally",
                                "SKIP".yellow().bold(),
                                component_label(&file.component)
                            );
                        }
                    }
                }
                if all_ok {
                    println!("  {}", "All verified files OK.".green());
                }
            } else {
                println!();
                println!(
                    "  {} --verify requires a manifest model (not custom config)",
                    "note:".dimmed()
                );
            }
        }
    } else {
        println!(
            "  {:<16} {} — run {} to download",
            "Status:".dimmed(),
            "Not installed".red(),
            format!("mold pull {canonical}").bold()
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn component_labels_are_all_nonempty() {
        let components = [
            ModelComponent::Transformer,
            ModelComponent::TransformerShard,
            ModelComponent::Vae,
            ModelComponent::T5Encoder,
            ModelComponent::ClipEncoder,
            ModelComponent::T5Tokenizer,
            ModelComponent::ClipTokenizer,
            ModelComponent::ClipEncoder2,
            ModelComponent::ClipTokenizer2,
            ModelComponent::TextEncoder,
            ModelComponent::TextTokenizer,
        ];
        for c in &components {
            assert!(!component_label(c).is_empty());
        }
    }

    #[test]
    fn unknown_model_returns_error() {
        let result = run("nonexistent-model-xyz", false);
        assert!(result.is_err());
    }

    #[test]
    fn known_model_succeeds() {
        // flux-schnell is always in the manifest registry
        let result = run("flux-schnell", false);
        assert!(result.is_ok());
    }

    #[test]
    fn sdxl_model_succeeds() {
        let result = run("sdxl-base", false);
        assert!(result.is_ok());
    }

    #[test]
    fn sd15_model_succeeds() {
        let result = run("sd15", false);
        assert!(result.is_ok());
    }

    #[test]
    fn legacy_name_resolves() {
        // flux-dev-q4 should resolve to flux-dev:q4
        let result = run("flux-dev-q4", false);
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn overview_returns_ok() {
        // No server running is fine — overview should print "Not running" and succeed
        let result = run_overview().await;
        assert!(result.is_ok());
    }
}
