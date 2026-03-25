use anyhow::Result;
use colored::Colorize;
use mold_core::manifest::{find_manifest, resolve_model_name, ModelComponent};
use mold_core::{Config, ModelPaths};
use sha2::{Digest, Sha256};

use crate::theme;
use crate::ui::format_family;

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
                "  {:<16} 🔒 Requires HF_TOKEN for download",
                "Auth:".dimmed(),
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
            let gated_marker = if file.gated { " 🔒" } else { "" };
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
            format_args!("✅ {}", "Installed".green().bold())
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
            format_args!("📦 {}", "Not installed".red()),
            format_args!("mold pull {}", canonical).to_string().bold()
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
}
