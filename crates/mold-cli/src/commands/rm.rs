use std::collections::HashMap;
use std::io::{self, Write};

use anyhow::Result;
use clap_complete::engine::CompletionCandidate;
use colored::Colorize;
use mold_core::manifest::resolve_model_name;
use mold_core::Config;

use crate::AlreadyReported;

/// Provide completions for installed (configured) model names only.
pub fn complete_installed_model_name() -> Vec<CompletionCandidate> {
    let config = Config::load_or_default();
    config.models.keys().map(CompletionCandidate::new).collect()
}

/// Format a byte count as a human-readable size string.
fn format_size(bytes: u64) -> String {
    if bytes >= 1_073_741_824 {
        format!("{:.1} GB", bytes as f64 / 1_073_741_824.0)
    } else if bytes >= 1_048_576 {
        format!("{:.1} MB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{} B", bytes)
    }
}

/// Get file size, returning 0 if the file doesn't exist or can't be read.
fn file_size(path: &str) -> u64 {
    std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
}

/// Build a map of file_path -> list of model names that reference it.
fn build_ref_counts(config: &Config) -> HashMap<String, Vec<String>> {
    let mut refs: HashMap<String, Vec<String>> = HashMap::new();
    for (model_name, model_config) in &config.models {
        for path in model_config.all_file_paths() {
            refs.entry(path).or_default().push(model_name.clone());
        }
    }
    refs
}

pub async fn run(models: &[String], force: bool) -> Result<()> {
    let mut config = Config::load_or_default();
    let mut any_error = false;

    for model_arg in models {
        let canonical = resolve_model_name(model_arg);

        // Check if the model is installed
        if !config.models.contains_key(&canonical) {
            eprintln!(
                "{} {} is not installed",
                "error:".red().bold(),
                canonical.bold()
            );
            any_error = true;
            continue;
        }

        // Build reference counts across all installed models
        let ref_counts = build_ref_counts(&config);

        let model_config = config.models.get(&canonical).unwrap();
        let all_paths = model_config.all_file_paths();

        // Classify files as unique (only this model) or shared
        let mut unique_files: Vec<(String, u64)> = Vec::new();
        let mut shared_files: Vec<(String, Vec<String>)> = Vec::new();

        for path in &all_paths {
            let refs = ref_counts.get(path).cloned().unwrap_or_default();
            let other_refs: Vec<String> =
                refs.into_iter().filter(|name| name != &canonical).collect();

            if other_refs.is_empty() {
                unique_files.push((path.clone(), file_size(path)));
            } else {
                shared_files.push((path.clone(), other_refs));
            }
        }

        let total_freed: u64 = unique_files.iter().map(|(_, size)| size).sum();

        // Display summary
        println!("{}", canonical.bold());
        for (path, size) in &unique_files {
            let filename = std::path::Path::new(path)
                .file_name()
                .map(|f| f.to_string_lossy().to_string())
                .unwrap_or_else(|| path.clone());
            println!(
                "  {}   {} ({})",
                "Delete:".red(),
                filename,
                format_size(*size)
            );
        }
        for (path, refs) in &shared_files {
            let filename = std::path::Path::new(path)
                .file_name()
                .map(|f| f.to_string_lossy().to_string())
                .unwrap_or_else(|| path.clone());
            println!(
                "  {}   {} (used by {})",
                "Shared:".yellow(),
                filename,
                refs.join(", ")
            );
        }

        if unique_files.is_empty() && shared_files.is_empty() {
            println!("  {}", "(no files configured)".dimmed());
        }

        println!();

        // Confirm
        if !force {
            print!(
                "Remove {}? This will free {}. [y/N] ",
                canonical.bold(),
                format_size(total_freed)
            );
            io::stdout().flush()?;
            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            if !input.trim().eq_ignore_ascii_case("y") {
                println!("Skipped {}", canonical);
                continue;
            }
        }

        // Delete unique files
        let mut freed: u64 = 0;
        for (path, size) in &unique_files {
            match std::fs::remove_file(path) {
                Ok(()) => freed += size,
                Err(e) if e.kind() == io::ErrorKind::NotFound => {
                    eprintln!("{} {} already deleted", "warning:".yellow().bold(), path);
                }
                Err(e) => {
                    eprintln!(
                        "{} failed to delete {}: {}",
                        "warning:".yellow().bold(),
                        path,
                        e
                    );
                }
            }
        }

        // Remove from config
        config.remove_model(&canonical);

        // Handle default_model update
        if config.default_model == canonical {
            let new_default = config
                .models
                .keys()
                .min()
                .cloned()
                .unwrap_or_else(|| "flux-schnell".to_string());
            if config.models.is_empty() {
                eprintln!(
                    "{} default model reset to {}",
                    "note:".dimmed(),
                    "flux-schnell".bold()
                );
            } else {
                eprintln!(
                    "{} default model changed to {}",
                    "note:".dimmed(),
                    new_default.bold()
                );
            }
            config.default_model = new_default;
        }

        config.save()?;

        println!(
            "Removed {} (freed {})",
            canonical.bold(),
            format_size(freed)
        );
    }

    if any_error {
        return Err(AlreadyReported.into());
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use mold_core::ModelConfig;

    #[test]
    fn format_size_gb() {
        assert_eq!(format_size(7_516_192_768), "7.0 GB");
    }

    #[test]
    fn format_size_mb() {
        assert_eq!(format_size(52_428_800), "50.0 MB");
    }

    #[test]
    fn format_size_zero() {
        assert_eq!(format_size(0), "0 B");
    }

    #[test]
    fn all_file_paths_collects_all() {
        let cfg = ModelConfig {
            transformer: Some("/a/transformer.gguf".into()),
            vae: Some("/a/vae.safetensors".into()),
            t5_encoder: Some("/a/t5.safetensors".into()),
            clip_encoder: Some("/a/clip.safetensors".into()),
            t5_tokenizer: Some("/a/t5.tokenizer.json".into()),
            clip_tokenizer: Some("/a/clip.tokenizer.json".into()),
            transformer_shards: Some(vec!["/a/shard1.safetensors".into()]),
            text_encoder_files: Some(vec!["/a/enc1.safetensors".into()]),
            text_tokenizer: Some("/a/text.tokenizer.json".into()),
            ..Default::default()
        };
        let paths = cfg.all_file_paths();
        assert_eq!(paths.len(), 9);
    }

    #[test]
    fn ref_counts_shared_file() {
        let mut config = Config::default();
        config.models.insert(
            "model-a".into(),
            ModelConfig {
                transformer: Some("/unique-a.gguf".into()),
                vae: Some("/shared-vae.safetensors".into()),
                ..Default::default()
            },
        );
        config.models.insert(
            "model-b".into(),
            ModelConfig {
                transformer: Some("/unique-b.gguf".into()),
                vae: Some("/shared-vae.safetensors".into()),
                ..Default::default()
            },
        );

        let refs = build_ref_counts(&config);
        assert_eq!(refs["/shared-vae.safetensors"].len(), 2);
        assert_eq!(refs["/unique-a.gguf"].len(), 1);
        assert_eq!(refs["/unique-b.gguf"].len(), 1);
    }

    #[test]
    fn complete_returns_installed_only() {
        // This just verifies it doesn't panic — actual model list depends on user config
        let candidates = complete_installed_model_name();
        // May be empty if no models installed; that's fine
        assert!(candidates.len() < 1000);
    }
}
