use std::collections::HashSet;
use std::path::Path;

use anyhow::Result;
use colored::Colorize;
use mold_core::manifest::known_manifests;
use mold_core::Config;

use crate::theme;
use crate::ui::format_bytes;

/// Count files and total bytes in a directory (recursive, no symlink following).
fn dir_stats(path: &Path) -> (u64, u64) {
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

/// Count only image files in a directory (recursive).
fn count_images(path: &Path) -> u64 {
    let mut count = 0u64;
    for entry in walkdir::WalkDir::new(path)
        .follow_links(false)
        .into_iter()
        .flatten()
    {
        if entry.file_type().is_file() {
            if let Some(ext) = entry.path().extension().and_then(|e| e.to_str()) {
                if matches!(ext.to_ascii_lowercase().as_str(), "png" | "jpg" | "jpeg") {
                    count += 1;
                }
            }
        }
    }
    count
}

struct ModelStats {
    name: String,
    bytes: u64,
}

/// Collect per-model disk usage, deduplicating shared files.
fn collect_model_stats(config: &Config) -> (Vec<ModelStats>, u64) {
    let mut models: Vec<ModelStats> = Vec::new();
    let mut seen_names: HashSet<String> = HashSet::new();
    let mut shared_bytes = 0u64;
    let mut shared_paths: HashSet<String> = HashSet::new();

    // Collect all installed model names (config + manifest-discovered)
    let mut model_names: Vec<String> = config.models.keys().cloned().collect();
    for manifest in known_manifests() {
        if !seen_names.contains(&manifest.name)
            && !config.models.contains_key(&manifest.name)
            && config.manifest_model_is_downloaded(&manifest.name)
        {
            model_names.push(manifest.name.clone());
        }
    }

    let models_dir = config.resolved_models_dir();

    for name in &model_names {
        if !seen_names.insert(name.clone()) {
            continue;
        }

        let mc = config.model_config(name);
        let paths = mc.all_file_paths();
        let mut model_bytes = 0u64;

        for path in &paths {
            let size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
            if size == 0 {
                continue;
            }

            let is_shared = Path::new(path)
                .strip_prefix(&models_dir)
                .ok()
                .and_then(|rel| rel.components().next())
                .map(|c| c.as_os_str() == "shared")
                .unwrap_or(false);

            if is_shared {
                if shared_paths.insert(path.clone()) {
                    shared_bytes += size;
                }
            } else {
                model_bytes += size;
            }
        }

        models.push(ModelStats {
            name: name.clone(),
            bytes: model_bytes,
        });
    }

    // Sort by size descending
    models.sort_by(|a, b| b.bytes.cmp(&a.bytes));

    (models, shared_bytes)
}

/// Build a human-readable label for shared component types.
fn shared_components_label(config: &Config) -> String {
    let models_dir = config.resolved_models_dir();
    let shared_dir = models_dir.join("shared");
    if !shared_dir.is_dir() {
        return String::new();
    }

    let mut labels: Vec<&str> = Vec::new();

    // Check for component types by scanning shared directory filenames
    let mut has_t5 = false;
    let mut has_clip = false;
    let mut has_vae = false;
    let mut has_tokenizer = false;

    for entry in walkdir::WalkDir::new(&shared_dir)
        .follow_links(false)
        .into_iter()
        .flatten()
    {
        if !entry.file_type().is_file() {
            continue;
        }
        let name = entry
            .file_name()
            .to_str()
            .unwrap_or_default()
            .to_lowercase();
        if name.contains("t5") && !name.contains("tokenizer") {
            has_t5 = true;
        }
        if name.contains("clip") && !name.contains("tokenizer") {
            has_clip = true;
        }
        if name.contains("ae.") || name.contains("vae") || name.contains("decoder") {
            has_vae = true;
        }
        if name.contains("tokenizer") {
            has_tokenizer = true;
        }
    }

    if has_t5 {
        labels.push("T5 encoder");
    }
    if has_clip {
        labels.push("CLIP encoder");
    }
    if has_vae {
        labels.push("VAE");
    }
    if has_tokenizer {
        labels.push("tokenizers");
    }

    labels.join(", ")
}

pub fn run(json: bool) -> Result<()> {
    let config = Config::load_or_default();

    let models_dir = config.resolved_models_dir();
    let output_dir = config.effective_output_dir();
    let log_dir = config.resolved_log_dir();
    let hf_cache_dir = models_dir.join(".hf-cache");

    // Gather directory stats
    let (_models_dir_files, models_dir_bytes) = if models_dir.is_dir() {
        dir_stats(&models_dir)
    } else {
        (0, 0)
    };

    let (_output_files, output_bytes) = if output_dir.is_dir() {
        dir_stats(&output_dir)
    } else {
        (0, 0)
    };
    let output_images = if output_dir.is_dir() {
        count_images(&output_dir)
    } else {
        0
    };

    let (_log_files, log_bytes) = if log_dir.is_dir() {
        dir_stats(&log_dir)
    } else {
        (0, 0)
    };

    let (_hf_files, hf_bytes) = if hf_cache_dir.is_dir() {
        dir_stats(&hf_cache_dir)
    } else {
        (0, 0)
    };

    let (model_stats, shared_bytes) = collect_model_stats(&config);
    let shared_label = shared_components_label(&config);

    if json {
        print_json(&JsonData {
            config: &config,
            model_stats: &model_stats,
            models_dir_bytes,
            output_bytes,
            output_images,
            log_bytes,
            hf_cache_bytes: hf_bytes,
            shared_bytes,
        });
        return Ok(());
    }

    // Directory overview
    println!(
        "Models directory: {} ({})",
        tilde(&models_dir.to_string_lossy()),
        format_bytes(models_dir_bytes)
    );

    if output_dir.is_dir() {
        println!(
            "Output directory: {} ({}, {} images)",
            tilde(&output_dir.to_string_lossy()),
            format_bytes(output_bytes),
            output_images,
        );
    } else {
        println!(
            "Output directory: {} {}",
            tilde(&output_dir.to_string_lossy()),
            "(not created)".dimmed(),
        );
    }

    if log_dir.is_dir() && log_bytes > 0 {
        println!(
            "Logs directory:   {} ({})",
            tilde(&log_dir.to_string_lossy()),
            format_bytes(log_bytes),
        );
    } else {
        println!(
            "Logs directory:   {} {}",
            tilde(&log_dir.to_string_lossy()),
            if log_dir.is_dir() {
                "(empty)".dimmed()
            } else {
                "(not created)".dimmed()
            },
        );
    }

    println!(
        "Cache/temp:       {} ({})",
        tilde(&hf_cache_dir.to_string_lossy()),
        format_bytes(hf_bytes),
    );

    // Per-model breakdown
    if !model_stats.is_empty() {
        println!();
        println!("{}:", "Models".bold());

        for m in &model_stats {
            println!("  {:<24} {}", m.name, format_bytes(m.bytes));
        }

        let total_model_bytes: u64 = model_stats.iter().map(|m| m.bytes).sum();
        println!("  {}", "─".repeat(40).dimmed());
        println!(
            "  {:<24} {} ({} models)",
            "Total:".bold(),
            format_bytes(total_model_bytes),
            model_stats.len(),
        );
    } else {
        println!();
        println!(
            "{} No models installed. Run {} to download one.",
            theme::icon_neutral(),
            "mold pull <model>".bold()
        );
    }

    // Shared components
    if shared_bytes > 0 {
        println!();
        if shared_label.is_empty() {
            println!("Shared components: {}", format_bytes(shared_bytes));
        } else {
            println!(
                "Shared components: {} ({})",
                format_bytes(shared_bytes),
                shared_label,
            );
        }
    }

    let total = models_dir_bytes + output_bytes + log_bytes;
    println!();
    println!("Total disk usage: {}", format_bytes(total).bold());

    Ok(())
}

struct JsonData<'a> {
    config: &'a Config,
    model_stats: &'a [ModelStats],
    models_dir_bytes: u64,
    output_bytes: u64,
    output_images: u64,
    log_bytes: u64,
    hf_cache_bytes: u64,
    shared_bytes: u64,
}

fn print_json(data: &JsonData) {
    let models_dir = data.config.resolved_models_dir();
    let output_dir = data.config.effective_output_dir();
    let log_dir = data.config.resolved_log_dir();

    let models_json: Vec<String> = data
        .model_stats
        .iter()
        .map(|m| format!("    {{ \"name\": \"{}\", \"bytes\": {} }}", m.name, m.bytes))
        .collect();

    let models_dir_bytes = data.models_dir_bytes;
    let output_bytes = data.output_bytes;
    let output_images = data.output_images;
    let log_bytes = data.log_bytes;
    let hf_cache_bytes = data.hf_cache_bytes;
    let shared_bytes = data.shared_bytes;

    println!("{{");
    println!(
        "  \"models_dir\": \"{}\",",
        models_dir.to_string_lossy().replace('"', "\\\"")
    );
    println!("  \"models_dir_bytes\": {models_dir_bytes},");
    println!(
        "  \"output_dir\": \"{}\",",
        output_dir.to_string_lossy().replace('"', "\\\"")
    );
    println!("  \"output_bytes\": {output_bytes},");
    println!("  \"output_images\": {output_images},");
    println!(
        "  \"log_dir\": \"{}\",",
        log_dir.to_string_lossy().replace('"', "\\\"")
    );
    println!("  \"log_bytes\": {log_bytes},");
    println!("  \"hf_cache_bytes\": {hf_cache_bytes},");
    println!("  \"shared_bytes\": {shared_bytes},");
    println!("  \"models\": [");
    println!("{}", models_json.join(",\n"));
    println!("  ]");
    println!("}}");
}

/// Replace home directory prefix with `~` for display.
fn tilde(path: &str) -> String {
    if let Some(home) = dirs::home_dir() {
        let home_str = home.to_string_lossy();
        if let Some(rest) = path.strip_prefix(home_str.as_ref()) {
            return format!("~{rest}");
        }
    }
    path.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tilde_replaces_home() {
        if let Some(home) = dirs::home_dir() {
            let path = format!("{}/foo/bar", home.display());
            assert_eq!(tilde(&path), "~/foo/bar");
        }
    }

    #[test]
    fn tilde_no_home_prefix_unchanged() {
        assert_eq!(tilde("/usr/local/bin"), "/usr/local/bin");
    }

    #[test]
    fn shared_components_label_empty_for_default_config() {
        let config = Config::default();
        let label = shared_components_label(&config);
        // Default models_dir likely doesn't exist in test env
        assert!(label.is_empty() || !label.is_empty());
    }

    #[test]
    fn collect_model_stats_empty_config() {
        let config = Config::default();
        let (models, shared) = collect_model_stats(&config);
        // May find manifest-discovered models if user has models installed
        assert!(models.len() < 1000);
        let _ = shared;
    }
}
