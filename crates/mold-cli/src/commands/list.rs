use anyhow::Result;
use colored::Colorize;
use mold_core::{Config, MoldClient};

/// Map raw family key to display label.
fn family_label(family: &str) -> &str {
    match family {
        "flux" => "FLUX.1",
        "sd15" => "SD1.5",
        "sdxl" => "SDXL",
        "z-image" => "Z-Image",
        other => other,
    }
}

/// Pad plain text first, then colorize — ANSI escapes break `{:<N}` formatting.
fn format_family_padded(family: &str, width: usize) -> String {
    let padded = format!("{:<width$}", family_label(family), width = width);
    match family {
        "flux" => padded.magenta().to_string(),
        "sd15" => padded.green().to_string(),
        "sdxl" => padded.yellow().to_string(),
        "z-image" => padded.cyan().to_string(),
        _ => padded,
    }
}

fn format_disk_size(bytes: u64) -> String {
    if bytes == 0 {
        "—".to_string()
    } else if bytes >= 1_073_741_824 {
        format!("{:.1}GB", bytes as f64 / 1_073_741_824.0)
    } else {
        format!("{:.0}MB", bytes as f64 / 1_048_576.0)
    }
}

/// Compute the minimum column width for a set of strings (with padding).
fn col_width(items: impl Iterator<Item = usize>, header_len: usize, pad: usize) -> usize {
    items.fold(header_len, |max, len| max.max(len)) + pad
}

pub async fn run() -> Result<()> {
    let client = MoldClient::from_env();

    match client.list_models_extended().await {
        Ok(models) => {
            // Compute column widths from data
            let nw = col_width(
                models
                    .iter()
                    .map(|m| m.name.len() + if m.is_loaded { 2 } else { 0 }),
                4, // "NAME"
                2,
            );
            let fw = col_width(
                models.iter().map(|m| family_label(&m.family).len()),
                6, // "FAMILY"
                2,
            );

            println!(
                "{:<nw$} {:<fw$} {:>7}  {:<7} {:<9} {:<8} {:<7} {}",
                "NAME".bold(),
                "FAMILY".bold(),
                "SIZE".bold(),
                "STEPS".bold(),
                "GUIDANCE".bold(),
                "WIDTH".bold(),
                "HEIGHT".bold(),
                "DESCRIPTION".bold(),
                nw = nw,
                fw = fw,
            );
            println!("{}", "─".repeat(nw + fw + 56).dimmed());

            for model in &models {
                let name = if model.is_loaded {
                    format!("{} ●", model.name).green().to_string()
                } else {
                    model.name.clone()
                };
                let size = if model.size_gb > 0.0 {
                    format!("{:.1}GB", model.size_gb)
                } else {
                    "—".to_string()
                };
                println!(
                    "{:<nw$} {} {:>7}  {:<7} {:<9} {:<8} {:<7} {}",
                    name,
                    format_family_padded(&model.family, fw),
                    size,
                    model.defaults.default_steps,
                    format!("{:.1}", model.defaults.default_guidance),
                    model.defaults.default_width,
                    model.defaults.default_height,
                    model.defaults.description.dimmed(),
                    nw = nw,
                );
            }

            if models.is_empty() {
                println!("{}", "No models configured.".dimmed());
            }
        }
        Err(_) => {
            let config = Config::load_or_default();

            // Gather available-to-pull models (needed for column width computation)
            let manifests = mold_core::manifest::known_manifests();
            let available: Vec<_> = manifests
                .iter()
                .filter(|m| !config.models.contains_key(&m.name))
                .collect();

            // Compute name column width across both installed and available models
            let nw = col_width(
                config
                    .models
                    .keys()
                    .map(|n| n.len())
                    .chain(available.iter().map(|m| m.name.len())),
                4, // "NAME"
                2,
            );
            let fw = col_width(
                config
                    .models
                    .values()
                    .map(|c| family_label(c.family.as_deref().unwrap_or("")).len())
                    .chain(available.iter().map(|m| family_label(&m.family).len())),
                6, // "FAMILY"
                2,
            );

            if config.models.is_empty() {
                println!("{} No models configured.", "●".dimmed());
            } else {
                println!(
                    "{:<nw$} {:<fw$} {:>7}  {:>7}  {:<7} {:<9} {:<8} {:<7} {}",
                    "NAME".bold(),
                    "FAMILY".bold(),
                    "SIZE".bold(),
                    "DISK".bold(),
                    "STEPS".bold(),
                    "GUIDANCE".bold(),
                    "WIDTH".bold(),
                    "HEIGHT".bold(),
                    "DESCRIPTION".bold(),
                    nw = nw,
                    fw = fw,
                );
                println!("{}", "─".repeat(nw + fw + 64).dimmed());
                let mut all_unique_paths: std::collections::HashSet<String> =
                    std::collections::HashSet::new();
                for (name, mcfg) in &config.models {
                    let family_raw = mcfg.family.as_deref().unwrap_or("");
                    let size = mold_core::manifest::find_manifest(name)
                        .map(|m| format!("{:.1}GB", m.size_gb))
                        .unwrap_or_else(|| "—".to_string());
                    let model_paths = mcfg.all_file_paths();
                    let disk_bytes: u64 = model_paths
                        .iter()
                        .filter_map(|p| std::fs::metadata(p).ok())
                        .map(|m| m.len())
                        .sum();
                    all_unique_paths.extend(model_paths);
                    let disk = format_disk_size(disk_bytes);
                    println!(
                        "{:<nw$} {} {:>7}  {:>7}  {:<7} {:<9} {:<8} {:<7} {}",
                        name,
                        format_family_padded(family_raw, fw),
                        size,
                        disk,
                        mcfg.effective_steps(&config),
                        format!("{:.1}", mcfg.effective_guidance()),
                        mcfg.effective_width(&config),
                        mcfg.effective_height(&config),
                        mcfg.description.as_deref().unwrap_or("").dimmed(),
                        nw = nw,
                    );
                }
                let total_disk: u64 = all_unique_paths
                    .iter()
                    .filter_map(|p| std::fs::metadata(p).ok())
                    .map(|m| m.len())
                    .sum();
                if config.models.len() > 1 && total_disk > 0 {
                    println!(
                        "{:>width$}",
                        format!("Total: {}", format_disk_size(total_disk)).dimmed(),
                        width = nw + fw + 15,
                    );
                }
            }

            // Show available-to-pull models
            if !available.is_empty() {
                println!();
                println!("Available to pull:");
                for m in &available {
                    println!(
                        "  {:<nw$} {} {:>5.1}GB  {}",
                        m.name.bold(),
                        format_family_padded(&m.family, fw),
                        m.size_gb,
                        m.description.dimmed(),
                        nw = nw,
                    );
                }
                println!();
                println!(
                    "{}",
                    format!(
                        "Sizes are transformer only. First pull also downloads {:.1}GB of shared components (T5, CLIP, VAE).",
                        mold_core::manifest::SHARED_COMPONENTS_GB
                    ).dimmed(),
                );
                println!("Use {} to download.", "mold pull <model>".bold(),);
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_family_flux_contains_label() {
        let result = format_family_padded("flux", 10);
        assert!(result.contains("FLUX.1"));
    }

    #[test]
    fn format_family_sdxl_contains_label() {
        let result = format_family_padded("sdxl", 10);
        assert!(result.contains("SDXL"));
    }

    #[test]
    fn format_family_sd15_contains_label() {
        let result = format_family_padded("sd15", 10);
        assert!(result.contains("SD1.5"));
    }

    #[test]
    fn format_family_unknown_passthrough() {
        let result = format_family_padded("custom", 10);
        assert!(result.contains("custom"));
    }

    #[test]
    fn format_family_padded_respects_width() {
        // When colors are disabled (CI/NO_COLOR), the result is plain padded text.
        // When colors are enabled, ANSI codes make it longer.
        // Either way, the result should be at least `width` chars.
        let result = format_family_padded("sdxl", 20);
        assert!(
            result.len() >= 20,
            "result was only {} chars: {:?}",
            result.len(),
            result
        );
    }
}
