use anyhow::Result;
use colored::Colorize;
use mold_core::{build_model_catalog, Config, MoldClient};

use crate::output::colorize_description;
use crate::ui::{family_label, format_family_padded};

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
            let downloaded: Vec<_> = models.iter().filter(|m| m.downloaded).collect();
            let available: Vec<_> = models.iter().filter(|m| !m.downloaded).collect();

            // Compute column widths across all models
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

            if downloaded.is_empty() {
                println!("{} No models downloaded.", "●".dimmed());
            } else {
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

                for model in &downloaded {
                    // Pad plain text first, then colorize — ANSI escapes break `{:<N}`.
                    let name = if model.is_loaded {
                        format!("{:<nw$}", format!("{} ●", model.name), nw = nw)
                            .green()
                            .to_string()
                    } else {
                        format!("{:<nw$}", model.name, nw = nw)
                    };
                    let size = if let Some(mf) = mold_core::manifest::find_manifest(&model.name) {
                        format!("{:.1}GB", mf.model_size_gb())
                    } else if model.size_gb > 0.0 {
                        format!("{:.1}GB", model.size_gb)
                    } else {
                        "—".to_string()
                    };
                    println!(
                        "{} {} {:>7}  {:<7} {:<9} {:<8} {:<7} {}",
                        name,
                        format_family_padded(&model.family, fw),
                        size,
                        model.defaults.default_steps,
                        format!("{:.1}", model.defaults.default_guidance),
                        model.defaults.default_width,
                        model.defaults.default_height,
                        colorize_description(
                            mold_core::manifest::find_manifest(&model.name)
                                .map(|m| m.description.as_str())
                                .unwrap_or(&model.defaults.description),
                        ),
                    );
                }
            }

            // Show available-to-pull models
            if !available.is_empty() {
                println!();
                println!(
                    "  {:<nw$} {:<fw$} {:>7}  {:>7}",
                    "Available to pull:".bold(),
                    "",
                    "SIZE".dimmed(),
                    "FETCH".dimmed(),
                    nw = nw,
                    fw = fw,
                );
                for m in &available {
                    let (size_str, fetch_col) = if let Some(mf) =
                        mold_core::manifest::find_manifest(&m.name)
                    {
                        let (_, remaining_bytes) = mold_core::manifest::compute_download_size(mf);
                        let model_gb = mf.model_size_gb() as f64;
                        let remaining_gb = remaining_bytes as f64 / 1_073_741_824.0;
                        let fetch = if remaining_bytes == 0 {
                            format!("{:>7}", "cached").dimmed().to_string()
                        } else {
                            format!("{:.1}GB", remaining_gb)
                        };
                        (format!("{:.1}GB", model_gb), fetch)
                    } else {
                        let s = if m.size_gb > 0.0 {
                            format!("{:.1}GB", m.size_gb)
                        } else {
                            "—".to_string()
                        };
                        (s.clone(), s)
                    };
                    println!(
                        "  {:<nw$} {} {:>7}  {:>7}  {}",
                        m.name.bold(),
                        format_family_padded(&m.family, fw),
                        size_str,
                        fetch_col,
                        colorize_description(&m.defaults.description),
                        nw = nw,
                    );
                }
                println!();
                println!("{}", "Use mold pull <model> to download.".dimmed());
            }
        }
        Err(_) => {
            let config = Config::load_or_default();
            let models = build_model_catalog(&config, None, false);

            let downloaded: Vec<_> = models.iter().filter(|m| m.downloaded).collect();
            let available: Vec<_> = models.iter().filter(|m| !m.downloaded).collect();

            // Compute name column width across both installed and available models
            let nw = col_width(
                downloaded
                    .iter()
                    .map(|m| m.name.len())
                    .chain(available.iter().map(|m| m.name.len())),
                4, // "NAME"
                2,
            );
            let fw = col_width(
                downloaded
                    .iter()
                    .map(|m| family_label(&m.family).len())
                    .chain(available.iter().map(|m| family_label(&m.family).len())),
                6, // "FAMILY"
                2,
            );

            if downloaded.is_empty() {
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
                for model in &downloaded {
                    let name = &model.name;
                    let mcfg = config.model_config(name);
                    let family_raw = &model.family;
                    let size = mold_core::manifest::find_manifest(name)
                        .map(|m| format!("{:.1}GB", m.model_size_gb()))
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
                        model.defaults.default_steps,
                        format!("{:.1}", model.defaults.default_guidance),
                        model.defaults.default_width,
                        model.defaults.default_height,
                        colorize_description(
                            // Prefer manifest description (has [alpha]/[beta] tags)
                            // over config description (may be stale from older pull)
                            mold_core::manifest::find_manifest(name)
                                .map(|m| m.description.as_str())
                                .unwrap_or(&model.defaults.description),
                        ),
                        nw = nw,
                    );
                }
                let total_disk: u64 = all_unique_paths
                    .iter()
                    .filter_map(|p| std::fs::metadata(p).ok())
                    .map(|m| m.len())
                    .sum();
                if downloaded.len() > 1 && total_disk > 0 {
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
                println!(
                    "  {:<nw$} {:<fw$} {:>7}  {:>7}",
                    "Available to pull:".bold(),
                    "",
                    "SIZE".dimmed(),
                    "FETCH".dimmed(),
                    nw = nw,
                    fw = fw,
                );
                for m in &available {
                    let manifest = mold_core::manifest::find_manifest(&m.name)
                        .expect("available models should come from the manifest");
                    let (_, remaining_bytes) = mold_core::manifest::compute_download_size(manifest);
                    let model_gb = manifest.model_size_gb() as f64;
                    let remaining_gb = remaining_bytes as f64 / 1_073_741_824.0;
                    let size_str = format!("{:.1}GB", model_gb);
                    let fetch_col = if remaining_bytes == 0 {
                        format!("{:>7}", "cached").dimmed().to_string()
                    } else {
                        format!("{:.1}GB", remaining_gb)
                    };
                    println!(
                        "  {:<nw$} {} {:>7}  {:>7}  {}",
                        m.name.bold(),
                        format_family_padded(&m.family, fw),
                        size_str,
                        fetch_col,
                        colorize_description(&m.defaults.description),
                        nw = nw,
                    );
                }
                println!();
                println!("Use {} to download.", "mold pull <model>".bold());
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::ui;

    #[test]
    fn format_family_flux_contains_label() {
        let result = ui::format_family_padded("flux", 10);
        assert!(result.contains("FLUX.1"));
    }
}
