use anyhow::Result;
use colored::Colorize;

use crate::control::{CliContext, ModelCatalogSource};
use crate::output::colorize_description;
use crate::theme;
use crate::ui::{col_width, family_label, format_disk_size, format_family_padded};

fn description_with_gated(desc: &str, model_name: &str) -> String {
    let is_gated = mold_core::manifest::find_manifest(model_name)
        .map(|m| m.is_gated())
        .unwrap_or(false);
    if is_gated && !desc.contains("[gated]") {
        format!("{desc} [gated]")
    } else {
        desc.to_string()
    }
}

fn format_fetch_size(remaining_bytes: u64) -> String {
    if remaining_bytes == 0 {
        format!("{:>7}", "cached").dimmed().to_string()
    } else {
        format!(
            "{:>7}",
            format!("{:.1}GB", remaining_bytes as f64 / 1_073_741_824.0)
        )
    }
}

pub async fn run() -> Result<()> {
    let ctx = CliContext::new(None);
    let default_model =
        mold_core::manifest::resolve_model_name(&ctx.config().resolved_default_model());

    match ctx.list_models().await? {
        ModelCatalogSource::Remote(models) => {
            let downloaded: Vec<_> = models.iter().filter(|m| m.downloaded).collect();
            let available: Vec<_> = models.iter().filter(|m| !m.downloaded).collect();

            // Compute column widths across all models (account for ● and ★ indicators)
            let nw = col_width(
                models.iter().map(|m| {
                    m.name.len()
                        + if m.is_loaded { 2 } else { 0 }
                        + if m.name == default_model { 2 } else { 0 }
                }),
                4, // "NAME"
                2,
            );
            let fw = col_width(
                models.iter().map(|m| family_label(&m.family).len()),
                6, // "FAMILY"
                2,
            );

            if downloaded.is_empty() {
                println!("{} No models downloaded.", theme::icon_neutral());
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

                for model in &downloaded {
                    let is_default = model.name == default_model;
                    // Pad plain text first (ANSI codes break {:<N} width).
                    // Then colorize indicators after padding.
                    let plain_label = match (model.is_loaded, is_default) {
                        (true, true) => format!("{} ● ★", model.name),
                        (true, false) => format!("{} ●", model.name),
                        (false, true) => format!("{} ★", model.name),
                        (false, false) => model.name.clone(),
                    };
                    let padded = format!("{:<nw$}", plain_label, nw = nw);
                    let name = if is_default {
                        // Replace the ★ with a yellow-colored version after padding
                        let colored = padded.replace("★", &"★".yellow().bold().to_string());
                        if model.is_loaded {
                            colored.replace(&model.name, &model.name.green().to_string())
                        } else {
                            colored
                        }
                    } else if model.is_loaded {
                        padded.green().to_string()
                    } else {
                        padded
                    };
                    let size = if let Some(mf) = mold_core::manifest::find_manifest(&model.name) {
                        format!("{:.1}GB", mf.model_size_gb())
                    } else if model.size_gb > 0.0 {
                        format!("{:.1}GB", model.size_gb)
                    } else {
                        "—".to_string()
                    };
                    let disk = model
                        .disk_usage_bytes
                        .map(format_disk_size)
                        .unwrap_or_else(|| "—".to_string());
                    println!(
                        "{} {} {:>7}  {:>7}  {:<7} {:<9} {:<8} {:<7} {}",
                        name,
                        format_family_padded(&model.family, fw),
                        size,
                        disk,
                        model.defaults.default_steps,
                        format!("{:.1}", model.defaults.default_guidance),
                        model.defaults.default_width,
                        model.defaults.default_height,
                        colorize_description(&description_with_gated(
                            mold_core::manifest::find_manifest(&model.name)
                                .map(|m| m.description.as_str())
                                .unwrap_or(&model.defaults.description),
                            &model.name,
                        )),
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
                    let (size_str, fetch_col) =
                        if let Some(mf) = mold_core::manifest::find_manifest(&m.name) {
                            let model_gb = mf.model_size_gb() as f64;
                            let remaining_bytes = m.remaining_download_bytes.unwrap_or_else(|| {
                                let (_, remaining_bytes) =
                                    mold_core::manifest::compute_download_size(mf);
                                remaining_bytes
                            });
                            let fetch = format_fetch_size(remaining_bytes);
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
                        colorize_description(&description_with_gated(
                            &m.defaults.description,
                            &m.name
                        ),),
                        nw = nw,
                    );
                }
                println!();
                println!("{}", "Use mold pull <model> to download.".dimmed());
            }
            let has_gated = models.iter().any(|m| {
                mold_core::manifest::find_manifest(&m.name)
                    .map(|mf| mf.is_gated())
                    .unwrap_or(false)
            });
            if has_gated {
                println!();
                println!(
                    "{}",
                    "Models marked [gated] require a HuggingFace token to download.".dimmed()
                );
                println!(
                    "{}",
                    "Set HF_TOKEN in your environment before running mold pull.".dimmed()
                );
            }
        }
        ModelCatalogSource::Local(models) => {
            let config = ctx.config();

            let downloaded: Vec<_> = models.iter().filter(|m| m.downloaded).collect();
            let available: Vec<_> = models.iter().filter(|m| !m.downloaded).collect();

            // Compute name column width across both installed and available models
            let nw = col_width(
                downloaded
                    .iter()
                    .map(|m| m.name.len() + if m.name == default_model { 2 } else { 0 })
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
                println!("{} No models configured.", theme::icon_neutral());
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
                    let is_default = model.name == default_model;
                    let display_name = if is_default {
                        let padded = format!("{:<nw$}", format!("{} ★", model.name), nw = nw);
                        padded.replace("★", &"★".yellow().bold().to_string())
                    } else {
                        format!("{:<nw$}", model.name, nw = nw)
                    };
                    let name = &model.name;
                    let mcfg = config.model_config(name);
                    let family_raw = &model.family;
                    let size = mold_core::manifest::find_manifest(name)
                        .map(|m| format!("{:.1}GB", m.model_size_gb()))
                        .unwrap_or_else(|| "—".to_string());
                    let model_paths = mcfg.all_file_paths();
                    let disk_bytes: u64 = model.disk_usage_bytes.unwrap_or_else(|| {
                        model_paths
                            .iter()
                            .filter_map(|p| std::fs::metadata(p).ok())
                            .map(|m| m.len())
                            .sum()
                    });
                    all_unique_paths.extend(model_paths);
                    let disk = format_disk_size(disk_bytes);
                    println!(
                        "{} {} {:>7}  {:>7}  {:<7} {:<9} {:<8} {:<7} {}",
                        display_name,
                        format_family_padded(family_raw, fw),
                        size,
                        disk,
                        model.defaults.default_steps,
                        format!("{:.1}", model.defaults.default_guidance),
                        model.defaults.default_width,
                        model.defaults.default_height,
                        colorize_description(&description_with_gated(
                            mold_core::manifest::find_manifest(name)
                                .map(|m| m.description.as_str())
                                .unwrap_or(&model.defaults.description),
                            name,
                        )),
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
                    let remaining_bytes = m.remaining_download_bytes.unwrap_or_else(|| {
                        let (_, remaining_bytes) =
                            mold_core::manifest::compute_download_size(manifest);
                        remaining_bytes
                    });
                    let model_gb = manifest.model_size_gb() as f64;
                    let size_str = format!("{:.1}GB", model_gb);
                    let fetch_col = format_fetch_size(remaining_bytes);
                    println!(
                        "  {:<nw$} {} {:>7}  {:>7}  {}",
                        m.name.bold(),
                        format_family_padded(&m.family, fw),
                        size_str,
                        fetch_col,
                        colorize_description(&description_with_gated(
                            &m.defaults.description,
                            &m.name
                        ),),
                        nw = nw,
                    );
                }
                println!();
                println!("Use {} to download.", "mold pull <model>".bold());
            }

            let has_gated = models.iter().any(|m| {
                mold_core::manifest::find_manifest(&m.name)
                    .map(|mf| mf.is_gated())
                    .unwrap_or(false)
            });
            if has_gated {
                println!();
                println!(
                    "{}",
                    "Models marked [gated] require a HuggingFace token to download.".dimmed()
                );
                println!(
                    "{}",
                    "Set HF_TOKEN in your environment before running mold pull.".dimmed()
                );
            }
        }
    }

    Ok(())
}
