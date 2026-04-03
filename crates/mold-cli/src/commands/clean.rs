use std::time::Duration;

use anyhow::{bail, Result};
use colored::Colorize;
use mold_core::Config;

use super::cleanup::{
    clean_hf_cache, clean_orphaned_shared_files, clean_stale_pull_markers,
    count_hf_cache_transients, find_orphaned_shared_files, find_stale_pull_markers,
    is_path_older_than,
};
use crate::theme;
use crate::ui::format_bytes;

/// Parse a human-readable duration string like "30d", "7d", "24h", "12h", "60m".
pub fn parse_duration(s: &str) -> Result<Duration> {
    let s = s.trim();
    if s.is_empty() {
        bail!("empty duration string");
    }

    let (num_str, unit) = if let Some(prefix) = s.strip_suffix('d') {
        (prefix, 'd')
    } else if let Some(prefix) = s.strip_suffix('h') {
        (prefix, 'h')
    } else if let Some(prefix) = s.strip_suffix('m') {
        (prefix, 'm')
    } else {
        bail!(
            "invalid duration '{}': expected a suffix of 'd' (days), 'h' (hours), or 'm' (minutes)",
            s
        );
    };

    let n: u64 = num_str.parse().map_err(|_| {
        anyhow::anyhow!(
            "invalid duration '{}': '{}' is not a valid number",
            s,
            num_str
        )
    })?;

    if n == 0 {
        bail!("duration must be greater than zero");
    }

    let secs = match unit {
        'd' => n * 86400,
        'h' => n * 3600,
        'm' => n * 60,
        _ => unreachable!(),
    };

    Ok(Duration::from_secs(secs))
}

/// Scan output directory for images older than the given age.
/// Returns (count, bytes).
fn scan_old_output_images(dir: &std::path::Path, age: Duration) -> (u64, u64) {
    if !dir.is_dir() {
        return (0, 0);
    }
    let mut count = 0u64;
    let mut bytes = 0u64;
    for entry in walkdir::WalkDir::new(dir)
        .follow_links(false)
        .into_iter()
        .flatten()
    {
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path();
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            if !matches!(ext.to_ascii_lowercase().as_str(), "png" | "jpg" | "jpeg") {
                continue;
            }
        } else {
            continue;
        }
        if is_path_older_than(path, age) {
            count += 1;
            bytes += entry.metadata().map(|m| m.len()).unwrap_or(0);
        }
    }
    (count, bytes)
}

/// Delete output images older than the given age. Returns (count, bytes_freed).
fn remove_old_output_images(dir: &std::path::Path, age: Duration) -> (u64, u64) {
    if !dir.is_dir() {
        return (0, 0);
    }
    let mut count = 0u64;
    let mut bytes = 0u64;
    for entry in walkdir::WalkDir::new(dir)
        .follow_links(false)
        .into_iter()
        .flatten()
    {
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path();
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            if !matches!(ext.to_ascii_lowercase().as_str(), "png" | "jpg" | "jpeg") {
                continue;
            }
        } else {
            continue;
        }
        if is_path_older_than(path, age) {
            let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
            if std::fs::remove_file(path).is_ok() {
                count += 1;
                bytes += size;
            }
        }
    }
    (count, bytes)
}

fn format_duration_human(d: &Duration) -> String {
    let secs = d.as_secs();
    if secs >= 86400 && secs.is_multiple_of(86400) {
        let days = secs / 86400;
        format!("{} day{}", days, if days == 1 { "" } else { "s" })
    } else if secs >= 3600 && secs.is_multiple_of(3600) {
        let hours = secs / 3600;
        format!("{} hour{}", hours, if hours == 1 { "" } else { "s" })
    } else {
        let mins = secs / 60;
        format!("{} minute{}", mins, if mins == 1 { "" } else { "s" })
    }
}

pub fn run(force: bool, older_than: Option<&str>) -> Result<()> {
    let config = Config::load_or_default();

    let older_than_duration = older_than.map(parse_duration).transpose()?;

    // Scan for cleanable items
    let stale_markers = find_stale_pull_markers(&config);
    let (orphan_count, orphan_bytes) = find_orphaned_shared_files(&config);
    let (cache_count, cache_bytes) = count_hf_cache_transients(&config);

    let (output_count, output_bytes) = if let Some(age) = older_than_duration {
        let output_dir = config.effective_output_dir();
        scan_old_output_images(&output_dir, age)
    } else {
        (0, 0)
    };

    let total_items = stale_markers.len() as u64 + orphan_count + cache_count + output_count;

    if total_items == 0 {
        println!("{} Nothing to clean.", theme::icon_done());
        return Ok(());
    }

    let action = if force { "Removed" } else { "Would remove" };

    // Stale pull markers
    if !stale_markers.is_empty() {
        println!(
            "  {} {} stale pull marker{}:",
            action,
            stale_markers.len(),
            if stale_markers.len() == 1 { "" } else { "s" },
        );
        for name in &stale_markers {
            println!("    {}", name.dimmed());
        }
    }

    // Orphaned shared files
    if orphan_count > 0 {
        println!(
            "  {} {} orphaned shared file{} ({})",
            action,
            orphan_count,
            if orphan_count == 1 { "" } else { "s" },
            format_bytes(orphan_bytes),
        );
    }

    // hf-cache transients
    if cache_count > 0 {
        println!(
            "  {} {} hf-cache transient file{} ({})",
            action,
            cache_count,
            if cache_count == 1 { "" } else { "s" },
            format_bytes(cache_bytes),
        );
    }

    // Old output images
    if output_count > 0 {
        let age_label = format_duration_human(
            &older_than_duration.expect("duration set when output_count > 0"),
        );
        println!(
            "  {} {} output image{} older than {} ({})",
            action,
            output_count,
            if output_count == 1 { "" } else { "s" },
            age_label,
            format_bytes(output_bytes),
        );
    }

    let total_bytes = orphan_bytes + cache_bytes + output_bytes;
    println!();

    if force {
        // Actually clean
        clean_stale_pull_markers(&config);
        clean_orphaned_shared_files(&config);
        clean_hf_cache(&config);

        if let Some(age) = older_than_duration {
            let output_dir = config.effective_output_dir();
            remove_old_output_images(&output_dir, age);
        }

        println!(
            "{} Cleaned up {} (freed {})",
            theme::icon_done(),
            plural(total_items, "item"),
            format_bytes(total_bytes).bold(),
        );
    } else {
        println!("Total: {} would be freed", format_bytes(total_bytes).bold(),);
        println!();
        println!("Run with {} to delete.", "--force".bold(),);
    }

    Ok(())
}

fn plural(n: u64, word: &str) -> String {
    if n == 1 {
        format!("{n} {word}")
    } else {
        format!("{n} {word}s")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_duration_days() {
        let d = parse_duration("30d").unwrap();
        assert_eq!(d, Duration::from_secs(30 * 86400));
    }

    #[test]
    fn parse_duration_hours() {
        let d = parse_duration("24h").unwrap();
        assert_eq!(d, Duration::from_secs(24 * 3600));
    }

    #[test]
    fn parse_duration_minutes() {
        let d = parse_duration("60m").unwrap();
        assert_eq!(d, Duration::from_secs(60 * 60));
    }

    #[test]
    fn parse_duration_single_day() {
        let d = parse_duration("1d").unwrap();
        assert_eq!(d, Duration::from_secs(86400));
    }

    #[test]
    fn parse_duration_with_whitespace() {
        let d = parse_duration("  7d  ").unwrap();
        assert_eq!(d, Duration::from_secs(7 * 86400));
    }

    #[test]
    fn parse_duration_invalid_no_suffix() {
        assert!(parse_duration("30").is_err());
    }

    #[test]
    fn parse_duration_invalid_not_a_number() {
        assert!(parse_duration("abcd").is_err());
    }

    #[test]
    fn parse_duration_zero_rejected() {
        assert!(parse_duration("0d").is_err());
    }

    #[test]
    fn parse_duration_empty_rejected() {
        assert!(parse_duration("").is_err());
    }

    #[test]
    fn format_duration_human_days() {
        assert_eq!(format_duration_human(&Duration::from_secs(86400)), "1 day");
        assert_eq!(
            format_duration_human(&Duration::from_secs(30 * 86400)),
            "30 days"
        );
    }

    #[test]
    fn format_duration_human_hours() {
        assert_eq!(format_duration_human(&Duration::from_secs(3600)), "1 hour");
        assert_eq!(
            format_duration_human(&Duration::from_secs(12 * 3600)),
            "12 hours"
        );
    }

    #[test]
    fn format_duration_human_minutes() {
        assert_eq!(format_duration_human(&Duration::from_secs(60)), "1 minute");
        assert_eq!(
            format_duration_human(&Duration::from_secs(45 * 60)),
            "45 minutes"
        );
    }

    #[test]
    fn plural_singular() {
        assert_eq!(plural(1, "item"), "1 item");
    }

    #[test]
    fn plural_multiple() {
        assert_eq!(plural(5, "item"), "5 items");
    }

    #[test]
    fn scan_old_output_images_nonexistent_dir() {
        let (count, bytes) =
            scan_old_output_images(std::path::Path::new("/nonexistent"), Duration::from_secs(1));
        assert_eq!(count, 0);
        assert_eq!(bytes, 0);
    }
}
