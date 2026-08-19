//! `mold trash` — inspect and act on the gallery trash of a running server.
//!
//! Every subcommand talks HTTP to `$MOLD_HOST` (with `MOLD_API_KEY` when
//! configured). There is deliberately no local fallback: the trash is a
//! property of the serving host's gallery (`<output_dir>/.trash/` plus the
//! `generations.trashed_at_ms` flag in that host's `mold.db`), so an
//! unreachable server is reported as such rather than silently acting on a
//! different gallery.

use anyhow::{bail, Context, Result};
use colored::Colorize;
use mold_core::{GalleryImage, MoldClient};

use crate::TrashAction;

/// Gallery view name the server uses for trashed prints.
const TRASH_VIEW: &str = "trash";

pub async fn run(action: TrashAction) -> Result<()> {
    let client = MoldClient::from_env();
    match action {
        TrashAction::List { json } => trash_list(&client, json).await,
        TrashAction::Restore { filenames } => trash_restore(&client, &filenames).await,
        TrashAction::Empty { yes } => trash_empty(&client, yes).await,
        TrashAction::Sweep => trash_sweep(&client).await,
    }
}

async fn fetch_trash(client: &MoldClient) -> Result<Vec<GalleryImage>> {
    client
        .list_gallery_view(TRASH_VIEW)
        .await
        .with_context(|| format!("could not list the trash on {}", client.host()))
}

async fn trash_list(client: &MoldClient, json: bool) -> Result<()> {
    let rows = fetch_trash(client).await?;
    if json {
        println!("{}", serde_json::to_string_pretty(&rows)?);
        return Ok(());
    }
    let now = now_secs();
    print!("{}", render_listing(&rows, now));
    Ok(())
}

async fn trash_restore(client: &MoldClient, filenames: &[String]) -> Result<()> {
    if filenames.is_empty() {
        bail!("no filenames given; run `mold trash list` to see what can be restored");
    }
    client
        .restore_trashed(filenames)
        .await
        .with_context(|| format!("restore failed on {}", client.host()))?;
    for name in filenames {
        println!("{} {}", "restored".green(), name);
    }
    Ok(())
}

async fn trash_empty(client: &MoldClient, yes: bool) -> Result<()> {
    if !yes {
        let rows = fetch_trash(client).await?;
        if rows.is_empty() {
            println!("Trash on {} is already empty.", client.host());
            return Ok(());
        }
        if !confirm_empty(rows.len(), client.host())? {
            bail!("empty aborted");
        }
    }
    let result = client
        .empty_trash()
        .await
        .with_context(|| format!("empty trash failed on {}", client.host()))?;
    println!("{} purged={}", "trash emptied".green(), result.purged);
    Ok(())
}

async fn trash_sweep(client: &MoldClient) -> Result<()> {
    let result = client
        .sweep_trash()
        .await
        .with_context(|| format!("sweep failed on {}", client.host()))?;
    println!(
        "{} purged={} remaining={}",
        "sweep complete".green(),
        result.purged,
        result.remaining
    );
    Ok(())
}

fn confirm_empty(count: usize, host: &str) -> Result<bool> {
    use std::io::{self, Write};
    eprint!(
        "Permanently delete {count} trashed print{} on {host}? This cannot be undone. [y/N] ",
        if count == 1 { "" } else { "s" }
    );
    io::stderr().flush().ok();
    let mut line = String::new();
    io::stdin().read_line(&mut line)?;
    Ok(matches!(
        line.trim().to_ascii_lowercase().as_str(),
        "y" | "yes"
    ))
}

fn now_secs() -> u64 {
    mold_core::time::now_epoch_ms_u64() / 1000
}

/// Render the human-readable trash table. Pure so the layout is testable.
///
/// Columns: filename, title (or `—`), how long ago the print was trashed,
/// when the sweeper will purge it (`kept` under keep-forever retention), and
/// the on-disk size.
pub(crate) fn render_listing(rows: &[GalleryImage], now_secs: u64) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    if rows.is_empty() {
        out.push_str("Trash is empty.\n");
        return out;
    }
    let name_width = rows
        .iter()
        .map(|row| row.filename.chars().count())
        .max()
        .unwrap_or(8)
        .clamp(8, 60);
    let _ = writeln!(
        out,
        "{:<name_width$} {:<24} {:<10} {:<10} {:>9}",
        "FILENAME".bold(),
        "TITLE".bold(),
        "TRASHED".bold(),
        "PURGES".bold(),
        "SIZE".bold(),
    );
    let _ = writeln!(out, "{}", "─".repeat(name_width + 58).dimmed());
    for row in rows {
        let _ = writeln!(
            out,
            "{:<name_width$} {:<24} {:<10} {:<10} {:>9}",
            truncate(&row.filename, name_width),
            truncate(display_title(row), 24),
            row.trashed_at
                .map(|ts| relative_past(now_secs, ts))
                .unwrap_or_else(|| "—".to_string()),
            purge_label(now_secs, row.purge_at),
            row.size_bytes
                .map(mold_core::format::human_bytes_compact)
                .unwrap_or_else(|| "—".to_string()),
        );
    }
    out
}

/// The title shown in the table: the editable row title, else the title
/// recorded at creation, else `—`.
fn display_title(row: &GalleryImage) -> &str {
    row.title
        .as_deref()
        .or(row.metadata.title.as_deref())
        .filter(|title| !title.trim().is_empty())
        .unwrap_or("—")
}

/// "3h ago" style label for a past Unix timestamp.
pub(crate) fn relative_past(now_secs: u64, then_secs: u64) -> String {
    let elapsed = now_secs.saturating_sub(then_secs);
    if elapsed < 60 {
        "just now".to_string()
    } else if elapsed < 3_600 {
        format!("{}m ago", elapsed / 60)
    } else if elapsed < 86_400 {
        format!("{}h ago", elapsed / 3_600)
    } else {
        format!("{}d ago", elapsed / 86_400)
    }
}

/// "in 27d" style label for the purge deadline; `kept` when retention is
/// keep-forever (no deadline), `due` once the deadline has passed and the
/// next sweep will purge it.
pub(crate) fn purge_label(now_secs: u64, purge_at: Option<u64>) -> String {
    let Some(purge_at) = purge_at else {
        return "kept".to_string();
    };
    if purge_at <= now_secs {
        return "due".to_string();
    }
    let remaining = purge_at - now_secs;
    if remaining < 3_600 {
        format!("in {}m", (remaining / 60).max(1))
    } else if remaining < 86_400 {
        format!("in {}h", remaining / 3_600)
    } else {
        format!("in {}d", remaining / 86_400)
    }
}

fn truncate(value: &str, max: usize) -> String {
    if value.chars().count() <= max {
        value.to_string()
    } else {
        let suffix = "...";
        let keep = max.saturating_sub(suffix.len());
        let mut out = value.chars().take(keep).collect::<String>();
        out.push_str(suffix);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(
        filename: &str,
        title: Option<&str>,
        trashed_at: u64,
        purge_at: Option<u64>,
    ) -> GalleryImage {
        let mut image: GalleryImage = serde_json::from_value(serde_json::json!({
            "filename": filename,
            "metadata": {
                "prompt": "a print",
                "model": "flux-dev:q4",
                "seed": 1,
                "steps": 4,
                "guidance": 1.0,
                "width": 8,
                "height": 8,
                "version": "test"
            },
            "timestamp": trashed_at,
            "size_bytes": 2_048_u64
        }))
        .unwrap();
        image.title = title.map(str::to_string);
        image.trashed_at = Some(trashed_at);
        image.purge_at = purge_at;
        image
    }

    #[test]
    fn relative_past_buckets_minutes_hours_and_days() {
        assert_eq!(relative_past(1_000, 990), "just now");
        assert_eq!(relative_past(1_000, 1_000 - 5 * 60), "5m ago");
        assert_eq!(relative_past(100_000, 100_000 - 3 * 3_600), "3h ago");
        assert_eq!(relative_past(1_000_000, 1_000_000 - 2 * 86_400), "2d ago");
        // A clock that runs behind the server never underflows.
        assert_eq!(relative_past(10, 20), "just now");
    }

    #[test]
    fn purge_label_reports_kept_due_and_countdowns() {
        assert_eq!(purge_label(1_000, None), "kept");
        assert_eq!(purge_label(1_000, Some(1_000)), "due");
        assert_eq!(purge_label(1_000, Some(900)), "due");
        assert_eq!(purge_label(1_000, Some(1_000 + 30)), "in 1m");
        assert_eq!(purge_label(1_000, Some(1_000 + 5 * 3_600)), "in 5h");
        assert_eq!(purge_label(1_000, Some(1_000 + 27 * 86_400)), "in 27d");
    }

    #[test]
    fn listing_has_the_five_columns_and_one_line_per_print() {
        colored::control::set_override(false);
        let now = 2_000_000;
        let rows = vec![
            row(
                "mold-flux-dev-q4-1~smurf-village.png",
                Some("Smurf village"),
                now - 3 * 3_600,
                Some(now + 27 * 86_400),
            ),
            row("mold-flux-dev-q4-2.png", None, now - 2 * 86_400, None),
        ];
        let text = render_listing(&rows, now);
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 4, "header, rule, two rows: {text}");
        for column in ["FILENAME", "TITLE", "TRASHED", "PURGES", "SIZE"] {
            assert!(lines[0].contains(column), "missing {column}: {}", lines[0]);
        }
        assert!(lines[2].contains("mold-flux-dev-q4-1~smurf-village.png"));
        assert!(lines[2].contains("Smurf village"));
        assert!(lines[2].contains("3h ago"));
        assert!(lines[2].contains("in 27d"));
        assert!(lines[2].contains("2.0K"), "size column: {}", lines[2]);
        assert!(
            lines[3].contains("—"),
            "untitled shows a dash: {}",
            lines[3]
        );
        assert!(lines[3].contains("2d ago"));
        assert!(lines[3].contains("kept"));
    }

    #[test]
    fn listing_falls_back_to_the_creation_title_and_reports_empty() {
        colored::control::set_override(false);
        let mut untitled = row("a.png", None, 10, None);
        untitled.metadata.title = Some("From metadata".into());
        let text = render_listing(&[untitled], 20);
        assert!(text.contains("From metadata"), "{text}");
        assert_eq!(render_listing(&[], 20), "Trash is empty.\n");
    }
}
