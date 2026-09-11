//! `mold downloads` — the model download queue of one running server.
//!
//! Remote only, and deliberately so: a download queue is a property of the
//! machine whose disk fills up, so an unreachable server is reported rather
//! than silently answered from this one. `mold pull` remains the command that
//! also works with no server at all.
//!
//! `POST /api/downloads` takes MANIFEST names and
//! `POST /api/catalog/:id/download` takes `cv:` / `hf:` ids, each refusing the
//! other's shape. `mold downloads add` therefore refuses a catalog id here
//! and points at `mold pull`, which routes on the id itself.

use anyhow::{bail, Result};
use colored::Colorize;
use mold_core::types::{DownloadEvent, DownloadJob, DownloadsListing, JobStatus};
use mold_core::MoldClient;

use crate::theme;
use crate::ui::{col_width, format_disk_size};
use crate::DownloadsAction;

pub async fn run(host: Option<&str>, action: DownloadsAction) -> Result<()> {
    let client = crate::control::client_for_host(host);
    match action {
        DownloadsAction::List { json } => list(&client, json).await,
        DownloadsAction::Add {
            model,
            accept_license,
        } => add(&client, &model, &accept_license).await,
        DownloadsAction::Cancel { id } => cancel(&client, &id).await,
        DownloadsAction::Watch { json } => watch(&client, json).await,
    }
}

/// The refusal a catalog id gets from this door.
///
/// The two download doors reject each other's id shapes, so routing on the id
/// is the caller's job. `mold pull` already does exactly that — it resolves
/// `cv:` and `hf:` through the catalog and everything else through the
/// manifest — so the answer is a command, not a different flag.
pub fn catalog_id_refusal(model: &str) -> String {
    format!(
        "'{model}' is a catalog id, and the download queue takes model names from `mold list`. \
         Install it with `mold pull {model}`, which resolves the catalog entry and its companions."
    )
}

async fn add(client: &MoldClient, model: &str, accept_license: &[String]) -> Result<()> {
    if model.starts_with("cv:") || model.starts_with("hf:") {
        bail!("{}", catalog_id_refusal(model));
    }
    let accepted = accept_licenses_on_the_server(client, accept_license).await?;
    let enqueued = client.create_download(model, &accepted).await?;
    if enqueued.already_present {
        println!(
            "{} {model} is already in the queue as {} (position {})",
            theme::icon_neutral(),
            enqueued.response.id,
            enqueued.response.position
        );
        return Ok(());
    }
    println!(
        "{} queued {model} as {} ({})",
        theme::icon_ok(),
        enqueued.response.id,
        if enqueued.response.position == 0 {
            "starting now".to_string()
        } else {
            format!("position {}", enqueued.response.position)
        }
    );
    println!("  {} mold downloads watch", "next".dimmed());
    Ok(())
}

/// Resolve `--accept-license` ids against the terms held by the machine that
/// will RECORD the acceptance.
///
/// That machine is always the server here, because the download queue only
/// exists there — so unlike `mold pull` there is no local-fallback branch and
/// no case in which this build's own pinned terms are the right thing to
/// show. Displaying our revision and letting a server on another release
/// resolve the bare id to its own would record consent to text the user never
/// read.
async fn accept_licenses_on_the_server(
    client: &MoldClient,
    ids: &[String],
) -> Result<Vec<mold_core::LicenseAcceptance>> {
    if ids.is_empty() {
        return Ok(Vec::new());
    }
    let capabilities = client.capabilities().await?;
    let terms = if capabilities.licenses {
        Some(client.list_licenses().await?)
    } else {
        None
    };
    let mut accepted = Vec::with_capacity(ids.len());
    for id in ids {
        accepted.push(
            crate::commands::pull::resolve_and_show_license(id, terms.as_deref())?.acceptance,
        );
    }
    Ok(accepted)
}

/// Refuse an empty download id by name.
///
/// The same trap `mold mesh-workflow` has: a blank path segment resolves to
/// the collection route, so the verb is answered by something that is not the
/// job it was given, and the user reads an error about the answer rather than
/// about the blank argument.
pub fn require_download_id(id: &str) -> Result<&str> {
    let trimmed = id.trim();
    if trimmed.is_empty() {
        bail!("a download id is required; `mold downloads list` shows the ids on this machine");
    }
    Ok(trimmed)
}

async fn cancel(client: &MoldClient, id: &str) -> Result<()> {
    let id = require_download_id(id)?;
    client.cancel_download(id).await?;
    println!("{} cancelled {id}", theme::icon_ok());
    Ok(())
}

async fn list(client: &MoldClient, json: bool) -> Result<()> {
    let listing = client.list_downloads().await?;
    record_ids_for_completion(client.host(), &listing);
    if json {
        println!("{}", serde_json::to_string_pretty(&listing)?);
        return Ok(());
    }
    print_listing(&listing, client.host());
    Ok(())
}

/// Teach the shell the ids this listing just showed, so `downloads cancel`
/// completes.
///
/// A completer cannot ask a server (see `crate::completion_cache`), so the
/// command that already fetched the queue records what it saw — including the
/// machine that answered, which is only a completion candidate because the
/// request succeeded.
fn record_ids_for_completion(host: &str, listing: &DownloadsListing) {
    crate::completion_cache::record_reached_host(host, |cache| {
        cache.record_download_ids(
            listing
                .active_jobs
                .iter()
                .chain(listing.queued.iter())
                .chain(listing.history.iter())
                .map(|job| job.id.clone()),
        );
    });
}

/// Every row the listing knows about, in the order they matter: what is
/// transferring, what is waiting, and what recently finished.
///
/// `active_jobs` is the field new clients read; `active` is a compatibility
/// view of its first entry and would under-report a host running two.
fn print_listing(listing: &DownloadsListing, host: &str) {
    let rows: Vec<&DownloadJob> = listing
        .active_jobs
        .iter()
        .chain(listing.queued.iter())
        .chain(listing.history.iter())
        .collect();
    if rows.is_empty() {
        println!("{} Nothing downloading on {host}.", theme::icon_neutral());
        return;
    }
    let id_width = col_width(rows.iter().map(|job| job.id.len()), 2, 2);
    let model_width = col_width(rows.iter().map(|job| job.model.len()), 5, 2);
    println!(
        "{:<id_width$} {:<model_width$} {:<10} {:>9} {:>7}  {}",
        "ID".bold(),
        "MODEL".bold(),
        "STATE".bold(),
        "DONE".bold(),
        "FILES".bold(),
        "DETAIL".bold(),
    );
    println!("{}", "─".repeat(id_width + model_width + 40).dimmed());
    for job in rows {
        println!(
            "{:<id_width$} {:<model_width$} {} {:>9} {:>7}  {}",
            job.id,
            job.model,
            colored_status(job.status, 10),
            progress_fraction(job),
            format!("{}/{}", job.files_done, job.files_total),
            detail(job),
        );
    }
}

fn detail(job: &DownloadJob) -> String {
    if let Some(error) = &job.error {
        return error.red().to_string();
    }
    job.current_file.clone().unwrap_or_default()
}

fn progress_fraction(job: &DownloadJob) -> String {
    if job.bytes_total == 0 {
        return "—".to_string();
    }
    format!(
        "{:.0}%",
        (job.bytes_done as f64 / job.bytes_total as f64) * 100.0
    )
}

fn colored_status(status: JobStatus, width: usize) -> String {
    let text = match status {
        JobStatus::Queued => "queued",
        JobStatus::Active => "active",
        JobStatus::Completed => "completed",
        JobStatus::Failed => "failed",
        JobStatus::Cancelled => "cancelled",
    };
    // Pad the plain text first; ANSI codes break `{:<N}`.
    let padded = format!("{text:<width$}");
    match status {
        JobStatus::Completed => padded.green().to_string(),
        JobStatus::Failed => padded.red().to_string(),
        JobStatus::Cancelled => padded.yellow().to_string(),
        _ => padded,
    }
}

/// Follow the queue until the stream ends or the user stops it.
///
/// The server subscribes before it snapshots, so the first frame is always a
/// full listing — a watcher that attaches mid-download still knows what is
/// running rather than waiting for the next delta.
async fn watch(client: &MoldClient, json: bool) -> Result<()> {
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let host = client.host().to_string();
    let printer = tokio::spawn(async move {
        let mut rates = DownloadRates::default();
        while let Some(event) = rx.recv().await {
            // The server subscribes before it snapshots, so the first frame
            // is always a full listing — the same rows `downloads list`
            // shows, and the same ids `downloads cancel` completes from.
            // Recorded above the `--json` branch, because a watcher piping
            // JSON still just learned what is on that machine.
            if let DownloadEvent::Snapshot { listing } = &event {
                record_ids_for_completion(&host, listing);
            }
            if json {
                if let Ok(line) = serde_json::to_string(&event) {
                    println!("{line}");
                }
                continue;
            }
            if let Some(line) = rates.describe(&event, &host) {
                println!("{line}");
            }
        }
    });
    client.stream_downloads(tx).await?;
    let _ = printer.await;
    Ok(())
}

/// Per-job transfer rates, so a progress line can report speed and an ETA.
///
/// The estimate is smoothed over a window for the reason
/// `crate::ui::SmoothedRate` exists at all: download events arrive coarsely,
/// and an instantaneous rate computed across a zero-progress gap oscillates
/// wildly.
#[derive(Default)]
struct DownloadRates {
    rates: std::collections::HashMap<String, crate::ui::SmoothedRate>,
    totals: std::collections::HashMap<String, u64>,
}

impl DownloadRates {
    fn describe(&mut self, event: &DownloadEvent, host: &str) -> Option<String> {
        match event {
            DownloadEvent::Snapshot { listing } => {
                for job in listing.active_jobs.iter().chain(listing.queued.iter()) {
                    self.totals.insert(job.id.clone(), job.bytes_total);
                }
                let active = listing.active_jobs.len();
                let queued = listing.queued.len();
                Some(format!(
                    "{} {host}: {active} transferring, {queued} waiting",
                    theme::icon_neutral()
                ))
            }
            DownloadEvent::Enqueued {
                id,
                model,
                position,
            } => Some(format!(
                "{} queued {model} as {id} (position {position})",
                theme::icon_neutral()
            )),
            DownloadEvent::Dequeued { id } => {
                Some(format!("{} removed {id}", theme::icon_neutral()))
            }
            DownloadEvent::Started {
                id,
                files_total,
                bytes_total,
            } => {
                self.totals.insert(id.clone(), *bytes_total);
                Some(format!(
                    "{} started {id}: {files_total} file{}, {}",
                    theme::icon_neutral(),
                    if *files_total == 1 { "" } else { "s" },
                    format_disk_size(*bytes_total)
                ))
            }
            DownloadEvent::Progress {
                id,
                files_done,
                bytes_done,
                current_file,
            } => {
                let total = self.totals.get(id).copied().unwrap_or(0);
                let rate = self
                    .rates
                    .entry(id.clone())
                    .or_insert_with(crate::ui::SmoothedRate::for_downloads);
                rate.record(*bytes_done);
                Some(format!(
                    "  {id} {} {} {}{}",
                    if total > 0 {
                        format!("{:.0}%", (*bytes_done as f64 / total as f64) * 100.0)
                    } else {
                        format_disk_size(*bytes_done)
                    },
                    rate.speed_label(),
                    rate.eta_label(total),
                    match current_file {
                        Some(name) => format!(" · {name} ({files_done} done)"),
                        None => String::new(),
                    }
                ))
            }
            DownloadEvent::FileDone { id, filename } => {
                Some(format!("  {id} {} {filename}", "✓".green()))
            }
            DownloadEvent::JobDone { id, model } => {
                self.rates.remove(id);
                self.totals.remove(id);
                Some(format!("{} {model} ready ({id})", theme::icon_ok()))
            }
            DownloadEvent::JobFailed { id, error } => {
                self.rates.remove(id);
                Some(format!(
                    "{} {id} failed: {}",
                    theme::icon_fail(),
                    error.red()
                ))
            }
            DownloadEvent::JobCancelled { id } => {
                self.rates.remove(id);
                Some(format!("{} {id} cancelled", theme::icon_warn()))
            }
            DownloadEvent::CatalogReady { id, ok } => Some(format!(
                "{} catalog entry {id} {}",
                if *ok {
                    theme::icon_ok()
                } else {
                    theme::icon_fail()
                },
                if *ok { "installed" } else { "incomplete" }
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn job(id: &str, model: &str, status: JobStatus) -> DownloadJob {
        DownloadJob {
            id: id.into(),
            model: model.into(),
            catalog_id: None,
            status,
            files_done: 1,
            files_total: 4,
            bytes_done: 250,
            bytes_total: 1000,
            current_file: Some("model.safetensors".into()),
            started_at: None,
            completed_at: None,
            error: None,
        }
    }

    /// A catalog id belongs to the other download door, so it is refused
    /// with the command that does route on it.
    #[test]
    fn a_catalog_id_is_refused_with_the_command_that_takes_one() {
        for id in ["cv:827325", "hf:black-forest-labs/FLUX.1-dev"] {
            let message = catalog_id_refusal(id);
            assert!(message.contains("mold pull"), "{message}");
            assert!(message.contains(id), "{message}");
        }
    }

    /// An empty id is refused by name rather than reaching the collection
    /// route and failing with a sentence about the answer.
    #[test]
    fn an_empty_download_id_is_refused_rather_than_reaching_the_queue_route() {
        for blank in ["", "  "] {
            let message = require_download_id(blank).unwrap_err().to_string();
            assert!(message.contains("download id is required"), "{message}");
            assert!(message.contains("downloads list"), "{message}");
        }
        assert_eq!(require_download_id(" dl-1 ").unwrap(), "dl-1");
    }

    #[test]
    fn progress_reads_as_a_percentage_and_degrades_when_the_size_is_unknown() {
        assert_eq!(
            progress_fraction(&job("dl-1", "m", JobStatus::Active)),
            "25%"
        );
        let mut unknown = job("dl-1", "m", JobStatus::Queued);
        unknown.bytes_total = 0;
        assert_eq!(progress_fraction(&unknown), "—");
    }

    /// The snapshot the server opens with is what tells a fresh watcher how
    /// many transfers are already running.
    #[test]
    fn the_opening_snapshot_reports_the_queue_it_found() {
        let mut rates = DownloadRates::default();
        let line = rates
            .describe(
                &DownloadEvent::Snapshot {
                    listing: DownloadsListing {
                        active_jobs: vec![job("dl-1", "flux-dev:q4", JobStatus::Active)],
                        active: None,
                        queued: vec![job("dl-2", "sdxl-turbo:fp16", JobStatus::Queued)],
                        history: Vec::new(),
                    },
                },
                "http://plato:7680",
            )
            .unwrap();
        assert!(line.contains("1 transferring"), "{line}");
        assert!(line.contains("1 waiting"), "{line}");
        // The snapshot also records each job's size, so the first progress
        // frame can already report a percentage.
        assert_eq!(rates.totals.get("dl-1"), Some(&1000));
    }

    /// A settled job stops being tracked, so a long watch does not grow a
    /// rate window per download it has already reported.
    #[test]
    fn a_settled_download_is_forgotten() {
        let mut rates = DownloadRates::default();
        rates.describe(
            &DownloadEvent::Started {
                id: "dl-1".into(),
                files_total: 2,
                bytes_total: 1000,
            },
            "host",
        );
        rates.describe(
            &DownloadEvent::Progress {
                id: "dl-1".into(),
                files_done: 1,
                bytes_done: 500,
                current_file: None,
            },
            "host",
        );
        assert!(rates.rates.contains_key("dl-1"));
        rates.describe(
            &DownloadEvent::JobDone {
                id: "dl-1".into(),
                model: "flux-dev:q4".into(),
            },
            "host",
        );
        assert!(!rates.rates.contains_key("dl-1"));
        assert!(!rates.totals.contains_key("dl-1"));
    }
}
