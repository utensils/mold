//! `mold queue` — inspect and act on the generation queue of a running server.
//!
//! Every subcommand talks HTTP to `$MOLD_HOST` (with `MOLD_API_KEY` when
//! configured). There is deliberately no local fallback: a queue is a property
//! of one serving host — its durable rows live in that host's `mold.db` and
//! its running work is on that host's GPUs — so an unreachable server is
//! reported as such rather than silently acting on a different queue.
//!
//! The waiting vocabulary is NOT decided here. `mold_core::queue_wait` is the
//! one policy web, desktop, iPhone, and this CLI all resolve through, so the
//! same host cannot describe four identical queued jobs two different ways.

use anyhow::{bail, Context, Result};
use colored::Colorize;
use mold_core::queue_progress::QueueJobProgress;
use mold_core::queue_wait::{queue_wait_label, resolve_listed_wait, QueueWaitStatus};
use mold_core::{MoldClient, QueueJobEntryWire, QueuePlan};

use crate::QueueAction;

/// Wire value of the additive `held` lifecycle.
const STATE_HELD: &str = "held";
const STATE_PAUSED: &str = "paused";
/// Wire value of the running lifecycle.
const STATE_RUNNING: &str = "running";
/// Wire value of the plain queued lifecycle.
const STATE_QUEUED: &str = "queued";

pub async fn run(action: QueueAction) -> Result<()> {
    let client = MoldClient::from_env();
    match action {
        QueueAction::List { held, json } => queue_list(&client, held, json).await,
        QueueAction::Show { job_id, json } => queue_show(&client, &job_id, json).await,
        QueueAction::Cancel {
            job_ids,
            all,
            batch,
            yes,
        } => queue_cancel(&client, &job_ids, all, batch.as_deref(), yes).await,
        QueueAction::Retry { job_ids, held } => queue_retry(&client, &job_ids, held).await,
        QueueAction::Move { job_id, to } => queue_move(&client, &job_id, to).await,
        QueueAction::Pause => queue_pause(&client, true).await,
        QueueAction::Resume => queue_pause(&client, false).await,
        QueueAction::Sweep => queue_sweep(&client).await,
    }
}

/// One listed row with everything the table needs, resolved once.
#[derive(Debug, Clone)]
pub(crate) struct QueueRow {
    pub(crate) entry: QueueJobEntryWire,
    pub(crate) wait: QueueWaitStatus,
    /// Live step counter, read per running row from
    /// `GET /api/queue/{id}/preview`. Absent for queued and held rows, and on
    /// a host running with `MOLD_STEP_PREVIEW=0` that reports no steps.
    pub(crate) progress: Option<QueueJobProgress>,
}

/// Read the WHOLE queue, not one page.
///
/// `GET /api/queue` is bounded by the host's `queue_capacity`, so a backlog
/// longer than that has a tail no single page can see. Every operator action
/// here is exhaustive — "nothing is held", "that job is not held", "N waiting
/// jobs" — and each of those would otherwise be an answer about the first
/// page rather than about the queue.
async fn fetch_rows(client: &MoldClient, held_only: bool) -> Result<Vec<QueueRow>> {
    let listing = fetch_listing(client, held_only).await?;
    let plan = listing.plan.clone();
    let mut rows = Vec::new();
    for entry in listing.entries {
        rows.push(build_row(client, entry, plan.as_ref()).await);
    }
    Ok(rows)
}

/// The whole queue as the host described it, narrowed by `--held`.
///
/// The filter is applied here, once, so the table and `--json` can never
/// disagree about what `--held` selected.
async fn fetch_listing(
    client: &MoldClient,
    held_only: bool,
) -> Result<mold_core::QueueListingWire> {
    let mut listing = client
        .list_queue_all()
        .await
        .with_context(|| format!("could not read the queue on {}", client.host()))?;
    narrow_to_held(&mut listing, held_only);
    Ok(listing)
}

/// Apply `--held`. Pure, and the single application point, so the table and
/// `--json` cannot disagree about what the flag selected.
pub(crate) fn narrow_to_held(listing: &mut mold_core::QueueListingWire, held_only: bool) {
    if held_only {
        listing.entries.retain(|entry| entry.state == STATE_HELD);
    }
}

async fn build_row(
    client: &MoldClient,
    entry: QueueJobEntryWire,
    plan: Option<&QueuePlan>,
) -> QueueRow {
    // A running row's step counter is the one fact the listing cannot carry:
    // `/api/queue` is deliberately payload-free, so the count comes from the
    // per-job snapshot every other surface polls. A failure here is not a
    // listing failure — the row still renders, just without its steps.
    let progress = if entry.state == STATE_RUNNING {
        client.queue_job_progress(&entry.id).await.ok().flatten()
    } else {
        None
    };
    let wait = resolve_listed_wait(
        plan,
        &entry.id,
        Some(entry.position),
        entry.state == STATE_PAUSED,
        entry.state == STATE_HELD,
    );
    QueueRow {
        entry,
        wait,
        progress,
    }
}

async fn queue_list(client: &MoldClient, held: bool, json: bool) -> Result<()> {
    if json {
        // The server's own rows, not a re-serialization of the table's view:
        // `--json` exists so a script reads what the host actually said. Only
        // `--held` narrows it, and it narrows both forms identically.
        let listing = fetch_listing(client, held).await?;
        println!("{}", serde_json::to_string_pretty(&listing)?);
        return Ok(());
    }
    let rows = fetch_rows(client, held).await?;
    print!("{}", render_listing(&rows, held, now_secs()));
    Ok(())
}

async fn queue_show(client: &MoldClient, job_id: &str, json: bool) -> Result<()> {
    let detail = client
        .queue_job(job_id)
        .await
        .with_context(|| format!("could not read queue job {job_id} on {}", client.host()))?;
    let Some(detail) = detail else {
        bail!("queue job {job_id} is not queued on {}", client.host());
    };
    let progress = if detail.job.state == STATE_RUNNING {
        client.queue_job_progress(job_id).await.ok().flatten()
    } else {
        None
    };
    let batch = match detail.job.batch_id.as_deref() {
        Some(id) => client.generation_batch(id).await.unwrap_or(None),
        None => None,
    };
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "detail": detail,
                "progress": progress,
                "batch": batch,
            }))?
        );
        return Ok(());
    }
    // The single-job route answers with the job's own work item rather than a
    // plan, so it is wrapped in one to reach the same resolver every listed
    // row uses — the detail view must not become a second vocabulary.
    let plan = detail.work_item.clone().map(|item| QueuePlan {
        work_items: vec![item],
        ..Default::default()
    });
    let wait = resolve_listed_wait(
        plan.as_ref(),
        &detail.job.id,
        Some(detail.job.position),
        detail.job.state == STATE_PAUSED,
        detail.job.state == STATE_HELD,
    );
    let row = QueueRow {
        entry: detail.job.clone(),
        wait,
        progress,
    };
    print!(
        "{}",
        render_detail(&row, detail.work_item.as_ref(), batch.as_ref(), now_secs())
    );
    Ok(())
}

async fn queue_cancel(
    client: &MoldClient,
    job_ids: &[String],
    all: bool,
    batch: Option<&str>,
    yes: bool,
) -> Result<()> {
    if let Some(batch_id) = batch {
        let status = client
            .cancel_generation_batch(batch_id)
            .await
            .with_context(|| format!("could not cancel batch {batch_id} on {}", client.host()))?;
        println!(
            "{} {} ({} children)",
            "batch cancelled".green(),
            batch_id,
            status.children.len()
        );
        return Ok(());
    }
    if all {
        if !yes {
            let rows = fetch_rows(client, false).await?;
            let waiting = cancellable_by_cancel_all(&rows);
            if waiting == 0 {
                println!("Nothing is queued on {}.", client.host());
                return Ok(());
            }
            if !confirm_cancel_all(waiting, client.host())? {
                bail!("cancel aborted");
            }
        }
        let cancelled = client
            .cancel_all_queue_jobs()
            .await
            .with_context(|| format!("cancel-all failed on {}", client.host()))?;
        println!("{} cancelled={cancelled}", "queue cleared".green());
        return Ok(());
    }
    if job_ids.is_empty() {
        bail!("name at least one job id, or pass --all or --batch <BATCH-ID>");
    }
    for id in job_ids {
        client
            .cancel_queue_job(id)
            .await
            .with_context(|| format!("cancel failed for {id} on {}", client.host()))?;
        println!("{} {id}", "cancelled".green());
    }
    Ok(())
}

async fn queue_retry(client: &MoldClient, job_ids: &[String], held: bool) -> Result<()> {
    // The retry authority is instance + batch + client batch + job. Only the
    // instance belongs to the server, so it is read once and the rest comes
    // off each row.
    let instance_id = client
        .server_status()
        .await
        .with_context(|| format!("could not read the server identity on {}", client.host()))?
        .instance_id
        .unwrap_or_default();
    if instance_id.is_empty() {
        bail!(
            "{} did not report an instance id, so a retry cannot be authorized",
            client.host()
        );
    }
    let rows = fetch_rows(client, true).await?;
    let targets = select_retry_targets(&rows, job_ids, held)?;
    if targets.is_empty() {
        println!("Nothing retryable is held on {}.", client.host());
        return Ok(());
    }
    for entry in targets {
        let request = entry.retry_request(&instance_id).with_context(|| {
            format!(
                "queue job {} is not a durable batch child, so it has no retry authority",
                entry.id
            )
        })?;
        client
            .retry_queue_job(&request)
            .await
            .with_context(|| format!("retry failed for {} on {}", entry.id, client.host()))?;
        println!("{} {}", "retried".green(), entry.id);
    }
    Ok(())
}

/// Which held rows a retry acts on.
///
/// `--held` is every retryable hold; explicit ids are exactly those, and an id
/// that is not a retryable hold is an error rather than a silent skip — a
/// caller who named it is owed the reason.
pub(crate) fn select_retry_targets<'a>(
    rows: &'a [QueueRow],
    job_ids: &[String],
    held: bool,
) -> Result<Vec<&'a QueueJobEntryWire>> {
    if held {
        if !job_ids.is_empty() {
            bail!("--held retries every retryable hold; do not also name job ids");
        }
        return Ok(rows
            .iter()
            .filter(|row| row.entry.retryable == Some(true))
            .map(|row| &row.entry)
            .collect());
    }
    if job_ids.is_empty() {
        bail!("name at least one job id, or pass --held");
    }
    let mut targets = Vec::new();
    for id in job_ids {
        let Some(row) = rows.iter().find(|row| &row.entry.id == id) else {
            bail!("queue job {id} is not held; only a held job can be retried");
        };
        if row.entry.retryable != Some(true) {
            bail!("queue job {id} requires operator repair and cannot be retried");
        }
        targets.push(&row.entry);
    }
    Ok(targets)
}

async fn queue_move(client: &MoldClient, job_id: &str, to: usize) -> Result<()> {
    let entry = client
        .move_queue_job(job_id, to)
        .await
        .with_context(|| format!("reorder failed for {job_id} on {}", client.host()))?;
    // The server clamps a position past the tail, so what it returns is the
    // authority — reporting the requested number would be a lie at the tail.
    println!(
        "{} {job_id} position={}",
        "moved".green(),
        entry.position.to_string().bold()
    );
    Ok(())
}

async fn queue_pause(client: &MoldClient, pause: bool) -> Result<()> {
    let paused = if pause {
        client.pause_queue().await
    } else {
        client.resume_queue().await
    }
    .with_context(|| {
        format!(
            "could not {} the queue on {}",
            if pause { "pause" } else { "resume" },
            client.host()
        )
    })?;
    println!(
        "{} on {}",
        if paused {
            "dispatch paused".yellow()
        } else {
            "dispatch resumed".green()
        },
        client.host()
    );
    Ok(())
}

async fn queue_sweep(client: &MoldClient) -> Result<()> {
    let held = client
        .sweep_held_queue()
        .await
        .with_context(|| format!("held sweep failed on {}", client.host()))?;
    let batches = client
        .sweep_settled_batches()
        .await
        .with_context(|| format!("settled-batch sweep failed on {}", client.host()))?;
    println!(
        "{} held purged={} remaining={}{}",
        "sweep complete".green(),
        held.purged,
        held.remaining,
        if held.media_deferred > 0 {
            format!(" media_deferred={}", held.media_deferred)
        } else {
            String::new()
        }
    );
    println!(
        "               batches purged={} remaining={}",
        batches.purged, batches.remaining
    );
    Ok(())
}

/// How many rows `DELETE /api/queue` would actually remove.
///
/// The endpoint cancels queued and restart-paused rows. Held and running work
/// remain outside bulk cancellation.
pub(crate) fn cancellable_by_cancel_all(rows: &[QueueRow]) -> usize {
    rows.iter()
        .filter(|row| matches!(row.entry.state.as_str(), STATE_QUEUED | STATE_PAUSED))
        .count()
}

fn confirm_cancel_all(count: usize, host: &str) -> Result<bool> {
    use std::io::{self, Write};
    eprint!(
        "Cancel {count} waiting job{} on {host}? Running work is left alone. [y/N] ",
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

/// What one row is doing, in the shared vocabulary.
///
/// A running row is the one case the wait vocabulary does not cover: it is no
/// longer waiting, so it reports its own step counter (or the stage line when
/// the host reports no steps).
pub(crate) fn state_label(row: &QueueRow) -> String {
    if row.entry.state == STATE_RUNNING {
        return match row.progress.as_ref() {
            Some(progress) => match (progress.step, progress.total) {
                (Some(step), Some(total)) => format!("Running {step}/{total}"),
                _ => match progress.stage.as_deref() {
                    Some(stage) if !stage.is_empty() => format!("Running · {stage}"),
                    _ => "Running".to_string(),
                },
            },
            None => "Running".to_string(),
        };
    }
    if row.entry.state == STATE_HELD {
        return "Held".to_string();
    }
    if row.entry.state == STATE_PAUSED {
        return "Paused after restart".to_string();
    }
    queue_wait_label(&row.wait)
}

/// `batch-1 #2`, or an em dash for a row that belongs to no batch.
pub(crate) fn batch_label(entry: &QueueJobEntryWire) -> String {
    match (entry.batch_id.as_deref(), entry.batch_index) {
        (Some(batch), Some(index)) => format!("{batch} #{index}"),
        (Some(batch), None) => batch.to_string(),
        _ => "—".to_string(),
    }
}

/// The prompt a row was submitted with, or an em dash when the payload-free
/// projection carried no settings.
pub(crate) fn prompt_of(entry: &QueueJobEntryWire) -> &str {
    entry
        .metadata
        .as_deref()
        .map(|metadata| metadata.prompt.as_str())
        .map(str::trim)
        .filter(|prompt| !prompt.is_empty())
        .unwrap_or("—")
}

/// Render the human-readable queue table. Pure so the layout is testable.
///
/// Columns: job id, state, model, batch, prompt, and when the row was
/// admitted. Held rows are then listed again beneath with the server's own
/// error sentence and whether a retry is allowed — the two facts a hold
/// exists to communicate.
pub(crate) fn render_listing(rows: &[QueueRow], held_only: bool, now_secs: u64) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    if rows.is_empty() {
        out.push_str(if held_only {
            "Nothing is held.\n"
        } else {
            "Queue is empty.\n"
        });
        return out;
    }
    let id_width = rows
        .iter()
        .map(|row| row.entry.id.chars().count())
        .max()
        .unwrap_or(3)
        .clamp(3, 36);
    let _ = writeln!(
        out,
        "{:<id_width$} {:<16} {:<20} {:<16} {:<32} {:<10}",
        "JOB".bold(),
        "STATE".bold(),
        "MODEL".bold(),
        "BATCH".bold(),
        "PROMPT".bold(),
        "ADMITTED".bold(),
    );
    let _ = writeln!(out, "{}", "─".repeat(id_width + 99).dimmed());
    for row in rows {
        let _ = writeln!(
            out,
            "{:<id_width$} {:<16} {:<20} {:<16} {:<32} {:<10}",
            truncate(&row.entry.id, id_width),
            truncate(&state_label(row), 16),
            truncate(&row.entry.model, 20),
            truncate(&batch_label(&row.entry), 16),
            truncate(prompt_of(&row.entry), 32),
            crate::commands::trash::relative_past(now_secs, row.entry.started_at_unix_ms / 1000),
        );
    }
    let held: Vec<&QueueRow> = rows
        .iter()
        .filter(|row| row.entry.state == STATE_HELD)
        .collect();
    if held.is_empty() {
        return out;
    }
    let _ = writeln!(out);
    let _ = writeln!(out, "{}", "Held".bold());
    for row in held {
        let _ = writeln!(
            out,
            "  {} {} {}",
            row.entry.id,
            if row.entry.retryable == Some(true) {
                "retryable".green()
            } else {
                "needs repair".yellow()
            },
            held_error(&row.entry).dimmed(),
        );
    }
    out
}

/// The server's own sentence for a hold. `error` and `held_reason` are the
/// same fact under two field names; neither is invented here.
fn held_error(entry: &QueueJobEntryWire) -> &str {
    entry
        .error
        .as_deref()
        .or(entry.held_reason.as_deref())
        .unwrap_or("no reason reported")
}

/// Render one job in full. Pure so the layout is testable.
pub(crate) fn render_detail(
    row: &QueueRow,
    work_item: Option<&mold_core::QueueWorkItem>,
    batch: Option<&mold_core::GenerationBatchStatus>,
    now_secs: u64,
) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    let mut line = |label: &str, value: String| {
        let _ = writeln!(out, "{:<18} {value}", format!("{label}:").dimmed());
    };
    line("Job", row.entry.id.clone());
    line("State", state_label(row));
    line("Model", row.entry.model.clone());
    line("Position", row.entry.position.to_string());
    line(
        "Admitted",
        crate::commands::trash::relative_past(now_secs, row.entry.started_at_unix_ms / 1000),
    );
    line("Prompt", prompt_of(&row.entry).to_string());
    if let Some(gpu) = row.entry.gpu {
        line("GPU", gpu.to_string());
    }
    if let Some(target) = row.entry.target_gpu {
        line("Pinned GPU", target.to_string());
    }
    if let Some(durable) = row.entry.durable {
        line("Durable", durable.to_string());
    }
    if row.entry.replayed == Some(true) {
        line("Replayed", "yes".to_string());
    }
    if let Some(attempts) = row.entry.dispatch_attempts {
        line("Dispatch attempts", attempts.to_string());
    }
    if row.entry.batch_id.is_some() {
        line("Batch", batch_label(&row.entry));
    }
    if let Some(client_batch_id) = row.entry.client_batch_id.as_deref() {
        line("Client batch", client_batch_id.to_string());
    }
    if row.entry.state == "held" {
        line("Error", held_error(&row.entry).to_string());
        line(
            "Retryable",
            match row.entry.retryable {
                Some(true) => "yes".to_string(),
                Some(false) => "no — needs operator repair".to_string(),
                // The host reports this bit for every hold. Absence is an
                // unexpected shape, not a refusal, and saying so beats
                // inventing either answer.
                None => "not reported".to_string(),
            },
        );
    }
    if let Some(item) = work_item {
        line("Planned phase", item.presentation_phase().to_string());
        if let Some(label) = item.preparation_label() {
            line("Preparing", label);
        }
        if let Some(label) = item.runtime_label() {
            line("Current stage", label.to_string());
        }
        if let Some(device) = item.planned_device_id.as_deref() {
            line("Planned device", device.to_string());
        }
        if let Some(reason) = item.blocked_reason.as_ref() {
            line("Blocked reason", reason.as_str().to_string());
        }
    }
    if let Some(batch) = batch {
        let settled = batch
            .children
            .iter()
            .filter(|child| {
                matches!(
                    child.state,
                    mold_core::GenerationBatchChildState::Complete
                        | mold_core::GenerationBatchChildState::Failed
                        | mold_core::GenerationBatchChildState::Cancelled
                )
            })
            .count();
        line(
            "Batch progress",
            format!("{settled}/{} settled", batch.children.len()),
        );
    }
    out
}

fn truncate(value: &str, max: usize) -> String {
    if value.chars().count() <= max {
        return value.to_string();
    }
    let suffix = "...";
    let keep = max.saturating_sub(suffix.len());
    let mut out = value.chars().take(keep).collect::<String>();
    out.push_str(suffix);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(id: &str, state: &str, position: usize) -> QueueJobEntryWire {
        QueueJobEntryWire {
            id: id.to_string(),
            model: "flux-dev:q8".to_string(),
            state: state.to_string(),
            started_at_unix_ms: 1_000_000,
            position,
            ..Default::default()
        }
    }

    fn row(entry: QueueJobEntryWire, wait: QueueWaitStatus) -> QueueRow {
        QueueRow {
            entry,
            wait,
            progress: None,
        }
    }

    #[test]
    fn the_state_column_is_the_shared_wait_vocabulary() {
        assert_eq!(
            state_label(&row(entry("a", "queued", 0), QueueWaitStatus::Next)),
            "Next up"
        );
        assert_eq!(
            state_label(&row(entry("b", "queued", 3), QueueWaitStatus::Position(3))),
            "#3 in line"
        );
        assert_eq!(
            state_label(&row(entry("c", "queued", 0), QueueWaitStatus::Queued)),
            "Queued"
        );
        assert_eq!(
            state_label(&row(
                entry("d", "queued", 1),
                QueueWaitStatus::Blocked("Model not installed".into())
            )),
            "Model not installed"
        );
        assert_eq!(
            state_label(&row(entry("e", "held", 4), QueueWaitStatus::Position(4))),
            "Held"
        );
    }

    #[test]
    fn a_running_row_reports_its_steps_and_degrades_without_them() {
        let mut running = row(entry("a", "running", 0), QueueWaitStatus::Next);
        running.progress = Some(QueueJobProgress {
            step: Some(12),
            total: Some(20),
            ..Default::default()
        });
        assert_eq!(state_label(&running), "Running 12/20");

        // A host with MOLD_STEP_PREVIEW=0 reports no steps; the stage line is
        // what it does report, and a bare "Running" is the honest floor.
        running.progress = Some(QueueJobProgress {
            stage: Some("Loading weights".into()),
            ..Default::default()
        });
        assert_eq!(state_label(&running), "Running · Loading weights");
        running.progress = None;
        assert_eq!(state_label(&running), "Running");
    }

    #[test]
    fn batch_children_name_their_batch_and_one_based_index() {
        let mut child = entry("a", "queued", 0);
        child.batch_id = Some("batch-1".into());
        child.batch_index = Some(2);
        assert_eq!(batch_label(&child), "batch-1 #2");
        assert_eq!(batch_label(&entry("b", "queued", 1)), "—");
    }

    #[test]
    fn the_listing_has_one_line_per_job_and_a_held_block() {
        colored::control::set_override(false);
        let mut held = entry("job-held", "held", 1);
        held.error = Some("dependency download failed".into());
        held.retryable = Some(true);
        held.batch_id = Some("batch-1".into());
        held.batch_index = Some(2);
        let mut repair = entry("job-repair", "held", 2);
        repair.held_reason = Some("publication authority is invalid".into());
        repair.retryable = Some(false);
        let rows = vec![
            row(entry("job-run", "running", 0), QueueWaitStatus::Next),
            row(held, QueueWaitStatus::Position(1)),
            row(repair, QueueWaitStatus::Position(2)),
        ];
        let text = render_listing(&rows, false, 1_000 + 3 * 3_600);
        let lines: Vec<&str> = text.lines().collect();
        for column in ["JOB", "STATE", "MODEL", "BATCH", "PROMPT", "ADMITTED"] {
            assert!(lines[0].contains(column), "missing {column}: {}", lines[0]);
        }
        assert!(lines[2].contains("job-run"), "{text}");
        assert!(lines[3].contains("batch-1 #2"), "{text}");
        assert!(lines[3].contains("3h ago"), "{text}");
        assert!(text.contains("dependency download failed"), "{text}");
        assert!(text.contains("retryable"), "{text}");
        assert!(
            text.contains("needs repair") && text.contains("publication authority is invalid"),
            "a non-retryable hold must say so: {text}"
        );
        assert_eq!(render_listing(&[], false, 0), "Queue is empty.\n");
        assert_eq!(render_listing(&[], true, 0), "Nothing is held.\n");
    }

    #[test]
    fn held_rows_without_a_reported_reason_say_so_rather_than_nothing() {
        colored::control::set_override(false);
        let rows = vec![row(entry("job", "held", 0), QueueWaitStatus::Position(0))];
        assert!(
            render_listing(&rows, true, 1_000).contains("no reason reported"),
            "an unexplained hold must still be legible"
        );
    }

    #[test]
    fn cancel_all_counts_only_what_that_call_removes() {
        let rows = vec![
            row(entry("running", "running", 0), QueueWaitStatus::Next),
            row(entry("queued", "queued", 1), QueueWaitStatus::Position(1)),
            row(entry("paused", "paused", 1), QueueWaitStatus::Paused),
            row(entry("held", "held", 2), QueueWaitStatus::Position(2)),
        ];
        assert_eq!(cancellable_by_cancel_all(&rows), 2);
        // A queue of nothing but holds must not offer to cancel them: the
        // call would report `cancelled=0` and leave every row in place.
        let held_only = vec![row(entry("held", "held", 0), QueueWaitStatus::Position(0))];
        assert_eq!(cancellable_by_cancel_all(&held_only), 0);
    }

    #[test]
    fn the_held_filter_narrows_the_json_exactly_as_it_narrows_the_table() {
        let mut listing = mold_core::QueueListingWire {
            entries: vec![
                entry("running", "running", 0),
                entry("held", "held", 1),
                entry("queued", "queued", 2),
            ],
            ..Default::default()
        };
        let mut untouched = listing.clone();
        narrow_to_held(&mut untouched, false);
        assert_eq!(untouched.entries.len(), 3);
        narrow_to_held(&mut listing, true);
        assert_eq!(
            listing
                .entries
                .iter()
                .map(|entry| entry.id.as_str())
                .collect::<Vec<_>>(),
            vec!["held"]
        );
    }

    #[test]
    fn a_hold_whose_retryable_bit_is_missing_says_so_rather_than_guessing() {
        colored::control::set_override(false);
        let text = render_detail(
            &row(entry("job", "held", 0), QueueWaitStatus::Position(0)),
            None,
            None,
            1_000,
        );
        assert!(text.contains("not reported"), "{text}");
    }

    #[test]
    fn retry_targets_are_every_retryable_hold_or_exactly_the_named_ones() {
        let mut ok = entry("ok", "held", 0);
        ok.retryable = Some(true);
        let mut broken = entry("broken", "held", 1);
        broken.retryable = Some(false);
        let rows = vec![
            row(ok, QueueWaitStatus::Position(0)),
            row(broken, QueueWaitStatus::Position(1)),
        ];

        let all = select_retry_targets(&rows, &[], true).unwrap();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].id, "ok");

        let named = select_retry_targets(&rows, &["ok".into()], false).unwrap();
        assert_eq!(named.len(), 1);

        // A named job that cannot be retried is an error, never a silent skip.
        let refused = select_retry_targets(&rows, &["broken".into()], false).unwrap_err();
        assert!(
            format!("{refused}").contains("operator repair"),
            "{refused}"
        );
        let missing = select_retry_targets(&rows, &["nope".into()], false).unwrap_err();
        assert!(format!("{missing}").contains("is not held"), "{missing}");
        let both = select_retry_targets(&rows, &["ok".into()], true).unwrap_err();
        assert!(format!("{both}").contains("--held"), "{both}");
        let neither = select_retry_targets(&rows, &[], false).unwrap_err();
        assert!(format!("{neither}").contains("--held"), "{neither}");
    }

    #[test]
    fn the_detail_view_names_the_hold_and_its_retry_authority() {
        colored::control::set_override(false);
        let mut held = entry("job-held", "held", 4);
        held.error = Some("dependency download failed".into());
        held.retryable = Some(false);
        held.batch_id = Some("batch-1".into());
        held.client_batch_id = Some("client-1".into());
        held.batch_index = Some(2);
        held.dispatch_attempts = Some(2);
        held.replayed = Some(true);
        let text = render_detail(&row(held, QueueWaitStatus::Position(4)), None, None, 1_000);
        for expected in [
            "job-held",
            "Held",
            "batch-1 #2",
            "client-1",
            "dependency download failed",
            "needs operator repair",
            "Dispatch attempts",
            "Replayed",
        ] {
            assert!(text.contains(expected), "missing {expected}: {text}");
        }
    }

    #[test]
    fn a_row_with_no_settings_shows_a_dash_rather_than_an_empty_prompt() {
        // `GET /api/queue`'s durable projection is payload-free by design, so
        // a pre-dispatch durable row carries no metadata at all.
        assert_eq!(prompt_of(&entry("a", "queued", 0)), "—");
    }
}
