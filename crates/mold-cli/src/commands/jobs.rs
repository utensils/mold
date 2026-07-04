use anyhow::{bail, Result};
use colored::Colorize;
use mold_core::chain_job::{ChainJobDetail, ChainJobListing, ChainJobSummary, RetakeMode};
use mold_core::{Config, MoldClient};

use crate::{JobsAction, RetakeModeArg};

pub async fn run(action: JobsAction, _config: &Config) -> Result<()> {
    let client = MoldClient::from_env();
    match action {
        JobsAction::List { json } => jobs_list(&client, json).await,
        JobsAction::Show { id, json } => jobs_show(&client, &id, json).await,
        JobsAction::Resume { id } => jobs_resume(&client, &id).await,
        JobsAction::Retake {
            id,
            stage,
            mode,
            seed_offset,
            prompt,
        } => {
            jobs_retake(
                &client,
                &id,
                RetakeArgs {
                    stage,
                    mode,
                    seed_offset,
                    prompt,
                },
            )
            .await
        }
        JobsAction::Cancel { id } => jobs_cancel(&client, &id).await,
        JobsAction::Delete { id, yes } => jobs_delete(&client, &id, yes).await,
        JobsAction::Gc => jobs_gc(&client).await,
    }
}

struct RetakeArgs {
    stage: u32,
    mode: RetakeModeArg,
    seed_offset: Option<u64>,
    prompt: Option<String>,
}

async fn jobs_list(client: &MoldClient, json: bool) -> Result<()> {
    let listing = client.list_chain_jobs().await?;
    if json {
        println!("{}", serde_json::to_string_pretty(&listing)?);
        return Ok(());
    }
    print_listing(&listing);
    Ok(())
}

async fn jobs_show(client: &MoldClient, id: &str, json: bool) -> Result<()> {
    let detail = client.get_chain_job(id).await?;
    if json {
        println!("{}", serde_json::to_string_pretty(&detail)?);
        return Ok(());
    }
    print_detail(&detail);
    Ok(())
}

async fn jobs_resume(client: &MoldClient, id: &str) -> Result<()> {
    let summary = client.resume_chain_job(id).await?;
    print_summary_action("resumed", &summary);
    Ok(())
}

async fn jobs_retake(client: &MoldClient, id: &str, args: RetakeArgs) -> Result<()> {
    let req = mold_core::chain_job::RetakeRequest {
        stage_idx: args.stage,
        mode: match args.mode {
            RetakeModeArg::Cascade => RetakeMode::Cascade,
            RetakeModeArg::Splice => RetakeMode::Splice,
        },
        seed_offset: args.seed_offset,
        prompt: args.prompt,
    };
    let summary = client.retake_chain_job(id, &req).await?;
    print_summary_action("retake queued", &summary);
    Ok(())
}

async fn jobs_cancel(client: &MoldClient, id: &str) -> Result<()> {
    let summary = client.cancel_chain_job(id).await?;
    print_summary_action("cancel requested", &summary);
    Ok(())
}

async fn jobs_delete(client: &MoldClient, id: &str, yes: bool) -> Result<()> {
    if !yes {
        eprint!("Delete chain job {id}? Type the job id to confirm: ");
        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        if input.trim() != id {
            bail!("delete aborted");
        }
    }
    client.delete_chain_job(id).await?;
    println!("{} {}", "deleted".green(), id);
    Ok(())
}

async fn jobs_gc(client: &MoldClient) -> Result<()> {
    let outcome = client.gc_chain_jobs().await?;
    println!(
        "{} swept_ephemeral_jobs={} pruned_artifact_dirs={}",
        "gc complete".green(),
        outcome.swept_ephemeral_jobs,
        outcome.pruned_artifact_dirs
    );
    Ok(())
}

fn print_listing(listing: &ChainJobListing) {
    if listing.jobs.is_empty() {
        println!("No chain jobs.");
        return;
    }
    println!(
        "{:<36} {:<12} {:<28} {:>7} {:>7} {}",
        "ID".bold(),
        "STATE".bold(),
        "MODEL".bold(),
        "STAGE".bold(),
        "UPDATED".bold(),
        "ERROR".bold()
    );
    println!("{}", "─".repeat(104).dimmed());
    for job in &listing.jobs {
        println!(
            "{:<36} {:<12} {:<28} {:>3}/{:<3} {:>7} {}",
            job.id,
            job.state.as_str(),
            truncate(&job.model, 28),
            job.current_stage,
            job.stage_count,
            job.updated_at_unix_ms,
            job.error.as_deref().unwrap_or("")
        );
    }
}

fn print_detail(detail: &ChainJobDetail) {
    let summary = &detail.summary;
    print_summary_action("job", summary);
    println!();
    println!(
        "{:<7} {:<12} {:>8} {:>8} {}",
        "STAGE".bold(),
        "STATE".bold(),
        "FRAMES".bold(),
        "MS".bold(),
        "ERROR".bold()
    );
    println!("{}", "─".repeat(52).dimmed());
    for stage in &detail.stages {
        println!(
            "{:<7} {:<12} {:>8} {:>8} {}",
            stage.idx,
            stage.state.as_str(),
            stage
                .frames_emitted
                .map(|v| v.to_string())
                .unwrap_or_else(|| "—".into()),
            stage
                .generation_time_ms
                .map(|v| v.to_string())
                .unwrap_or_else(|| "—".into()),
            stage.error.as_deref().unwrap_or("")
        );
    }
    if !detail.finalizes.is_empty() {
        println!();
        println!("{}", "Finalizes".bold());
        for finalize in &detail.finalizes {
            println!("  take {} at {}", finalize.output, finalize.at_unix_ms);
        }
    }
}

fn print_summary_action(label: &str, summary: &ChainJobSummary) {
    println!(
        "{} {} state={} stage={}/{} model={}",
        label.green(),
        summary.id,
        summary.state.as_str(),
        summary.current_stage,
        summary.stage_count,
        summary.model
    );
    if let Some(error) = &summary.error {
        println!("{} {}", "error:".red(), error);
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
