use anyhow::{bail, Context, Result};
use colored::Colorize;
use mold_core::chain_job::{ChainJobDetail, ChainJobListing, ChainJobSummary, RetakeMode};
use mold_core::{Config, MoldClient};

use crate::{JobsAction, RetakeModeArg};

pub async fn run(action: JobsAction, _config: &Config) -> Result<()> {
    let client = MoldClient::from_env();
    match action {
        JobsAction::List { json } => jobs_list(&client, json).await,
        JobsAction::Show { id, json, script } => jobs_show(&client, &id, json, script).await,
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
        JobsAction::Amend {
            id,
            script,
            fps,
            seed,
            steps,
            guidance,
            strength,
            motion_tail,
            audio,
            no_audio,
            dry_run,
        } => {
            jobs_amend(
                &client,
                &id,
                AmendArgs {
                    script,
                    fps,
                    seed,
                    steps,
                    guidance,
                    strength,
                    motion_tail,
                    enable_audio: if audio {
                        Some(true)
                    } else if no_audio {
                        Some(false)
                    } else {
                        None
                    },
                    dry_run,
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
    // Teach the shell the ids this listing just showed: every other `mold
    // jobs` verb takes one, and a completer cannot ask the server (see
    // `crate::completion_cache`).
    crate::completion_cache::record_reached_host(client.host(), |cache| {
        cache.record_job_ids(listing.jobs.iter().map(|job| job.id.clone()));
    });
    if json {
        println!("{}", serde_json::to_string_pretty(&listing)?);
        return Ok(());
    }
    print_listing(&listing);
    Ok(())
}

async fn jobs_show(client: &MoldClient, id: &str, json: bool, script: bool) -> Result<()> {
    let detail = client.get_chain_job(id).await?;
    if script {
        // The document `mold jobs amend --script` reads, so the pair is a
        // round trip: `--json` prints a ChainJobDetail, which amend cannot
        // take and which telling a user to edit would never have worked.
        print!("{}", render_job_script(&detail)?);
        return Ok(());
    }
    if json {
        println!("{}", serde_json::to_string_pretty(&detail)?);
        return Ok(());
    }
    print_detail(&detail);
    Ok(())
}

/// The job's EFFECTIVE script (its original request with every retake and
/// amendment applied) as `mold.chain.v1` TOML.
fn render_job_script(detail: &ChainJobDetail) -> Result<String> {
    mold_core::chain_toml::write_script(&detail.script)
        .map_err(|error| anyhow::anyhow!("could not render the job's script: {error}"))
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

/// Flags of `mold jobs amend`, resolved from clap.
struct AmendArgs {
    script: std::path::PathBuf,
    fps: Option<u32>,
    seed: Option<u64>,
    steps: Option<u32>,
    guidance: Option<f64>,
    strength: Option<f64>,
    motion_tail: Option<u32>,
    enable_audio: Option<bool>,
    dry_run: bool,
}

/// `mold jobs amend <ID> --script edited.toml` — replace a sequence's stage
/// list from an edited script.
///
/// `AmendRequest` carries the FULL stage list in canonical order and has no
/// per-stage index, so the honest CLI shape is a script rather than a pile
/// of `--stage-N-prompt` flags. The script is read through the same
/// `read_script_resolving_paths` that `mold run --script` uses, so a
/// per-stage `source_image_path` resolves relative to the script file here
/// too.
async fn jobs_amend(client: &MoldClient, id: &str, args: AmendArgs) -> Result<()> {
    let text = std::fs::read_to_string(&args.script)
        .with_context(|| format!("could not read {}", args.script.display()))?;
    let script_dir = args
        .script
        .parent()
        .filter(|dir| !dir.as_os_str().is_empty())
        .unwrap_or_else(|| std::path::Path::new("."));
    let edited = mold_core::chain_toml::read_script_resolving_paths(&text, script_dir)
        .map_err(|error| anyhow::anyhow!("{}: {error}", args.script.display()))?;
    let detail = client
        .get_chain_job(id)
        .await
        .with_context(|| format!("could not read sequence {id} on {}", client.host()))?;
    refuse_unamendable_edits(&detail.script, &edited)?;
    let req = amend_request_from_script(&edited, &args);
    if args.dry_run {
        print!("{}", render_amend_plan(id, &req));
        return Ok(());
    }
    let response = client
        .amend_chain_job(id, &req)
        .await
        .with_context(|| format!("amend failed on {}", client.host()))?;
    print_summary_action("amended", &response.summary);
    println!(
        "{} {} of {} stage{} kept their rendered clips; rendering requeues from stage {}",
        "preserved".green(),
        response.preserved_stages,
        req.stages.len(),
        if req.stages.len() == 1 { "" } else { "s" },
        response.preserved_stages
    );
    Ok(())
}

/// Build the amend body: stages from the edited script, chain-level overlays
/// from its `[chain]` block, each overridable by the flag that names it.
///
/// Prompts are newline-normalized here rather than at the call site, so a
/// `--dry-run` shows exactly the text the amend would carry.
fn amend_request_from_script(
    edited: &mold_core::chain::ChainScript,
    args: &AmendArgs,
) -> mold_core::chain_job::AmendRequest {
    let mut req = mold_core::chain_job::AmendRequest {
        stages: edited.stages.clone(),
        motion_tail_frames: args.motion_tail.or(Some(edited.chain.motion_tail_frames)),
        fps: args.fps.or(Some(edited.chain.fps)),
        seed: args.seed.or(edited.chain.seed),
        steps: args.steps.or(Some(edited.chain.steps)),
        guidance: args.guidance.or(Some(edited.chain.guidance)),
        strength: args.strength.or(Some(edited.chain.strength)),
        enable_audio: args.enable_audio.or(edited.chain.enable_audio),
    };
    req.normalize_prompt_newlines();
    req
}

/// Refuse an edit to something amend cannot change.
///
/// The server would reject or silently ignore these; naming them here says
/// which line of the file is the problem, and that the answer is a new
/// sequence rather than a different flag.
fn refuse_unamendable_edits(
    current: &mold_core::chain::ChainScript,
    edited: &mold_core::chain::ChainScript,
) -> Result<()> {
    let mut changed = Vec::new();
    if current.chain.model != edited.chain.model {
        changed.push(format!(
            "model ({} -> {})",
            current.chain.model, edited.chain.model
        ));
    }
    if current.chain.width != edited.chain.width {
        changed.push(format!(
            "width ({} -> {})",
            current.chain.width, edited.chain.width
        ));
    }
    if current.chain.height != edited.chain.height {
        changed.push(format!(
            "height ({} -> {})",
            current.chain.height, edited.chain.height
        ));
    }
    if current.chain.output_format != edited.chain.output_format {
        changed.push(format!(
            "output_format ({} -> {})",
            current.chain.output_format, edited.chain.output_format
        ));
    }
    if changed.is_empty() {
        return Ok(());
    }
    bail!(
        "amend cannot change {}; render a new sequence with `mold run --script` instead",
        changed.join(", ")
    );
}

/// What `--dry-run` prints: the stages as they would be sent, and the
/// chain-level overlays riding with them.
fn render_amend_plan(id: &str, req: &mold_core::chain_job::AmendRequest) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    let _ = writeln!(
        out,
        "{} {id}: {} stage{}",
        "would amend".yellow(),
        req.stages.len(),
        if req.stages.len() == 1 { "" } else { "s" }
    );
    for (idx, stage) in req.stages.iter().enumerate() {
        let _ = writeln!(
            out,
            "  {idx:>2}  {:>5} frames  {:<8} {}",
            stage.frames,
            format!("{:?}", stage.transition).to_lowercase(),
            truncate(&stage.prompt.replace('\n', " "), 60)
        );
    }
    let mut overlays = Vec::new();
    if let Some(value) = req.fps {
        overlays.push(format!("fps={value}"));
    }
    if let Some(value) = req.seed {
        overlays.push(format!("seed={value}"));
    }
    if let Some(value) = req.steps {
        overlays.push(format!("steps={value}"));
    }
    if let Some(value) = req.guidance {
        overlays.push(format!("guidance={value}"));
    }
    if let Some(value) = req.strength {
        overlays.push(format!("strength={value}"));
    }
    if let Some(value) = req.motion_tail_frames {
        overlays.push(format!("motion_tail={value}"));
    }
    if let Some(value) = req.enable_audio {
        overlays.push(format!("audio={value}"));
    }
    let _ = writeln!(out, "  overlays: {}", overlays.join(" "));
    let _ = writeln!(
        out,
        "  the host decides how many leading stages keep their clips."
    );
    out
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

#[cfg(test)]
mod tests {
    use super::*;
    use mold_core::chain::{ChainScript, ChainScriptChain, ChainStage, TransitionMode};

    fn stage(prompt: &str, frames: u32) -> ChainStage {
        ChainStage {
            prompt: prompt.to_string(),
            frames,
            source_image: None,
            negative_prompt: None,
            seed_offset: None,
            transition: TransitionMode::default(),
            fade_frames: None,
            model: None,
            loras: Vec::new(),
            references: Vec::new(),
        }
    }

    fn script(stages: Vec<ChainStage>) -> ChainScript {
        ChainScript {
            schema: "mold.chain.v1".into(),
            chain: ChainScriptChain {
                model: "ltx-2-19b-distilled:fp8".into(),
                width: 1216,
                height: 704,
                fps: 24,
                seed: Some(7),
                steps: 8,
                guidance: 1.0,
                strength: 0.75,
                motion_tail_frames: 17,
                output_format: mold_core::OutputFormat::Mp4,
                enable_audio: Some(true),
            },
            stages,
        }
    }

    fn args() -> AmendArgs {
        AmendArgs {
            script: std::path::PathBuf::from("shot.toml"),
            fps: None,
            seed: None,
            steps: None,
            guidance: None,
            strength: None,
            enable_audio: None,
            motion_tail: None,
            dry_run: false,
        }
    }

    /// Amend replaces the WHOLE stage list, so the script's stages reach the
    /// request in the order they were authored — there is no stage index on
    /// the wire to put them back with.
    #[test]
    fn the_script_s_stages_reach_the_request_in_canonical_order() {
        let edited = script(vec![stage("one", 97), stage("two", 49), stage("three", 25)]);
        let req = amend_request_from_script(&edited, &args());
        assert_eq!(
            req.stages
                .iter()
                .map(|entry| entry.prompt.as_str())
                .collect::<Vec<_>>(),
            vec!["one", "two", "three"]
        );
        assert_eq!(
            req.stages
                .iter()
                .map(|entry| entry.frames)
                .collect::<Vec<_>>(),
            vec![97, 49, 25]
        );
    }

    /// The script's own `[chain]` block supplies the chain-level overlays —
    /// editing `fps` in the file is the point of editing the file — and a
    /// flag overrides exactly the one field it names.
    #[test]
    fn chain_overlays_come_from_the_script_and_flags_override_one_at_a_time() {
        let edited = script(vec![stage("one", 97)]);
        let from_script = amend_request_from_script(&edited, &args());
        assert_eq!(from_script.fps, Some(24));
        assert_eq!(from_script.seed, Some(7));
        assert_eq!(from_script.steps, Some(8));
        assert_eq!(from_script.guidance, Some(1.0));
        assert_eq!(from_script.strength, Some(0.75));
        assert_eq!(from_script.motion_tail_frames, Some(17));
        assert_eq!(from_script.enable_audio, Some(true));

        let overridden = amend_request_from_script(
            &edited,
            &AmendArgs {
                fps: Some(30),
                seed: Some(99),
                steps: Some(12),
                guidance: Some(3.5),
                strength: Some(0.5),
                enable_audio: Some(false),
                motion_tail: Some(0),
                ..args()
            },
        );
        assert_eq!(overridden.fps, Some(30));
        assert_eq!(overridden.seed, Some(99));
        assert_eq!(overridden.steps, Some(12));
        assert_eq!(overridden.guidance, Some(3.5));
        assert_eq!(overridden.strength, Some(0.5));
        assert_eq!(overridden.enable_audio, Some(false));
        assert_eq!(overridden.motion_tail_frames, Some(0));
    }

    /// Literal `\n` in an edited prompt means a newline, exactly as it does
    /// on every other prompt-bearing path.
    #[test]
    fn stage_prompts_are_newline_normalized_before_the_request_leaves() {
        let edited = script(vec![stage(r"a cat\na dog", 97)]);
        let req = amend_request_from_script(&edited, &args());
        assert_eq!(req.stages[0].prompt, "a cat\na dog");
    }

    /// `mold jobs show --script` writes the job's effective script as
    /// `mold.chain.v1` TOML, which is what `mold jobs amend --script` reads.
    ///
    /// `--json` prints `ChainJobDetail`, a different document — telling a
    /// user to edit that and hand it back would never have worked.
    #[tokio::test]
    async fn show_script_round_trips_into_the_amend_input() {
        colored::control::set_override(false);
        let server = amend_host(false).await;
        let client = MoldClient::new(&server.uri());
        let detail = client.get_chain_job("job-7").await.unwrap();
        let rendered = render_job_script(&detail).unwrap();

        let parsed = mold_core::chain_toml::read_script_resolving_paths(
            &rendered,
            std::path::Path::new("."),
        )
        .expect("show --script output must parse as a chain script");
        assert_eq!(
            parsed
                .stages
                .iter()
                .map(|stage| (stage.prompt.as_str(), stage.frames))
                .collect::<Vec<_>>(),
            vec![("a cat walks in", 97), ("the cat sits down", 49)]
        );
        assert_eq!(parsed.chain.model, "ltx-2-19b-distilled:fp8");
        assert_eq!(parsed.chain.fps, 24);

        // And the amend it feeds carries exactly those stages.
        let req = amend_request_from_script(&parsed, &args());
        assert_eq!(req.stages.len(), 2);
        refuse_unamendable_edits(&detail.script, &parsed).unwrap();
    }

    fn script_toml() -> &'static str {
        r#"schema = "mold.chain.v1"

[chain]
model = "ltx-2-19b-distilled:fp8"
width = 1216
height = 704
fps = 24
seed = 7
steps = 8
guidance = 1.0
strength = 0.75
motion_tail_frames = 17
output_format = "mp4"

[[stage]]
prompt = "a cat walks in"
frames = 97

[[stage]]
prompt = "the cat sits down"
frames = 49
"#
    }

    async fn amend_host(post: bool) -> wiremock::MockServer {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};
        let server = MockServer::start().await;
        let detail = serde_json::json!({
            "id": "job-7",
            "state": "paused",
            "model": "ltx-2-19b-distilled:fp8",
            "stage_count": 2,
            "current_stage": 0,
            "created_at_unix_ms": 1_700_000_000_000_u64,
            "updated_at_unix_ms": 1_700_000_000_000_u64,
            "error": null,
            "ephemeral": false,
            "stages": [],
            "finalizes": [],
            "retakes": [],
            "script": serde_json::from_str::<serde_json::Value>(
                &serde_json::to_string(&mold_core::chain_toml::read_script(script_toml()).unwrap())
                    .unwrap(),
            )
            .unwrap()
        });
        Mock::given(method("GET"))
            .and(path("/api/chain-jobs/job-7"))
            .respond_with(ResponseTemplate::new(200).set_body_json(detail))
            .mount(&server)
            .await;
        if post {
            Mock::given(method("POST"))
                .and(path("/api/chain-jobs/job-7/amend"))
                .respond_with(ResponseTemplate::new(202).set_body_json(serde_json::json!({
                    "id": "job-7",
                    "state": "queued",
                    "model": "ltx-2-19b-distilled:fp8",
                    "stage_count": 2,
                    "current_stage": 1,
                    "created_at_unix_ms": 1_700_000_000_000_u64,
                    "updated_at_unix_ms": 1_700_000_000_001_u64,
                    "error": null,
                    "ephemeral": false,
                    "preserved_stages": 1
                })))
                .mount(&server)
                .await;
        }
        server
    }

    fn write_script(dir: &std::path::Path, body: &str) -> std::path::PathBuf {
        let path = dir.join("edited.toml");
        std::fs::write(&path, body).unwrap();
        path
    }

    /// The whole verb end to end: read the edited script, check it against
    /// the job, POST it, and report what the host kept.
    #[tokio::test]
    async fn amend_posts_the_edited_script_and_reports_preserved_stages() {
        colored::control::set_override(false);
        let server = amend_host(true).await;
        let client = MoldClient::new(&server.uri());
        let dir = tempfile::tempdir().unwrap();
        let script = write_script(
            dir.path(),
            &script_toml().replace("the cat sits down", "the cat leaps"),
        );
        jobs_amend(&client, "job-7", AmendArgs { script, ..args() })
            .await
            .unwrap();
    }

    /// `--dry-run` reads the job and the script and prints the plan; it
    /// never posts. The mock has no amend route, so a POST would fail.
    #[tokio::test]
    async fn a_dry_run_prints_the_plan_and_posts_nothing() {
        colored::control::set_override(false);
        let server = amend_host(false).await;
        let client = MoldClient::new(&server.uri());
        let dir = tempfile::tempdir().unwrap();
        let script = write_script(dir.path(), script_toml());
        jobs_amend(
            &client,
            "job-7",
            AmendArgs {
                script,
                dry_run: true,
                ..args()
            },
        )
        .await
        .unwrap();

        let plan = render_amend_plan(
            "job-7",
            &amend_request_from_script(
                &mold_core::chain_toml::read_script(script_toml()).unwrap(),
                &args(),
            ),
        );
        assert!(plan.contains("2 stages"), "{plan}");
        assert!(plan.contains("a cat walks in"), "{plan}");
        assert!(plan.contains("fps=24"), "{plan}");
        assert!(plan.contains("motion_tail=17"), "{plan}");
    }

    /// Model, size and container are not amendable, so a script that changed
    /// one is refused BY NAME rather than posted and half-applied.
    #[test]
    fn a_script_that_changes_what_amend_cannot_is_refused_by_name() {
        let current = script(vec![stage("one", 97)]);
        let mut edited = script(vec![stage("one", 97)]);
        edited.chain.model = "wan22-ti2v-5b".into();
        edited.chain.width = 512;
        let error = refuse_unamendable_edits(&current, &edited)
            .unwrap_err()
            .to_string();
        assert!(error.contains("model"), "{error}");
        assert!(error.contains("width"), "{error}");
        assert!(!error.contains("height"), "unchanged field named: {error}");
        assert!(error.contains("new sequence"), "{error}");

        refuse_unamendable_edits(&current, &script(vec![stage("edited", 49)])).unwrap();
    }
}
