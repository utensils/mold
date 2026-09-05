//! `mold runpod` — native RunPod pod management.
//!
//! Wraps `mold_core::runpod::RunPodClient` so users can manage cloud GPU pods
//! end-to-end from the mold CLI: list/create/stop/delete pods, hand logs off
//! to RunPod's console,
//! check spend, and (via `run`) generate images on a freshly-provisioned pod
//! with one command.

use anyhow::{bail, Context, Result};
use clap_complete::engine::CompletionCandidate;
use colored::Colorize;
use mold_core::config::Config;
use mold_core::error::MoldError;
use mold_core::runpod::{
    canonical_supported_gpu_type_id, legacy_gpu_type_id_from_display_name,
    valid_network_volume_size, CreateNetworkVolumeRequest, CreatePodRequest, GpuType,
    NetworkVolume, Pod, RunPodClient, UpdateNetworkVolumeRequest, API_KEY_ENV, DEFAULT_ENDPOINT,
    NETWORK_VOLUME_MAX_GB, NETWORK_VOLUME_MIN_GB,
};

use crate::commands::durable_generation::{
    canonical_singleton_artifact_observed, CanonicalGenerationArtifact, CanonicalGenerationEvent,
};

/// Render one print on a pod, keeping a spinner alive from the durable
/// batch's own progress and state transitions.
///
/// A pod render takes minutes, and this path deliberately does not attach to
/// `/api/generate/stream` because RunPod's proxy times out at 100 seconds.
/// The durable observer polls each running child's progress snapshot instead,
/// which restores the per-step and model-pull lines on top of one
/// authoritative event per committed child transition.
async fn generate_with_runpod_transport(
    client: &mold_core::MoldClient,
    request: &mold_core::GenerateRequest,
) -> Result<CanonicalGenerationArtifact> {
    let (events, mut rx) = tokio::sync::mpsc::unbounded_channel::<CanonicalGenerationEvent>();
    let spinner = tokio::spawn(async move {
        let mut pb: Option<indicatif::ProgressBar> = None;
        let mut progress = mold_core::queue_progress::ProgressEventStream::new();
        while let Some(event) = rx.recv().await {
            let message = match event {
                // Per-step progress, including the model-pull download bars
                // this path exists for: RunPod's proxy times out at 100
                // seconds, so a silent pull looks like a dead pod.
                CanonicalGenerationEvent::Progress {
                    job_id,
                    progress: snapshot,
                    ..
                } => {
                    let Some(last) = progress.events(&job_id, &snapshot).pop() else {
                        continue;
                    };
                    format_progress_event(&last)
                }
                CanonicalGenerationEvent::Admitted { status, .. }
                | CanonicalGenerationEvent::Snapshot { status, .. } => {
                    let Some(child) = status.children.first() else {
                        continue;
                    };
                    format!("{:?}", child.state).to_lowercase()
                }
                CanonicalGenerationEvent::ReconcileDelayed { .. } => continue,
            };
            let bar = pb.get_or_insert_with(|| {
                let bar = indicatif::ProgressBar::new_spinner();
                bar.set_style(
                    indicatif::ProgressStyle::with_template(&format!(
                        "{{spinner:.{}}} {{msg}}",
                        theme::SPINNER_STYLE
                    ))
                    .unwrap(),
                );
                bar.enable_steady_tick(std::time::Duration::from_millis(80));
                bar
            });
            bar.set_message(message);
        }
        if let Some(bar) = pb {
            bar.finish_and_clear();
        }
    });
    let artifact = canonical_singleton_artifact_observed(client, request, Some(&events)).await;
    drop(events);
    let _ = spinner.await;
    artifact.with_context(|| "durable generation failed")
}
use crate::theme;
use crate::AlreadyReported;

// ── Local state persistence ────────────────────────────────────────────

/// Filename for persisted pod state (last-used pod id + lifetime log).
const STATE_FILE: &str = "runpod-state.json";
const HISTORY_FILE: &str = "runpod-history.jsonl";

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct RunPodState {
    /// Pod id of the last pod created by this CLI (warm pod for `run`).
    pub last_pod_id: Option<String>,
    /// Unix-timestamp seconds when `last_pod_id` was created.
    pub last_pod_created_at: Option<u64>,
    /// Unix-timestamp seconds of the most recent interaction with `last_pod_id`.
    pub last_pod_last_used_at: Option<u64>,
    /// GPU display name of the warm pod (used for display in `run`).
    pub last_pod_gpu: Option<String>,
    /// Hourly cost of the warm pod.
    pub last_pod_cost_per_hr: Option<f64>,
}

/// One spinner line per progress event, matching what the attached
/// `/api/generate/stream` path used to print on this transport.
fn format_progress_event(ev: &mold_core::types::SseProgressEvent) -> String {
    use mold_core::types::SseProgressEvent as E;
    match ev {
        E::DependencyWait { dependency, reason } => {
            format!("waiting for dependency {dependency}: {reason}")
        }
        E::Preview { step, total, .. } => format!("denoise preview {step}/{total}"),
        E::StageStart { name } => format!("stage: {name}"),
        E::StageProgress {
            name,
            current,
            total,
        } => format!("{name} {current}/{total}"),
        E::StageDone { name, elapsed_ms } => format!("done: {name} ({elapsed_ms}ms)"),
        E::Info { message } => message.clone(),
        E::CacheHit { resource } => format!("cached: {resource}"),
        E::DenoiseStep { step, total, .. } => format!("denoise {step}/{total}"),
        E::DownloadProgress {
            filename,
            file_index,
            total_files,
            bytes_downloaded,
            bytes_total,
            ..
        } => format!(
            "pull [{file_index}/{total_files}] {filename} {}/{}",
            human_bytes(*bytes_downloaded),
            human_bytes(*bytes_total)
        ),
        E::DownloadDone {
            filename,
            file_index,
            total_files,
            ..
        } => format!("\u{2713} [{file_index}/{total_files}] {filename}"),
        E::PullComplete { model } => format!("pull complete: {model}"),
        E::Queued { position, .. } => format!("queued (#{position})"),
        E::WeightLoad {
            component,
            bytes_loaded,
            bytes_total,
        } => format!(
            "loading {component} {}/{}",
            human_bytes(*bytes_loaded),
            human_bytes(*bytes_total)
        ),
    }
}

fn human_bytes(b: u64) -> String {
    mold_core::format::human_bytes_compact(b)
}

fn runpod_state_dir() -> Result<std::path::PathBuf> {
    let dir = Config::mold_dir()
        .ok_or_else(|| anyhow::anyhow!("could not resolve MOLD_HOME / mold config dir"))?;
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("failed to create state dir {}", dir.display()))?;
    Ok(dir)
}

fn state_path() -> Result<std::path::PathBuf> {
    Ok(runpod_state_dir()?.join(STATE_FILE))
}

fn history_path() -> Result<std::path::PathBuf> {
    Ok(runpod_state_dir()?.join(HISTORY_FILE))
}

pub fn load_state() -> RunPodState {
    let path = match state_path() {
        Ok(p) => p,
        Err(_) => return RunPodState::default(),
    };
    match std::fs::read_to_string(&path) {
        Ok(s) => serde_json::from_str(&s).unwrap_or_default(),
        Err(_) => RunPodState::default(),
    }
}

pub fn save_state(state: &RunPodState) -> Result<()> {
    let path = state_path()?;
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, serde_json::to_string_pretty(state)?)?;
    std::fs::rename(&tmp, &path)?;
    Ok(())
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HistoryEntry {
    pub pod_id: String,
    pub created_at: u64,
    pub deleted_at: Option<u64>,
    pub cost_per_hr: f64,
    pub gpu: String,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub prompt: Option<String>,
}

pub fn append_history(entry: &HistoryEntry) -> Result<()> {
    let path = history_path()?;
    let mut line = serde_json::to_string(entry)?;
    line.push('\n');
    use std::io::Write as _;
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .with_context(|| format!("open {}", path.display()))?;
    f.write_all(line.as_bytes())?;
    Ok(())
}

pub fn read_history() -> Vec<HistoryEntry> {
    let path = match history_path() {
        Ok(p) => p,
        Err(_) => return Vec::new(),
    };
    let Ok(text) = std::fs::read_to_string(&path) else {
        return Vec::new();
    };
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| serde_json::from_str::<HistoryEntry>(l).ok())
        .collect()
}

fn now_epoch() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn network_volume_selection_mismatch(attached: Option<&str>, requested: Option<&str>) -> bool {
    attached != requested
}

fn warm_datacenter_requirement(
    cli: Option<&str>,
    config: Option<&str>,
    network_volume: Option<&str>,
) -> Option<String> {
    if network_volume.is_some() {
        None
    } else {
        cli.or(config).map(str::to_owned)
    }
}

fn authoritative_datacenter(
    cli: Option<&str>,
    config: Option<&str>,
    network_volume: Option<&str>,
) -> Option<String> {
    network_volume.or(cli).or(config).map(str::to_owned)
}

fn network_volume_placement(
    volume: Option<&NetworkVolume>,
    fallback_datacenter: Option<String>,
    fallback_cloud: &str,
    fallback_workspace_gb: u32,
) -> (Option<String>, String, u32) {
    match volume {
        Some(volume) => (Some(volume.data_center_id.clone()), "SECURE".to_string(), 0),
        None => (
            fallback_datacenter,
            fallback_cloud.to_string(),
            fallback_workspace_gb,
        ),
    }
}

// ── Client construction + error translation ────────────────────────────

fn build_client() -> Result<RunPodClient> {
    let config = Config::load_or_default();
    RunPodClient::from_settings(&config.runpod).map_err(|e| e.into())
}

fn explain_runpod_error(err: &anyhow::Error) {
    if let Some(MoldError::RunPodAuth(msg)) = err.downcast_ref::<MoldError>() {
        eprintln!("{} {msg}", theme::prefix_error());
        eprintln!(
            "       {} get a key at {}",
            theme::prefix_hint(),
            "https://www.runpod.io/console/user/settings".cyan()
        );
        eprintln!(
            "       {} then run {} or {}",
            theme::prefix_hint(),
            "mold config set runpod.api_key <key>".bold(),
            format!("export {API_KEY_ENV}=<key>").bold(),
        );
    } else if let Some(MoldError::RunPodNoStock(msg)) = err.downcast_ref::<MoldError>() {
        eprintln!("{} {msg}", theme::prefix_error());
        eprintln!(
            "       {} no GPUs free in that datacenter — try another or a different GPU",
            theme::prefix_hint(),
        );
    } else if let Some(MoldError::RunPodNotFound(msg)) = err.downcast_ref::<MoldError>() {
        eprintln!("{} {msg}", theme::prefix_error());
    } else {
        eprintln!("{} {err}", theme::prefix_error());
    }
}

/// Run an async block with a spinner. Returns the result, propagating the
/// error after the spinner is finished.
async fn with_spinner<F, T>(msg: &str, fut: F) -> anyhow::Result<T>
where
    F: std::future::Future<Output = anyhow::Result<T>>,
{
    let spinner = indicatif::ProgressBar::new_spinner();
    spinner.set_style(
        indicatif::ProgressStyle::with_template(&format!(
            "{{spinner:.{}}} {{msg}}",
            theme::SPINNER_STYLE
        ))
        .unwrap()
        .tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏ "),
    );
    spinner.set_message(msg.to_string());
    spinner.enable_steady_tick(std::time::Duration::from_millis(80));
    let result = fut.await;
    spinner.finish_and_clear();
    result
}

// ── Read-only subcommands ──────────────────────────────────────────────

/// `mold runpod doctor` — check API key, connectivity, account info.
pub async fn run_doctor() -> Result<()> {
    println!("{} RunPod diagnostics", theme::icon_info());
    let config = Config::load_or_default();

    // 1. API key presence
    let env_key = std::env::var(API_KEY_ENV).ok().filter(|k| !k.is_empty());
    let cfg_key = config.runpod.api_key.clone();
    match (&env_key, &cfg_key) {
        (Some(_), _) => println!(
            "{} api key source: {}",
            theme::icon_ok(),
            API_KEY_ENV.bold()
        ),
        (None, Some(_)) => println!(
            "{} api key source: {}",
            theme::icon_ok(),
            "config.toml runpod.api_key".bold()
        ),
        (None, None) => {
            println!(
                "{} api key not set — export {} or run {}",
                theme::icon_fail(),
                API_KEY_ENV.bold(),
                "mold config set runpod.api_key <key>".bold(),
            );
            return Err(AlreadyReported.into());
        }
    }

    // 2. Endpoint
    let endpoint = config
        .runpod
        .endpoint
        .clone()
        .unwrap_or_else(|| DEFAULT_ENDPOINT.to_string());
    println!("{} endpoint: {}", theme::icon_ok(), endpoint);

    // 3. Auth check via /user
    let client = match build_client() {
        Ok(c) => c,
        Err(e) => {
            explain_runpod_error(&e);
            return Err(AlreadyReported.into());
        }
    };
    match client.user().await {
        Ok(u) => {
            println!(
                "{} authenticated as {} (id: {})",
                theme::icon_ok(),
                u.email.bold(),
                u.id.dimmed()
            );
            println!(
                "{} balance: {} (spend: ${:.4}/hr)",
                theme::icon_ok(),
                format!("${:.2}", u.client_balance).bold(),
                u.current_spend_per_hr,
            );
            if let Some(limit) = u.spend_limit {
                println!("{} spend limit: ${:.2}", theme::icon_info(), limit);
            }
        }
        Err(e) => {
            explain_runpod_error(&e);
            return Err(AlreadyReported.into());
        }
    }
    Ok(())
}

/// `mold runpod gpus [--json]` — list available GPU types.
pub async fn run_gpus(json: bool, all: bool) -> Result<()> {
    let client = match build_client() {
        Ok(c) => c,
        Err(e) => {
            explain_runpod_error(&e);
            return Err(AlreadyReported.into());
        }
    };
    let gpus = match with_spinner("fetching gpu types…", client.gpu_types()).await {
        Ok(v) => v,
        Err(e) => {
            explain_runpod_error(&e);
            return Err(AlreadyReported.into());
        }
    };
    let filtered: Vec<&GpuType> = gpus
        .iter()
        .filter(|g| all || is_interesting_gpu(&g.display_name))
        .collect();
    if json {
        let payload = serde_json::to_string_pretty(&filtered)?;
        println!("{payload}");
        return Ok(());
    }
    println!(
        "{:<28}{:<6}{:<20}{:<10}{:<12}{:<12}",
        "GPU".bold(),
        "VRAM".bold(),
        "stock".bold(),
        "secure".bold(),
        "$/hr secure".bold(),
        "$/hr comm.".bold()
    );
    for g in filtered {
        let stock = g.stock_status.as_deref().unwrap_or("—");
        let stock_colored = color_stock(stock);
        let secure = if g.secure_cloud { "yes" } else { "no" };
        let rate = |cloud: &str| {
            g.hourly_price(cloud)
                .map(|price| format!("{price:.2}"))
                .unwrap_or_else(|| "—".into())
        };
        println!(
            "{:<28}{:<6}{:<20}{:<10}{:<12}{:<12}",
            g.display_name,
            g.memory_in_gb
                .map(|memory| format!("{memory}G"))
                .unwrap_or_else(|| "—".into()),
            stock_colored,
            secure,
            rate("SECURE"),
            rate("COMMUNITY")
        );
    }
    Ok(())
}

fn is_interesting_gpu(display: &str) -> bool {
    let d = display.to_lowercase();
    d.contains("4090")
        || d.contains("5090")
        || d.contains("l40")
        || d.contains("a100")
        || d.contains("h100")
        || d.contains("a6000")
        || d.contains("3090")
}

/// VRAM capacity (GB) for GPUs we care about. Used when picking a default
/// GPU that has enough headroom for the requested model.
fn gpu_vram_gb(display: &str) -> Option<u32> {
    let d = display.to_lowercase();
    if d.contains("h100") || d.contains("a100") {
        Some(80)
    } else if d.contains("l40") || d.contains("a6000") {
        Some(48)
    } else if d.contains("5090") {
        Some(32)
    } else if d.contains("4090") || d.contains("3090") {
        Some(24)
    } else {
        None
    }
}

/// Estimated peak VRAM requirement for a model, in GB. Combines manifest
/// weight size with a multiplier for activations + KV cache + image latents.
/// Returns `None` for unknown models — falls back to the legacy preference.
fn estimated_vram_need_gb(model_name: &str) -> Option<u32> {
    let resolved = mold_core::manifest::resolve_model_name(model_name);
    let manifest = mold_core::manifest::find_manifest(&resolved)?;
    // Headroom: weights ≈ 55% of peak VRAM during inference once text
    // encoder + latents + workspace are co-resident, so scale by ~1.8.
    let need = (manifest.total_size_gb() * 1.8).ceil() as u32;
    Some(need.max(12))
}

fn color_stock(stock: &str) -> String {
    match stock.to_lowercase().as_str() {
        "high" => stock.green().to_string(),
        "medium" => stock.yellow().to_string(),
        "low" => stock.red().to_string(),
        _ => stock.dimmed().to_string(),
    }
}

/// `mold runpod datacenters [--gpu <name>] [--json]` — list datacenters.
pub async fn run_datacenters(gpu_filter: Option<String>, json: bool) -> Result<()> {
    let client = match build_client() {
        Ok(c) => c,
        Err(e) => {
            explain_runpod_error(&e);
            return Err(AlreadyReported.into());
        }
    };
    let dcs = match with_spinner("fetching datacenters…", client.datacenters()).await {
        Ok(v) => v,
        Err(e) => {
            explain_runpod_error(&e);
            return Err(AlreadyReported.into());
        }
    };
    if json {
        println!("{}", serde_json::to_string_pretty(&dcs)?);
        return Ok(());
    }
    println!(
        "{:<18}{:<30}{:<10}",
        "datacenter".bold(),
        "match".bold(),
        "stock".bold()
    );
    for dc in &dcs {
        let location = dc.location.clone().unwrap_or_default();
        if let Some(filter) = &gpu_filter {
            let filter_l = filter.to_lowercase();
            let matches: Vec<_> = dc
                .gpu_availability
                .iter()
                .filter(|g| g.display_name.to_lowercase().contains(&filter_l))
                .collect();
            for g in matches {
                let stock = g.stock_status.as_deref().unwrap_or("—");
                println!(
                    "{:<18}{:<30}{:<10}",
                    dc.id,
                    format!("{} @ {}", g.display_name, location),
                    color_stock(stock)
                );
            }
        } else {
            println!("{:<18}{:<30}", dc.id, location);
        }
    }
    Ok(())
}

fn print_network_volume(volume: &NetworkVolume) {
    println!("{} {}", theme::icon_ok(), volume.id);
    println!("  name: {}", volume.name);
    println!("  size: {} GB", volume.size);
    println!("  datacenter: {}", volume.data_center_id);
}

pub async fn run_network_volume_list(json: bool) -> Result<()> {
    let client = build_client()?;
    let volumes = match with_spinner("listing network volumes…", client.network_volumes()).await {
        Ok(volumes) => volumes,
        Err(error) => {
            explain_runpod_error(&error);
            return Err(AlreadyReported.into());
        }
    };
    if json {
        println!("{}", serde_json::to_string_pretty(&volumes)?);
    } else if volumes.is_empty() {
        println!("{} no network volumes.", theme::icon_neutral());
    } else {
        println!(
            "{:<18}{:<28}{:<12}{}",
            "id".bold(),
            "name".bold(),
            "size".bold(),
            "datacenter".bold()
        );
        for volume in volumes {
            println!(
                "{:<18}{:<28}{:<12}{}",
                volume.id,
                volume.name,
                format!("{} GB", volume.size),
                volume.data_center_id
            );
        }
    }
    Ok(())
}

pub async fn run_network_volume_get(volume_id: String, json: bool) -> Result<()> {
    let client = build_client()?;
    let volume = match client.get_network_volume(&volume_id).await {
        Ok(volume) => volume,
        Err(error) => {
            explain_runpod_error(&error);
            return Err(AlreadyReported.into());
        }
    };
    if json {
        println!("{}", serde_json::to_string_pretty(&volume)?);
    } else {
        print_network_volume(&volume);
    }
    Ok(())
}

pub async fn run_network_volume_create(
    name: String,
    size: u32,
    datacenter: String,
    json: bool,
) -> Result<()> {
    let name = name.trim();
    let datacenter = datacenter.trim();
    if name.is_empty() {
        bail!("network volume name cannot be empty");
    }
    if datacenter.is_empty() {
        bail!("network volume datacenter cannot be empty");
    }
    if !valid_network_volume_size(size) {
        bail!(
            "network volume size must be between {NETWORK_VOLUME_MIN_GB} and {NETWORK_VOLUME_MAX_GB} GB"
        );
    }
    let client = build_client()?;
    let volume = client
        .create_network_volume(&CreateNetworkVolumeRequest {
            name: name.into(),
            size,
            data_center_id: datacenter.into(),
        })
        .await?;
    if json {
        println!("{}", serde_json::to_string_pretty(&volume)?);
    } else {
        print_network_volume(&volume);
        println!("  note: network volumes are billed until explicitly deleted");
    }
    Ok(())
}

pub async fn run_network_volume_update(
    volume_id: String,
    name: Option<String>,
    size: Option<u32>,
    json: bool,
) -> Result<()> {
    if name.is_none() && size.is_none() {
        bail!("provide --name and/or --size");
    }
    let name = name.map(|name| name.trim().to_string());
    if name.as_deref().is_some_and(str::is_empty) {
        bail!("network volume name cannot be empty");
    }
    if size.is_some_and(|size| !valid_network_volume_size(size)) {
        bail!(
            "network volume size must be between {NETWORK_VOLUME_MIN_GB} and {NETWORK_VOLUME_MAX_GB} GB"
        );
    }
    let client = build_client()?;
    if let Some(new_size) = size {
        let current = client.get_network_volume(&volume_id).await?;
        if new_size <= current.size {
            bail!(
                "network volume size can only grow (current: {} GB, requested: {new_size} GB)",
                current.size
            );
        }
    }
    let volume = client
        .update_network_volume(&volume_id, &UpdateNetworkVolumeRequest { name, size })
        .await?;
    if json {
        println!("{}", serde_json::to_string_pretty(&volume)?);
    } else {
        print_network_volume(&volume);
    }
    Ok(())
}

pub async fn run_network_volume_delete(volume_id: String, json: bool) -> Result<()> {
    let client = build_client()?;
    client
        .delete_network_volume_if_detached(&volume_id)
        .await
        .context("verify network volume is detached and delete it")?;
    if json {
        println!("{}", serde_json::json!({ "deleted": volume_id }));
    } else {
        println!("{} deleted network volume {volume_id}", theme::icon_ok());
    }
    Ok(())
}

/// `mold runpod list [--json]` — list pods.
pub async fn run_list(json: bool) -> Result<()> {
    reap_idle_warm_pod_if_needed().await;
    let client = match build_client() {
        Ok(c) => c,
        Err(e) => {
            explain_runpod_error(&e);
            return Err(AlreadyReported.into());
        }
    };
    let pods = match with_spinner("listing pods…", client.list_pods()).await {
        Ok(v) => v,
        Err(e) => {
            explain_runpod_error(&e);
            return Err(AlreadyReported.into());
        }
    };
    if json {
        println!("{}", serde_json::to_string_pretty(&pods)?);
        return Ok(());
    }
    if pods.is_empty() {
        println!("{} no pods.", theme::icon_neutral());
        return Ok(());
    }
    println!(
        "{:<18}{:<20}{:<18}{:<10}{:<10}",
        "id".bold(),
        "name".bold(),
        "gpu".bold(),
        "status".bold(),
        "$/hr".bold()
    );
    for p in &pods {
        let gpu = p.gpu_name().unwrap_or("—");
        let status = p.desired_status.clone();
        println!(
            "{:<18}{:<20}{:<18}{:<10}${:.2}",
            p.id,
            p.name.clone().unwrap_or_default(),
            gpu,
            status,
            p.cost_per_hr,
        );
    }
    Ok(())
}

/// `mold runpod get <pod-id> [--json]` — show pod details.
pub async fn run_get(pod_id: String, json: bool) -> Result<()> {
    let client = match build_client() {
        Ok(c) => c,
        Err(e) => {
            explain_runpod_error(&e);
            return Err(AlreadyReported.into());
        }
    };
    let pod = match with_spinner(&format!("fetching pod {pod_id}…"), client.get_pod(&pod_id)).await
    {
        Ok(p) => p,
        Err(e) => {
            explain_runpod_error(&e);
            return Err(AlreadyReported.into());
        }
    };
    if json {
        println!("{}", serde_json::to_string_pretty(&pod)?);
        return Ok(());
    }
    print_pod_detail(&pod);
    Ok(())
}

fn print_pod_detail(pod: &Pod) {
    println!("{} {} ({})", theme::icon_ok(), pod.id, pod.desired_status);
    if let Some(name) = &pod.name {
        println!("  name: {name}");
    }
    if let Some(image) = &pod.image_name {
        println!("  image: {image}");
    }
    if let Some(gpu_line) = pod_detail_gpu_line(pod) {
        println!("  gpu: {gpu_line}");
    }
    println!("  cost: ${:.2}/hr", pod.cost_per_hr);
    if pod.uptime_seconds > 0 {
        println!("  uptime: {}s", pod.uptime_seconds);
    }
    println!(
        "  volume: {} GB at {}",
        pod.volume_in_gb,
        pod.volume_mount_path.clone().unwrap_or_default()
    );
    let proxy = format!("https://{}-7680.proxy.runpod.net", pod.id);
    println!("  proxy: {}", proxy.cyan());
}

fn pod_detail_gpu_line(pod: &Pod) -> Option<String> {
    let gpu = pod.gpu_name()?;
    Some(
        match pod.datacenter_id().filter(|location| !location.is_empty()) {
            Some(location) => format!("{gpu} ({location})"),
            None => gpu.to_owned(),
        },
    )
}

/// `mold runpod usage [--since 7d] [--json]` — show spend summary.
pub async fn run_usage(since: Option<String>, json: bool) -> Result<()> {
    reap_idle_warm_pod_if_needed().await;
    let client = match build_client() {
        Ok(c) => c,
        Err(e) => {
            explain_runpod_error(&e);
            return Err(AlreadyReported.into());
        }
    };
    let user = match client.user().await {
        Ok(u) => u,
        Err(e) => {
            explain_runpod_error(&e);
            return Err(AlreadyReported.into());
        }
    };
    let pods = client.list_pods().await.unwrap_or_default();
    let history = read_history();
    let cutoff = since_to_epoch(since.as_deref())?;

    let session_spend: f64 = pods
        .iter()
        .map(|p| p.cost_per_hr * (p.uptime_seconds as f64 / 3600.0))
        .sum();
    let historical_spend: f64 = history
        .iter()
        .filter(|e| e.created_at >= cutoff)
        .map(|e| {
            let end = e.deleted_at.unwrap_or_else(now_epoch);
            let duration_hrs = (end.saturating_sub(e.created_at)) as f64 / 3600.0;
            e.cost_per_hr * duration_hrs
        })
        .sum();

    if json {
        let payload = serde_json::json!({
            "account": {
                "email": user.email,
                "balance": user.client_balance,
                "spend_per_hr": user.current_spend_per_hr,
            },
            "active_pods": pods.len(),
            "session_spend_usd": session_spend,
            "historical_spend_usd": historical_spend,
            "history_window_secs": now_epoch().saturating_sub(cutoff),
        });
        println!("{}", serde_json::to_string_pretty(&payload)?);
        return Ok(());
    }

    println!("{} account {}", theme::icon_info(), user.email.bold());
    println!("  balance: ${:.2}", user.client_balance);
    println!("  current burn rate: ${:.4}/hr", user.current_spend_per_hr);
    println!();
    println!(
        "{} active pods ({}): est session cost ${:.2}",
        theme::icon_info(),
        pods.len(),
        session_spend.max(0.0),
    );
    for p in &pods {
        let gpu = p.gpu_name().unwrap_or("—");
        println!(
            "  · {} {} {} — ${:.2}/hr × {:.2}h = ${:.2}",
            p.id,
            gpu,
            p.desired_status,
            p.cost_per_hr,
            p.uptime_seconds as f64 / 3600.0,
            p.cost_per_hr * (p.uptime_seconds as f64 / 3600.0),
        );
    }
    println!();
    println!(
        "{} historical spend (since {}): ${:.2}",
        theme::icon_info(),
        since.as_deref().unwrap_or("forever"),
        historical_spend.max(0.0),
    );
    Ok(())
}

fn since_to_epoch(since: Option<&str>) -> Result<u64> {
    let Some(s) = since else { return Ok(0) };
    if s.is_empty() {
        return Ok(0);
    }
    let (num, unit) = s.split_at(s.len() - 1);
    let n: u64 = num
        .parse()
        .with_context(|| format!("invalid --since value: {s}"))?;
    let secs = match unit {
        "s" => n,
        "m" => n * 60,
        "h" => n * 3600,
        "d" => n * 86400,
        "w" => n * 86400 * 7,
        _ => bail!("unknown --since unit '{unit}' — use s/m/h/d/w"),
    };
    Ok(now_epoch().saturating_sub(secs))
}

// ── Mutating subcommands ───────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct CreateOptions {
    pub name: Option<String>,
    pub gpu: Option<String>,
    pub datacenter: Option<String>,
    pub cloud: CloudType,
    pub volume_gb: u32,
    pub disk_gb: u32,
    pub image_tag: Option<String>,
    pub model: Option<String>,
    pub hf_token: bool,
    pub network_volume_id: Option<String>,
    pub dry_run: bool,
    pub json: bool,
}

#[derive(Debug, Clone, Copy)]
pub enum CloudType {
    Secure,
    Community,
}

impl CloudType {
    fn as_str(self) -> &'static str {
        match self {
            Self::Secure => "SECURE",
            Self::Community => "COMMUNITY",
        }
    }
}

impl std::str::FromStr for CloudType {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "secure" => Ok(Self::Secure),
            "community" => Ok(Self::Community),
            _ => bail!("invalid cloud type: {s} (expected secure or community)"),
        }
    }
}

/// Resolve the `HF_TOKEN` value to ship to the pod.
///
/// Priorities, in order:
///   1. Model is gated per its manifest → we NEED a token. Prefer the local
///      `HF_TOKEN` env var (guaranteed to work) and fall back to the RunPod
///      secret template `{{ RUNPOD_SECRET_HF_TOKEN }}` if the user has set
///      `--hf-token` but no local var.
///   2. Model is not gated AND `--hf-token` was passed → pass the RunPod
///      secret template (user explicitly opted in).
///   3. Otherwise → no `HF_TOKEN` in the pod env (don't leak credentials for
///      models that don't need them).
///
/// The local-env branch is the "smart" fallback the user asked for: if the
/// RunPod account doesn't have an `HF_TOKEN` secret configured, we still
/// make gated-model downloads work using the shell-exported token.
pub fn resolve_hf_token(opts: &CreateOptions) -> Option<String> {
    let manifest_gated = opts
        .model
        .as_deref()
        .map(mold_core::manifest::resolve_model_name)
        .and_then(|n| mold_core::manifest::find_manifest(&n))
        .is_some_and(|m| m.files.iter().any(|f| f.gated));

    let want_token = manifest_gated || opts.hf_token;
    if !want_token {
        return None;
    }

    // Local env wins — always works, no dependency on RunPod secret config.
    if let Ok(tok) = std::env::var("HF_TOKEN") {
        if !tok.is_empty() {
            return Some(tok);
        }
    }
    // No local env: fall back to the RunPod secret template. If the
    // account doesn't have this secret configured, the pod's model pull
    // will fail with 401 — caller should warn the user at plan time.
    Some("{{ RUNPOD_SECRET_HF_TOKEN }}".to_string())
}

/// Validate both provider fields, then retain the authoritative type ID as
/// the single source for allocation and distribution-image architecture.
fn ensure_published_runpod_gpu_identity<'a>(
    gpu_type_id: &'a str,
    display_name: &str,
) -> Result<&'a str> {
    mold_core::cuda_distribution::ensure_published_image_platform(gpu_type_id)?;
    mold_core::cuda_distribution::ensure_published_image_platform(display_name)?;
    Ok(gpu_type_id)
}

fn authoritative_runpod_gpu_type_id(gpu: &GpuType) -> String {
    gpu.allocation_type_id()
        .map(str::to_owned)
        .unwrap_or_default()
}

fn advertised_vram_gb(gpu: &GpuType) -> u32 {
    gpu.memory_in_gb
        .unwrap_or_else(|| gpu_vram_gb(&gpu.display_name).unwrap_or(0))
}

/// Build a `CreatePodRequest` from resolved defaults.
pub async fn build_create_request(
    opts: &CreateOptions,
    client: &RunPodClient,
    config: &Config,
) -> Result<CreatePodRequest> {
    // Resolve GPU — either user-supplied, config default, or cheapest available.
    let (gpu_name, gpu_display) = resolve_gpu(opts, client, config).await?;
    let supported_gpu_ids = client
        .supported_pod_gpu_type_ids()
        .await
        .context("load RunPod Pod API GPU types")?;
    let Some(gpu_name) = canonical_supported_gpu_type_id(&supported_gpu_ids, &gpu_name) else {
        bail!(
            "GPU type {gpu_name:?} is not currently accepted by RunPod's Pod API; \
             refresh inventory or choose another GPU"
        );
    };
    let gpu_name = gpu_name.to_string();
    let distribution_gpu_name = ensure_published_runpod_gpu_identity(&gpu_name, &gpu_display)?;

    // Resolve image tag.
    let image = if let Some(image_tag) = opts.image_tag.clone() {
        if image_tag.starts_with("sha256:") {
            let reference = format!("ghcr.io/utensils/mold@{image_tag}");
            if !mold_core::cuda_distribution::is_exact_image_digest_reference(&reference) {
                bail!("--image-tag sha256 digest must contain exactly 64 hexadecimal characters");
            }
            reference
        } else {
            format!("ghcr.io/utensils/mold:{image_tag}")
        }
    } else {
        mold_core::cuda_distribution::resolve_distribution_image_reference(
            mold_core::cuda_distribution::OFFICIAL_IMAGE_REPOSITORY,
            distribution_gpu_name,
            mold_core::cuda_distribution::distribution_image_version(),
        )
        .await?
    };

    // Resolve volume + network volume.
    let volume_id = opts
        .network_volume_id
        .clone()
        .or_else(|| config.runpod.default_network_volume_id.clone());
    let network_volume = match volume_id.as_deref() {
        Some(id) => Some(
            client
                .get_network_volume(id)
                .await
                .with_context(|| format!("load network volume {id}"))?,
        ),
        None => None,
    };

    // A network volume is physically scoped to one Secure Cloud datacenter;
    // it is authoritative over explicit/default cloud and DC preferences.
    let fallback_dc = if network_volume.is_none() {
        resolve_datacenter(opts, client, config, &gpu_display).await?
    } else {
        None
    };
    let (dc, cloud_type, workspace_gb) = network_volume_placement(
        network_volume.as_ref(),
        fallback_dc,
        opts.cloud.as_str(),
        opts.volume_gb,
    );

    // Build env.
    let mut env = serde_json::Map::new();
    if let Some(model) = opts.model.clone() {
        env.insert("MOLD_DEFAULT_MODEL".into(), model.into());
    }
    env.insert("MOLD_LOG".into(), "info".into());
    if let Some(tok) = resolve_hf_token(opts) {
        env.insert("HF_TOKEN".into(), tok.into());
    }

    let name = opts
        .name
        .clone()
        .unwrap_or_else(|| format!("mold-{}", short_timestamp()));
    Ok(CreatePodRequest {
        name,
        image_name: image,
        gpu_type_ids: vec![gpu_name],
        cloud_type,
        data_center_ids: dc.map(|id| vec![id]),
        gpu_count: 1,
        container_disk_in_gb: opts.disk_gb,
        volume_in_gb: workspace_gb,
        volume_mount_path: "/workspace".to_string(),
        ports: vec!["7680/http".into(), "22/tcp".into()],
        env,
        network_volume_id: volume_id,
    })
}

async fn resolve_gpu(
    opts: &CreateOptions,
    client: &RunPodClient,
    config: &Config,
) -> Result<(String, String)> {
    if let Some(gpu) = opts.gpu.clone() {
        let resolved = normalize_gpu_id(&gpu);
        if resolved.is_empty() {
            bail!("GPU type cannot be blank");
        }
        return Ok((resolved.clone(), friendly_gpu_name(&resolved)));
    }
    if let Some(gpu) = config.runpod.default_gpu.clone() {
        let resolved = normalize_gpu_id(&gpu);
        if resolved.is_empty() {
            bail!("configured RunPod GPU type cannot be blank");
        }
        return Ok((resolved.clone(), friendly_gpu_name(&resolved)));
    }
    let gpus = client.gpu_types().await?;

    // If a model is requested, size the GPU to its VRAM need. LTX-2 22B fp8
    // doesn't fit on a 24GB 4090 once the encoder + latents + image are
    // resident, so we bump up to a 5090/L40S automatically.
    let need = opts
        .model
        .as_deref()
        .and_then(estimated_vram_need_gb)
        .unwrap_or(18); // FLUX/SDXL-size default

    // Preference: cheapest GPU that (a) has enough VRAM and (b) has stock.
    // Ordering is by ascending VRAM so we don't overshoot on small models.
    let ranked: &[&str] = &[
        "RTX 4090",
        "RTX 3090",
        "RTX 5090",
        "L40",
        "L40S",
        "RTX A6000",
        "A100 PCIe",
        "A100 SXM",
        "H100 NVL",
        "H100 SXM",
    ];
    for preferred in ranked {
        if let Some(g) = gpus.iter().find(|g| g.display_name == *preferred) {
            let stock = g.stock_status.as_deref().unwrap_or("");
            if advertised_vram_gb(g) >= need && matches!(stock, "High" | "Medium") {
                return Ok((authoritative_runpod_gpu_type_id(g), g.display_name.clone()));
            }
        }
    }
    // Fallback: any interesting GPU with High stock + enough VRAM.
    for g in &gpus {
        if g.stock_status.as_deref() == Some("High")
            && is_interesting_gpu(&g.display_name)
            && advertised_vram_gb(g) >= need
        {
            return Ok((authoritative_runpod_gpu_type_id(g), g.display_name.clone()));
        }
    }
    bail!(
        "no GPUs with ≥{need}GB VRAM and High/Medium stock — try explicit --gpu \
         (model estimated to need {need}GB)"
    )
}

async fn resolve_datacenter(
    opts: &CreateOptions,
    client: &RunPodClient,
    config: &Config,
    _gpu_display: &str,
) -> Result<Option<String>> {
    // Priority:
    //   1. Explicit --dc wins.
    //   2. config.runpod.default_datacenter.
    //   3. If a network volume is attached (via --network-volume or
    //      config.runpod.default_network_volume_id), look up the
    //      volume's datacenter and pin to it. Network volumes are
    //      datacenter-scoped — RunPod will reject the pod if we try to
    //      mount one from a different region.
    //   4. Otherwise leave `dataCenterIds` unset and let RunPod pick.
    //      Retry logic in `ensure_pod` handles DC fallback if the
    //      scheduler can't place the pod.
    if let Some(dc) = opts.datacenter.clone() {
        return Ok(Some(dc));
    }
    if let Some(dc) = config.runpod.default_datacenter.clone() {
        return Ok(Some(dc));
    }
    let volume_id = opts
        .network_volume_id
        .clone()
        .or_else(|| config.runpod.default_network_volume_id.clone());
    if let Some(vid) = volume_id {
        if let Ok(volumes) = client.network_volumes().await {
            if let Some(v) = volumes.iter().find(|v| v.id == vid) {
                if !v.data_center_id.is_empty() {
                    return Ok(Some(v.data_center_id.clone()));
                }
            }
        }
    }
    Ok(None)
}

/// Convert RunPod's `displayName` (e.g. `"RTX 5090"`) to the `gpuId` the REST
/// API actually accepts (e.g. `"NVIDIA GeForce RTX 5090"`).
pub fn friendly_to_gpu_id(display: &str) -> String {
    legacy_gpu_type_id_from_display_name(display).to_string()
}

fn normalize_gpu_id(user: &str) -> String {
    // Allow users to pass the short alias (e.g. "4090", "5090", "a100") or
    // the full NVIDIA name. If we don't recognize it, pass through unchanged.
    let user = user.trim();
    let u = user.to_lowercase();
    if u == "4090" || u == "rtx4090" || u == "rtx 4090" {
        return friendly_to_gpu_id("RTX 4090");
    }
    if u == "5090" || u == "rtx5090" || u == "rtx 5090" {
        return friendly_to_gpu_id("RTX 5090");
    }
    if u == "3090" || u == "rtx3090" {
        return friendly_to_gpu_id("RTX 3090");
    }
    if u == "l40s" {
        return friendly_to_gpu_id("L40S");
    }
    if u == "l40" {
        return friendly_to_gpu_id("L40");
    }
    if u == "a100" {
        return friendly_to_gpu_id("A100 PCIe");
    }
    if u == "h100" {
        return friendly_to_gpu_id("H100 SXM");
    }
    if u == "a6000" {
        return friendly_to_gpu_id("RTX A6000");
    }
    if user.contains("NVIDIA") {
        return user.to_string();
    }
    // Fallback — pass through (assume user typed the NVIDIA name)
    user.to_string()
}

fn friendly_gpu_name(gpu_id: &str) -> String {
    // Reverse of friendly_to_gpu_id — best-effort.
    if gpu_id.contains("5090") {
        "RTX 5090".into()
    } else if gpu_id.contains("4090") {
        "RTX 4090".into()
    } else if gpu_id.contains("3090") {
        "RTX 3090".into()
    } else if gpu_id.contains("A100") {
        "A100 PCIe".into()
    } else if gpu_id.contains("H100") {
        "H100 SXM".into()
    } else if gpu_id.contains("L40S") {
        "L40S".into()
    } else if gpu_id.contains("L40") {
        "L40".into()
    } else if gpu_id.contains("A6000") {
        "RTX A6000".into()
    } else {
        gpu_id.to_string()
    }
}

fn short_timestamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs().to_string())
        .unwrap_or_else(|_| "pod".into())
}

async fn require_create_model_activation(config: &mut Config, opts: &CreateOptions) -> Result<()> {
    if let Some(model) = opts.model.as_deref() {
        crate::catalog_bridge::require_cloud_model_activation(config, model).await?;
    }
    Ok(())
}

async fn require_run_model_activation(config: &mut Config, opts: &RunOptions) -> Result<String> {
    let effective_model = opts
        .model
        .clone()
        .unwrap_or_else(|| config.default_model.clone());
    crate::catalog_bridge::require_cloud_model_activation(config, &effective_model).await?;

    if let Some(preload_model) = opts.create.model.as_deref() {
        if preload_model != effective_model {
            crate::catalog_bridge::require_cloud_model_activation(config, preload_model).await?;
        }
    }
    Ok(effective_model)
}

/// `mold runpod create` — create a new pod.
pub async fn run_create(opts: CreateOptions) -> Result<()> {
    let mut config = Config::load_or_default();
    require_create_model_activation(&mut config, &opts).await?;

    reap_idle_warm_pod_if_needed().await;
    let client = match build_client() {
        Ok(c) => c,
        Err(e) => {
            explain_runpod_error(&e);
            return Err(AlreadyReported.into());
        }
    };
    let req = build_create_request(&opts, &client, &config).await?;
    if !opts.dry_run {
        enforce_cost_alert(&client, &config).await?;
    }

    if opts.dry_run {
        if opts.json {
            println!("{}", serde_json::to_string_pretty(&req)?);
        } else {
            print_create_plan(&req);
        }
        return Ok(());
    }

    if opts.json {
        // nothing — we'll print the pod JSON after creation
    } else {
        print_create_plan(&req);
    }

    let pod = match with_spinner("creating pod…", client.create_pod(&req)).await {
        Ok(p) => p,
        Err(e) => {
            explain_runpod_error(&e);
            return Err(AlreadyReported.into());
        }
    };

    // Manually-created pods are tracked in history for spend reporting,
    // but NOT enrolled as the warm `last_pod_id`. The idle reaper only
    // cleans up pods that `mold runpod run` parked for reuse — pods the
    // user explicitly created should only be deleted when the user says
    // so (`mold runpod delete <id>`).
    let gpu_display = pod
        .gpu_name()
        .map(str::to_owned)
        .unwrap_or_else(|| friendly_gpu_name(&req.gpu_type_ids[0]));
    let _ = append_history(&HistoryEntry {
        pod_id: pod.id.clone(),
        created_at: now_epoch(),
        deleted_at: None,
        cost_per_hr: pod.cost_per_hr,
        gpu: gpu_display,
        model: opts.model.clone(),
        prompt: None,
    });

    if opts.json {
        println!("{}", serde_json::to_string_pretty(&pod)?);
    } else {
        println!();
        println!("{} pod created", theme::icon_ok());
        print_pod_detail(&pod);
        println!();
        let proxy = format!("https://{}-7680.proxy.runpod.net", pod.id);
        println!(
            "  {} {}",
            theme::prefix_hint(),
            format!("export MOLD_HOST={proxy}").bold()
        );
        println!("  {} gallery: {}", theme::prefix_hint(), proxy.cyan());
    }
    Ok(())
}

fn print_create_plan(req: &CreatePodRequest) {
    println!("{} plan", theme::icon_info());
    println!("  name: {}", req.name);
    println!("  image: {}", req.image_name);
    println!("  gpu: {} × {}", req.gpu_count, req.gpu_type_ids[0]);
    println!("  cloud: {}", req.cloud_type);
    if let Some(dcs) = &req.data_center_ids {
        println!("  datacenter: {}", dcs.join(","));
    } else {
        println!("  datacenter: (any)");
    }
    println!(
        "  disk: {} GB / volume: {} GB @ {}",
        req.container_disk_in_gb, req.volume_in_gb, req.volume_mount_path
    );
    println!("  ports: {}", req.ports.join(","));
    if !req.env.is_empty() {
        let mut keys: Vec<&String> = req.env.keys().collect();
        keys.sort();
        println!(
            "  env: {}",
            keys.iter()
                .map(|k| k.as_str())
                .collect::<Vec<_>>()
                .join(",")
        );
        // Show HF_TOKEN source so users can tell template vs local env.
        if let Some(v) = req.env.get("HF_TOKEN").and_then(|v| v.as_str()) {
            let source = if v.contains("RUNPOD_SECRET") {
                "(RunPod secret)"
            } else {
                "(local HF_TOKEN env)"
            };
            println!("  hf_token: {source}");
        }
    }
    if let Some(nv) = &req.network_volume_id {
        println!("  network volume: {nv}");
    }
}

/// `mold runpod stop <pod-id>` — stop a pod (billing paused, storage kept).
pub async fn run_stop(pod_id: String, json: bool) -> Result<()> {
    let client = match build_client() {
        Ok(c) => c,
        Err(e) => {
            explain_runpod_error(&e);
            return Err(AlreadyReported.into());
        }
    };
    match with_spinner(&format!("stopping {pod_id}…"), client.stop_pod(&pod_id)).await {
        Ok(_) => {
            if json {
                println!("{}", serde_json::json!({"stopped": true, "id": pod_id}));
            } else {
                println!("{} stopped {}", theme::icon_ok(), pod_id);
            }
            Ok(())
        }
        Err(e) => {
            explain_runpod_error(&e);
            Err(AlreadyReported.into())
        }
    }
}

/// `mold runpod start <pod-id>` — resume a stopped pod.
pub async fn run_start(pod_id: String, json: bool) -> Result<()> {
    let client = match build_client() {
        Ok(c) => c,
        Err(e) => {
            explain_runpod_error(&e);
            return Err(AlreadyReported.into());
        }
    };
    match with_spinner(&format!("starting {pod_id}…"), client.start_pod(&pod_id)).await {
        Ok(_) => {
            if json {
                println!("{}", serde_json::json!({"started": true, "id": pod_id}));
            } else {
                println!("{} started {}", theme::icon_ok(), pod_id);
            }
            Ok(())
        }
        Err(e) => {
            explain_runpod_error(&e);
            Err(AlreadyReported.into())
        }
    }
}

/// `mold runpod delete <pod-id> [--force]` — delete a pod.
///
/// No interactive confirmation: passing an explicit pod id is enough signal
/// of intent. `--force` is retained as a no-op alias for backward compat.
pub async fn run_delete(pod_id: String, _force: bool, json: bool) -> Result<()> {
    let client = match build_client() {
        Ok(c) => c,
        Err(e) => {
            explain_runpod_error(&e);
            return Err(AlreadyReported.into());
        }
    };
    match with_spinner(&format!("deleting {pod_id}…"), client.delete_pod(&pod_id)).await {
        Ok(_) => {
            // Update history + warm-pod state.
            let mut state = load_state();
            if state.last_pod_id.as_deref() == Some(pod_id.as_str()) {
                state.last_pod_id = None;
                state.last_pod_created_at = None;
                state.last_pod_last_used_at = None;
                state.last_pod_gpu = None;
                state.last_pod_cost_per_hr = None;
                let _ = save_state(&state);
            }
            mark_history_deleted(&pod_id);
            if json {
                println!("{}", serde_json::json!({"deleted": true, "id": pod_id}));
            } else {
                println!("{} deleted {}", theme::icon_ok(), pod_id);
            }
            Ok(())
        }
        Err(e) => {
            explain_runpod_error(&e);
            Err(AlreadyReported.into())
        }
    }
}

fn mark_history_deleted(pod_id: &str) {
    let Ok(path) = history_path() else { return };
    let Ok(text) = std::fs::read_to_string(&path) else {
        return;
    };
    let mut entries: Vec<HistoryEntry> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| serde_json::from_str::<HistoryEntry>(l).ok())
        .collect();
    for e in entries.iter_mut() {
        if e.pod_id == pod_id && e.deleted_at.is_none() {
            e.deleted_at = Some(now_epoch());
        }
    }
    let mut buf = String::new();
    for e in &entries {
        if let Ok(l) = serde_json::to_string(e) {
            buf.push_str(&l);
            buf.push('\n');
        }
    }
    let _ = std::fs::write(&path, buf);
}

async fn delete_tracked_pod(client: &RunPodClient, pod_id: &str) -> Result<()> {
    delete_then_finalize(client, pod_id, || {
        mark_history_deleted(pod_id);
        let mut state = load_state();
        if state.last_pod_id.as_deref() == Some(pod_id) {
            state.last_pod_id = None;
            state.last_pod_created_at = None;
            state.last_pod_last_used_at = None;
            state.last_pod_gpu = None;
            state.last_pod_cost_per_hr = None;
            save_state(&state)?;
        }
        Ok(())
    })
    .await
}

async fn delete_then_finalize(
    client: &RunPodClient,
    pod_id: &str,
    finalize: impl FnOnce() -> Result<()>,
) -> Result<()> {
    client.delete_pod(pod_id).await.with_context(|| {
        format!("delete pod {pod_id}; it remains tracked and may still be billing")
    })?;
    finalize()
}

fn tracked_pod_is_absent(error: &anyhow::Error) -> bool {
    matches!(
        error.downcast_ref::<MoldError>(),
        Some(MoldError::RunPodNotFound(_))
    )
}

fn pod_is_scheduled(pod: &Pod) -> bool {
    pod.uptime_seconds > 0 || pod.gpu_name().is_some_and(|name| !name.is_empty())
}

/// `mold runpod connect <pod-id>` — print an `export MOLD_HOST=…` snippet.
pub async fn run_connect(pod_id: String, check: bool) -> Result<()> {
    if check {
        // Optionally verify the pod is reachable.
        let client = build_client().ok();
        if let Some(c) = client {
            if let Ok(pod) = c.get_pod(&pod_id).await {
                eprintln!(
                    "{} pod {} ({})",
                    theme::icon_ok(),
                    pod.id,
                    pod.desired_status
                );
            }
        }
    }
    let proxy = format!("https://{pod_id}-7680.proxy.runpod.net");
    println!("export MOLD_HOST={proxy}");
    eprintln!("{} gallery: {}", theme::prefix_hint(), proxy.cyan());
    Ok(())
}

/// `mold runpod logs <pod-id>` — hand off to RunPod's supported web console.
pub async fn run_logs(pod_id: String, follow: bool) -> Result<()> {
    let client = match build_client() {
        Ok(c) => c,
        Err(e) => {
            explain_runpod_error(&e);
            return Err(AlreadyReported.into());
        }
    };
    let pod = match client.get_pod(&pod_id).await {
        Ok(pod) => pod,
        Err(e) => {
            explain_runpod_error(&e);
            return Err(AlreadyReported.into());
        }
    };
    if follow {
        eprintln!(
            "{} RunPod does not expose Pod logs through its REST API; --follow cannot stream them.",
            theme::prefix_hint()
        );
    }
    println!("pod: {} ({})", pod.id, pod.desired_status);
    println!("logs: https://www.runpod.io/console/pods");
    println!("Open the Pod row and select Logs (container or system).");
    Ok(())
}

// ── Killer feature: run ────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct RunOptions {
    pub prompt: String,
    pub model: Option<String>,
    pub output_dir: std::path::PathBuf,
    pub keep: bool,
    pub seed: Option<u64>,
    pub steps: Option<u32>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub create: CreateOptions,
    pub wait_ready_timeout_secs: u64,
}

/// `mold runpod run "<prompt>"` — end-to-end: reuse warm pod or create one,
/// generate, save to local repo, park (or delete with --keep).
pub async fn run_run(opts: RunOptions) -> Result<()> {
    let mut config = Config::load_or_default();
    let effective_model = require_run_model_activation(&mut config, &opts).await?;

    reap_idle_warm_pod_if_needed().await;
    let client = match build_client() {
        Ok(c) => c,
        Err(e) => {
            explain_runpod_error(&e);
            return Err(AlreadyReported.into());
        }
    };

    // Warn up front if this model is gated but we have no HF_TOKEN source
    // at all — the user is about to burn pod startup time for nothing.
    {
        let model = effective_model.as_str();
        let resolved = mold_core::manifest::resolve_model_name(model);
        let gated = mold_core::manifest::find_manifest(&resolved)
            .is_some_and(|m| m.files.iter().any(|f| f.gated));
        let local_token = std::env::var("HF_TOKEN").ok().filter(|v| !v.is_empty());
        if gated && local_token.is_none() {
            eprintln!(
                "{} {} is gated on Hugging Face. No local HF_TOKEN detected — \
                 falling back to the RunPod secret named {}. If that secret \
                 isn't configured in your RunPod account, the model pull will \
                 fail inside the pod.",
                theme::icon_warn(),
                resolved,
                "HF_TOKEN".bold(),
            );
        }
    }

    // Pick or create a pod.
    let pod = ensure_pod(&client, &config, &opts).await?;
    println!(
        "{} using pod {} on {}",
        theme::icon_ok(),
        pod.id.bold(),
        pod.gpu_name().unwrap_or_default()
    );

    // Wait for proxy readiness.
    let mold_host = format!("https://{}-7680.proxy.runpod.net", pod.id);
    wait_for_mold(&mold_host, opts.wait_ready_timeout_secs).await?;

    // Delegate to the existing run pipeline via MoldClient.
    std::env::set_var("MOLD_HOST", &mold_host);
    std::fs::create_dir_all(&opts.output_dir)
        .with_context(|| format!("create output dir {}", opts.output_dir.display()))?;

    // Build a request using per-model defaults from config. This mirrors
    // the behaviour of `commands::run` so `mold runpod run` and local
    // `mold run` produce the same image for the same inputs.
    let model = effective_model;
    // Pick a sensible output format: video families always need video
    // containers. PNG is the fallback for image models.
    let is_video_model =
        mold_core::manifest::find_manifest(&mold_core::manifest::resolve_model_name(&model))
            .is_some_and(|m| matches!(m.family.as_str(), "ltx-video" | "ltx2"));
    let output_format = if is_video_model {
        mold_core::OutputFormat::Mp4
    } else {
        mold_core::OutputFormat::Png
    };
    let ext = if is_video_model { "mp4" } else { "png" };
    let filename = format!("runpod-{}-{}.{ext}", pod.id, short_timestamp());
    let output_path = opts.output_dir.join(&filename);
    let model_cfg = config.models.get(&model).cloned().unwrap_or_default();
    let effective_guidance = model_cfg.effective_guidance();
    let effective_width = opts
        .width
        .unwrap_or_else(|| model_cfg.effective_width(&config));
    let effective_height = opts
        .height
        .unwrap_or_else(|| model_cfg.effective_height(&config));
    let effective_steps = opts
        .steps
        .unwrap_or_else(|| model_cfg.default_steps.unwrap_or(config.default_steps));
    let negative_prompt = model_cfg
        .negative_prompt
        .clone()
        .or_else(|| config.default_negative_prompt.clone());
    let req = mold_core::GenerateRequest {
        offload: None,
        mesh: None,
        video_only: None,
        collection: None,
        tags: None,
        title: None,
        source_fit: None,
        hdr_exr_dir: None,
        hdr_exr_full_float: false,
        guidance_overrides: None,
        sample_shift: None,
        distill_strength_high: None,
        distill_strength_low: None,
        prompt: opts.prompt.clone(),
        negative_prompt,
        model,
        width: effective_width,
        height: effective_height,
        steps: effective_steps,
        guidance: effective_guidance,
        seed: opts.seed,
        batch_size: 1,
        output_format: Some(output_format),
        embed_metadata: None,
        scheduler: None,
        cfg_plus: None,
        source_image: None,
        source_image_name: None,
        edit_images: None,
        references: None,
        strength: 0.75,
        mask_image: None,
        control_image: None,
        control_model: None,
        control_scale: 1.0,
        expand: None,
        save_to_gallery: None,
        original_prompt: None,
        prompt_transform: None,
        batch_id: None,
        batch_index: None,
        batch_count: None,
        lora: None,
        frames: None,
        fps: None,
        upscale_model: None,
        gif_preview: false,
        enable_audio: None,
        audio_file: None,
        audio_file_path: None,
        source_video: None,
        source_video_path: None,
        extend_video: None,
        extend_video_path: None,
        extend_overlap_frames: None,
        keyframes: None,
        pipeline: None,
        ic_lora_control: None,
        loras: None,
        retake_range: None,
        spatial_upscale: None,
        temporal_upscale: None,
        placement: None,
        id_image: None,
        id_image_name: None,
        id_weight: None,
        id_start_step: None,
        id_images: None,
        id_image_names: None,
        true_cfg: None,
        cfg_start_step: None,
    };
    let http = mold_core::MoldClient::new(&mold_host);

    println!(
        "{} generating → {}",
        theme::icon_info(),
        output_path.display().to_string().cyan()
    );

    let artifact = match generate_with_runpod_transport(&http, &req).await {
        Ok(artifact) => artifact,
        Err(error) => {
            let msg = format!("{error:#}");
            let gpu = pod.gpu_name().unwrap_or_default();
            if msg.contains("404") {
                bail!(
                    "generation failed: the pod does not serve \
                     POST /api/generation-batches. The mold binary in the \
                     container may not match the host GPU ({gpu}). Try \
                     `--image-tag latest-sm80` for broad compat, or check pod \
                     logs via `mold runpod logs {}`.",
                    pod.id
                );
            }
            return Err(error);
        }
    };
    std::fs::write(&output_path, &artifact.bytes)?;
    match (
        artifact.result.generation_time_ms,
        artifact.result.seed.or(Some(artifact.metadata.seed)),
    ) {
        (Some(elapsed_ms), Some(seed)) => println!(
            "{} saved {} ({:.1}s on seed {seed}, durable output {})",
            theme::icon_done(),
            output_path.display(),
            elapsed_ms as f64 / 1000.0,
            artifact.filename
        ),
        _ => println!(
            "{} saved {} (durable output {})",
            theme::icon_done(),
            output_path.display(),
            artifact.filename
        ),
    }

    // Update state + history.
    let mut state = load_state();
    state.last_pod_id = Some(pod.id.clone());
    state.last_pod_last_used_at = Some(now_epoch());
    let _ = save_state(&state);

    // Surface the web gallery URL — the container bundles the SPA now,
    // so every generation lands on a browsable page.
    println!("  {} browse at {}", theme::prefix_hint(), mold_host.cyan());

    if !opts.keep && config.runpod.auto_teardown {
        println!(
            "{} auto-teardown enabled — deleting pod",
            theme::icon_info()
        );
        delete_tracked_pod(&client, &pod.id).await?;
    } else if opts.keep {
        println!(
            "{} pod {} kept running (--keep). Delete with {}",
            theme::icon_info(),
            pod.id,
            format!("mold runpod delete {}", pod.id).bold()
        );
    } else {
        let idle = config.runpod.auto_teardown_idle_mins;
        if idle > 0 {
            println!(
                "{} pod {} parked for reuse — auto-deleted on next {} invocation \
                 if idle > {} min",
                theme::icon_info(),
                pod.id,
                "mold runpod".bold(),
                idle,
            );
        } else {
            println!(
                "{} pod {} parked for reuse — delete with {}",
                theme::icon_info(),
                pod.id,
                format!("mold runpod delete {}", pod.id).bold()
            );
        }
    }
    Ok(())
}

/// Check the session spend against `runpod.cost_alert_usd` and abort with a
/// clear error if exceeded. `0.0` (the default) disables the guard. Uses the
/// same "active pods × hourly rate × uptime" calculation as `mold runpod usage`.
pub async fn enforce_cost_alert(client: &RunPodClient, config: &Config) -> Result<()> {
    let threshold = config.runpod.cost_alert_usd;
    if threshold <= 0.0 {
        return Ok(());
    }
    let Ok(pods) = client.list_pods().await else {
        return Ok(());
    };
    let session_spend: f64 = pods
        .iter()
        .map(|p| p.cost_per_hr * (p.uptime_seconds as f64 / 3600.0))
        .sum();
    if session_spend >= threshold {
        eprintln!(
            "{} cost alert: session spend ${:.2} reached threshold ${:.2} \
             ({} active pods). Aborting.",
            theme::prefix_error(),
            session_spend,
            threshold,
            pods.len()
        );
        eprintln!(
            "       {} increase with {} or delete pods with {}",
            theme::prefix_hint(),
            "mold config set runpod.cost_alert_usd <N>".bold(),
            "mold runpod list".bold(),
        );
        return Err(AlreadyReported.into());
    }
    Ok(())
}

/// Lazy idle-reap: if the warm pod has been idle longer than
/// `runpod.auto_teardown_idle_mins`, delete it. Called by every
/// mutating runpod subcommand so users don't pay for abandoned pods.
///
/// Returns `true` if a pod was reaped.
pub async fn reap_idle_warm_pod_if_needed() -> bool {
    let config = Config::load_or_default();
    let idle_mins = config.runpod.auto_teardown_idle_mins;
    if idle_mins == 0 {
        return false;
    }
    let state = load_state();
    let Some(pod_id) = state.last_pod_id.clone() else {
        return false;
    };
    let Some(last_used) = state.last_pod_last_used_at else {
        return false;
    };
    let idle_secs = now_epoch().saturating_sub(last_used);
    if idle_secs < (idle_mins as u64) * 60 {
        return false;
    }
    // Idle window exceeded — delete.
    let Ok(client) = build_client() else {
        return false;
    };
    // Verify the pod still exists + is running (not already gone).
    let pod = match client.get_pod(&pod_id).await {
        Ok(pod) => pod,
        Err(error) if tracked_pod_is_absent(&error) => {
            let mut state = state;
            state.last_pod_id = None;
            state.last_pod_created_at = None;
            state.last_pod_last_used_at = None;
            state.last_pod_gpu = None;
            state.last_pod_cost_per_hr = None;
            let _ = save_state(&state);
            return false;
        }
        Err(error) => {
            eprintln!(
                "{} could not verify tracked pod {pod_id}; retaining it: {error:#}",
                theme::icon_warn()
            );
            return false;
        }
    };
    if pod.desired_status != "RUNNING" {
        return false;
    }
    eprintln!(
        "{} reaping idle warm pod {} (unused for {}m)",
        theme::icon_info(),
        pod_id,
        idle_secs / 60,
    );
    match delete_tracked_pod(&client, &pod_id).await {
        Ok(()) => true,
        Err(error) => {
            eprintln!("{} {error:#}", theme::icon_warn());
            false
        }
    }
}

/// Outcome of racing `wait_for_schedule` against Ctrl-C.
enum WaitOutcome {
    Scheduled(Box<Pod>),
    Cancelled,
    Failed(anyhow::Error),
}

async fn ensure_pod(client: &RunPodClient, config: &Config, opts: &RunOptions) -> Result<Pod> {
    // Abort before any billable action if session spend exceeds the guard.
    enforce_cost_alert(client, config).await?;
    // 1. Warm pod? Only reuse if actually scheduled on a machine. If a
    //    previous run saved a pod id that never scheduled (e.g. Ctrl-C
    //    during wait) delete it before we create a new one — otherwise it
    //    keeps billing as an orphan.
    let state = load_state();
    if let Some(id) = state.last_pod_id.clone() {
        match client.get_pod(&id).await {
            Ok(pod) if pod.desired_status == "RUNNING" => {
                // Respect explicit overrides before reuse. If the user
                // passed --gpu or --dc (or pinned one via config), the
                // warm pod has to match, otherwise a fresh provision is
                // strictly what was asked for.
                let warm_gpu = pod
                    .gpu_name()
                    .map(str::to_owned)
                    .or_else(|| state.last_pod_gpu.clone())
                    .unwrap_or_default();
                let warm_dc = pod.datacenter_id().unwrap_or_default();
                let want_gpu = opts
                    .create
                    .gpu
                    .clone()
                    .or_else(|| config.runpod.default_gpu.clone());
                let want_volume = opts
                    .create
                    .network_volume_id
                    .clone()
                    .or_else(|| config.runpod.default_network_volume_id.clone());
                let want_dc = warm_datacenter_requirement(
                    opts.create.datacenter.as_deref(),
                    config.runpod.default_datacenter.as_deref(),
                    want_volume.as_deref(),
                );
                let gpu_mismatch = want_gpu
                    .as_deref()
                    .map(normalize_gpu_id)
                    .map(|want| {
                        let warm_id = normalize_gpu_id(&warm_gpu);
                        !warm_id.is_empty()
                            && !warm_id.eq_ignore_ascii_case(&want)
                            && !warm_gpu.eq_ignore_ascii_case(want.as_str())
                    })
                    .unwrap_or(false);
                let dc_mismatch = want_dc
                    .as_deref()
                    .map(|want| !warm_dc.is_empty() && !warm_dc.eq_ignore_ascii_case(want))
                    .unwrap_or(false);
                let volume_mismatch = network_volume_selection_mismatch(
                    pod.attached_network_volume_id(),
                    want_volume.as_deref(),
                );
                if gpu_mismatch || dc_mismatch || volume_mismatch {
                    eprintln!(
                        "{} warm pod {} does not match the requested GPU, datacenter, or network volume — \
                         deleting warm pod and provisioning fresh",
                        theme::icon_warn(),
                        id,
                    );
                    delete_tracked_pod(client, &id).await?;
                } else {
                    // REST v1 can leave `runtime` / `machine.gpu_display_name`
                    // unpopulated on fully-booted pods (same reason we had
                    // to rework `wait_for_schedule`). Don't use those as a
                    // usability test — instead probe the proxy directly.
                    // If the proxy answers at all, the pod is scheduled
                    // and routable, which is all we need.
                    let proxy_url = format!("https://{id}-7680.proxy.runpod.net/api/status");
                    let http = reqwest::Client::builder()
                        .timeout(std::time::Duration::from_secs(5))
                        .build()
                        .unwrap_or_default();
                    if http.get(&proxy_url).send().await.is_ok() {
                        return Ok(pod);
                    }
                    // Running according to RunPod but the proxy isn't
                    // answering — treat as stale, kill it to avoid leaked
                    // billing.
                    eprintln!(
                        "{} stale pod {} (not reachable via proxy) — deleting",
                        theme::icon_warn(),
                        id
                    );
                    delete_tracked_pod(client, &id).await?;
                }
            }
            Ok(_) => {
                // Non-RUNNING state (STOPPED, EXITED, TERMINATED…) — kill
                // it and clear state.
                delete_tracked_pod(client, &id).await?;
            }
            Err(error) if tracked_pod_is_absent(&error) => {
                let mut state = load_state();
                state.last_pod_id = None;
                state.last_pod_created_at = None;
                state.last_pod_last_used_at = None;
                state.last_pod_gpu = None;
                state.last_pod_cost_per_hr = None;
                save_state(&state)?;
            }
            Err(error) => {
                return Err(error).with_context(|| {
                    format!("check tracked warm pod {id}; refusing to provision a replacement")
                });
            }
        }
    }
    // 2. Create new with scheduling guard + DC fallback.
    let mut create_opts = opts.create.clone();
    if create_opts.model.is_none() {
        create_opts.model = opts.model.clone();
    }
    let mut base_req = build_create_request(&create_opts, client, config).await?;

    // Candidate DC list strategy:
    //   1. If user pinned --dc, use only that.
    //   2. Otherwise, prepend "" (let RunPod pick any schedulable DC).
    //   3. Then append DCs sorted by stock so we have a fallback chain.
    //
    // The "any DC" attempt is critical because:
    //   - RunPod's REST /pods enum only accepts a specific subset of DCs even
    //     though GraphQL exposes more. Pinning a DC that isn't in the REST
    //     enum yields a 400 schema error.
    //   - GraphQL stockStatus is a rough indicator and doesn't always match
    //     what the scheduler can actually place. Letting RunPod pick is the
    //     most reliable first attempt.
    let mut candidates: Vec<String> = Vec::new();
    // Any DC baked into the request by `build_create_request` (e.g. from
    // an attached network volume) is authoritative — we MUST pin to it
    // or the pod create will fail with a volume/DC mismatch.
    let volume_pinned_dc: Option<String> = base_req
        .data_center_ids
        .as_ref()
        .and_then(|v| v.first().cloned());
    if let Some(dc) = authoritative_datacenter(
        create_opts.datacenter.as_deref(),
        config.runpod.default_datacenter.as_deref(),
        volume_pinned_dc.as_deref(),
    ) {
        candidates.push(dc);
    } else {
        // Try "any DC" first.
        candidates.push(String::new());
        // Then add DCs sorted by stock for this GPU, highest → lowest.
        let gpu_display = friendly_gpu_name(&base_req.gpu_type_ids[0]);
        if let Ok(dcs) = client.datacenters().await {
            let mut ranked: Vec<(String, u8)> = dcs
                .into_iter()
                .filter_map(|dc| {
                    let stock = dc
                        .gpu_availability
                        .iter()
                        .find(|g| g.display_name.eq_ignore_ascii_case(&gpu_display))
                        .and_then(|g| g.stock_status.clone())
                        .unwrap_or_default();
                    let rank = match stock.as_str() {
                        "High" => 3,
                        "Medium" => 2,
                        "Low" => 1,
                        _ => 0,
                    };
                    if rank > 0 {
                        Some((dc.id, rank))
                    } else {
                        None
                    }
                })
                .collect();
            ranked.sort_by_key(|entry| std::cmp::Reverse(entry.1));
            for (id, _) in ranked {
                if !candidates.contains(&id) {
                    candidates.push(id);
                }
            }
        }
    }
    print_create_plan(&base_req);
    let mut last_err: Option<anyhow::Error> = None;
    for (idx, dc) in candidates.iter().enumerate() {
        if !dc.is_empty() {
            base_req.data_center_ids = Some(vec![dc.clone()]);
        } else {
            base_req.data_center_ids = None;
        }
        let label = if dc.is_empty() { "any DC" } else { dc };
        println!(
            "{} attempt {}/{} — {}",
            theme::icon_info(),
            idx + 1,
            candidates.len(),
            label.bold()
        );
        let pod = match with_spinner(
            &format!("creating pod in {label}…"),
            client.create_pod(&base_req),
        )
        .await
        {
            Ok(p) => p,
            Err(e) => {
                eprintln!("{} create failed: {e}", theme::icon_warn());
                last_err = Some(e);
                continue;
            }
        };

        // Persist *before* waiting so Ctrl-C during wait doesn't orphan the pod.
        let mut state = load_state();
        state.last_pod_id = Some(pod.id.clone());
        state.last_pod_created_at = Some(now_epoch());
        state.last_pod_last_used_at = Some(now_epoch());
        state.last_pod_gpu = base_req.gpu_type_ids.first().cloned();
        state.last_pod_cost_per_hr = Some(pod.cost_per_hr);
        let _ = save_state(&state);
        let _ = append_history(&HistoryEntry {
            pod_id: pod.id.clone(),
            created_at: now_epoch(),
            deleted_at: None,
            cost_per_hr: pod.cost_per_hr,
            gpu: friendly_gpu_name(&base_req.gpu_type_ids[0]),
            model: opts.model.clone(),
            prompt: Some(opts.prompt.clone()),
        });

        let outcome = tokio::select! {
            biased;
            _ = tokio::signal::ctrl_c() => WaitOutcome::Cancelled,
            res = wait_for_schedule(client, &pod.id, 90) => match res {
                Ok(scheduled) => WaitOutcome::Scheduled(Box::new(scheduled)),
                Err(e) => WaitOutcome::Failed(e),
            },
        };
        match outcome {
            WaitOutcome::Scheduled(scheduled) => {
                println!(
                    "{} scheduled on {} ({})",
                    theme::icon_ok(),
                    scheduled.gpu_name().unwrap_or_default(),
                    scheduled.datacenter_id().unwrap_or_default(),
                );
                return Ok(*scheduled);
            }
            WaitOutcome::Cancelled => {
                eprintln!(
                    "\n{} cancelled — deleting {} to stop billing",
                    theme::icon_warn(),
                    pod.id
                );
                delete_tracked_pod(client, &pod.id).await?;
                bail!("interrupted by user");
            }
            WaitOutcome::Failed(e) => {
                eprintln!(
                    "{} {} didn't schedule — deleting and trying next",
                    theme::icon_warn(),
                    pod.id
                );
                delete_tracked_pod(client, &pod.id).await?;
                last_err = Some(e);
            }
        }
    }
    Err(last_err.unwrap_or_else(|| {
        anyhow::anyhow!("no datacenters with schedulable capacity — try explicit --dc or --gpu")
    }))
}

/// Poll the pod until it's actually scheduled on a machine. Because RunPod
/// REST v1 doesn't reliably populate `runtime` / `machine.gpuDisplayName`
/// (we've seen `status=RUNNING uptime=0` persist on a fully booted pod),
/// we treat any of these as "scheduled":
///   * `uptime_seconds > 0`
///   * `machine` identifies an allocated GPU via `gpuDisplayName` or `gpuTypeId`
///   * the public proxy `https://<id>-7680.proxy.runpod.net/api/status`
///     responds at all (DNS + edge routing only land once scheduled)
async fn wait_for_schedule(client: &RunPodClient, pod_id: &str, timeout_secs: u64) -> Result<Pod> {
    use std::time::{Duration, Instant};
    let pb = indicatif::ProgressBar::new_spinner();
    pb.set_style(
        indicatif::ProgressStyle::with_template(&format!(
            "{{spinner:.{}}} waiting for machine assignment… {{elapsed_precise}} {{msg}}",
            theme::SPINNER_STYLE
        ))
        .unwrap(),
    );
    pb.enable_steady_tick(Duration::from_millis(100));
    let probe_url = format!("https://{pod_id}-7680.proxy.runpod.net/api/status");
    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .unwrap_or_default();
    let start = Instant::now();
    loop {
        // Racing both signals makes us robust to either source going silent.
        let (rest, proxy) = tokio::join!(client.get_pod(pod_id), http.get(&probe_url).send(),);
        let proxy_reached = proxy.is_ok();
        match rest {
            Ok(pod) => {
                let rest_scheduled = pod_is_scheduled(&pod);
                if rest_scheduled || proxy_reached {
                    pb.finish_and_clear();
                    return Ok(pod);
                }
                pb.set_message(format!(
                    "status={} uptime={}s",
                    pod.desired_status, pod.uptime_seconds
                ));
            }
            Err(e) => {
                if proxy_reached {
                    // Proxy can see it — trust that.
                    pb.finish_and_clear();
                    if let Ok(pod) = client.get_pod(pod_id).await {
                        return Ok(pod);
                    }
                }
                pb.set_message(format!("probe error: {e}"));
            }
        }
        if start.elapsed().as_secs() > timeout_secs {
            pb.finish_and_clear();
            bail!(
                "pod {pod_id} did not schedule on a machine within {timeout_secs}s \
                 (RunPod DC likely has no real capacity despite stock signal)"
            );
        }
        tokio::time::sleep(Duration::from_secs(5)).await;
    }
}

async fn wait_for_mold(mold_host: &str, timeout_secs: u64) -> Result<()> {
    use std::time::{Duration, Instant};
    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .unwrap_or_default();
    let status_url = format!("{mold_host}/api/status");
    let start = Instant::now();
    let pb = indicatif::ProgressBar::new_spinner();
    pb.set_style(
        indicatif::ProgressStyle::with_template(&format!(
            "{{spinner:.{}}} waiting for mold server to boot… {{elapsed_precise}} {{msg}}",
            theme::SPINNER_STYLE
        ))
        .unwrap(),
    );
    pb.enable_steady_tick(Duration::from_millis(100));
    loop {
        // The real readiness test is whether `/api/status` returns JSON with
        // a `version` field. Before the mold process is up, the RunPod
        // Cloudflare proxy returns an empty 404 or 502. We need to see the
        // actual mold server response, not the proxy's.
        let progress = match http.get(&status_url).send().await {
            Ok(r) if r.status().is_success() => match r.json::<serde_json::Value>().await {
                Ok(v) if v.get("version").is_some() => {
                    pb.finish_and_clear();
                    println!(
                        "{} mold server {} ready",
                        theme::icon_ok(),
                        v.get("version").and_then(|x| x.as_str()).unwrap_or("?"),
                    );
                    return Ok(());
                }
                Ok(_) => "response missing version field".to_string(),
                Err(_) => "non-JSON response".to_string(),
            },
            Ok(r) => format!("HTTP {}", r.status().as_u16()),
            Err(e) => {
                let s = e.to_string();
                if s.len() > 40 {
                    format!("{}…", &s[..40])
                } else {
                    s
                }
            }
        };
        pb.set_message(format!("({progress})"));
        if start.elapsed().as_secs() > timeout_secs {
            pb.finish_and_clear();
            bail!(
                "pod didn't become reachable after {timeout_secs}s — last probe: {progress} — check {}",
                status_url
            );
        }
        tokio::time::sleep(Duration::from_secs(5)).await;
    }
}

// ── Shell completion candidates ────────────────────────────────────────

pub fn complete_pod_id() -> Vec<CompletionCandidate> {
    // Prefer cached state; fall back to live API if available (best effort).
    let mut out = Vec::new();
    let state = load_state();
    if let Some(id) = state.last_pod_id {
        out.push(CompletionCandidate::new(id));
    }
    // A live API call would be nicer, but completions run inside the
    // #[tokio::main] runtime (CompleteEnv::complete() is called early in
    // main), and block_on from there deadlocks. Running list_pods on a
    // fresh std::thread + mini-runtime works but makes every tab-press
    // hit the network. For now we stick with the persisted warm pod id,
    // which is populated whenever the user `create`s or `run`s a pod.
    // Users can always see live pods with `mold runpod list` and copy-paste.
    out
}

pub fn complete_gpu_id() -> Vec<CompletionCandidate> {
    vec![
        CompletionCandidate::new("4090"),
        CompletionCandidate::new("5090"),
        CompletionCandidate::new("l40s"),
        CompletionCandidate::new("a100"),
        CompletionCandidate::new("h100"),
        CompletionCandidate::new("a6000"),
    ]
}

pub fn complete_dc_id() -> Vec<CompletionCandidate> {
    // Static list of commonly-populated DCs; dynamic would be nicer but
    // requires a blocking API call in completion context.
    [
        "EU-CZ-1", "EU-RO-1", "EUR-IS-1", "EUR-IS-2", "EUR-NO-1", "US-CA-2", "US-IL-1", "US-NC-1",
        "US-TX-3", "CA-MTL-3",
    ]
    .into_iter()
    .map(CompletionCandidate::new)
    .collect()
}

pub fn complete_network_volume_id() -> Vec<CompletionCandidate> {
    Config::load_or_default()
        .runpod
        .default_network_volume_id
        .into_iter()
        .map(CompletionCandidate::new)
        .collect()
}

pub fn complete_cloud_type() -> Vec<CompletionCandidate> {
    vec![
        CompletionCandidate::new("secure"),
        CompletionCandidate::new("community"),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    };
    use wiremock::{
        matchers::{method, path},
        Mock, MockServer, Request, Respond, ResponseTemplate,
    };

    fn transport_request() -> mold_core::GenerateRequest {
        serde_json::from_value(serde_json::json!({
            "prompt": "runpod transport proof",
            "model": "flux-dev:q4",
            "width": 64,
            "height": 64,
            "steps": 1,
            "guidance": 1.0
        }))
        .unwrap()
    }

    async fn mount_canonical_capabilities(server: &MockServer) {
        let mut capabilities = mold_core::ServerCapabilities::default();
        capabilities.queue.heterogeneous_batch_max_outputs = Some(64);
        Mock::given(method("GET"))
            .and(path("/api/capabilities"))
            .respond_with(ResponseTemplate::new(200).set_body_json(capabilities))
            .expect(1)
            .mount(server)
            .await;
    }

    struct HeldCanonicalAdmission;

    impl Respond for HeldCanonicalAdmission {
        fn respond(&self, request: &Request) -> ResponseTemplate {
            let body: serde_json::Value = serde_json::from_slice(&request.body).unwrap();
            ResponseTemplate::new(202).set_body_json(serde_json::json!({
                "id": "batch-1",
                "client_batch_id": body["client_batch_id"],
                "instance_id": "instance-1",
                "durable": true,
                "children": [{
                    "index": 1,
                    "job_id": "durable-runpod-1",
                    "state": "held",
                    "error": "dependency is unavailable",
                    "retryable": true
                }]
            }))
        }
    }

    struct CompleteCanonicalAdmission;

    impl Respond for CompleteCanonicalAdmission {
        fn respond(&self, request: &Request) -> ResponseTemplate {
            let body: serde_json::Value = serde_json::from_slice(&request.body).unwrap();
            ResponseTemplate::new(202).set_body_json(serde_json::json!({
                "id": "batch-1",
                "client_batch_id": body["client_batch_id"],
                "instance_id": "instance-1",
                "durable": true,
                "children": [{
                    "index": 1,
                    "job_id": "durable-runpod-1",
                    "state": "complete",
                    "result": {"filename": "durable.png"}
                }]
            }))
        }
    }

    #[tokio::test]
    async fn runpod_transport_uses_canonical_admission_without_raw_or_sse() {
        let server = MockServer::start().await;
        mount_canonical_capabilities(&server).await;
        Mock::given(method("POST"))
            .and(path("/api/generation-batches"))
            .respond_with(HeldCanonicalAdmission)
            .expect(1)
            .mount(&server)
            .await;
        let Err(error) = generate_with_runpod_transport(
            &mold_core::MoldClient::new(&server.uri()),
            &transport_request(),
        )
        .await
        else {
            panic!("held canonical work is not a successful artifact");
        };

        assert!(error.to_string().contains("durable generation failed"));
        let requests = server.received_requests().await.unwrap();
        assert!(requests
            .iter()
            .any(|request| request.url.path() == "/api/generation-batches"));
        assert!(!requests.iter().any(|request| {
            matches!(request.url.path(), "/api/generate" | "/api/generate/stream")
        }));
    }

    #[tokio::test]
    async fn runpod_canonical_success_hydrates_the_exact_durable_artifact() {
        let server = MockServer::start().await;
        mount_canonical_capabilities(&server).await;
        Mock::given(method("POST"))
            .and(path("/api/generation-batches"))
            .respond_with(CompleteCanonicalAdmission)
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/api/gallery/image/durable.png"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes([1_u8, 2, 3]))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/api/gallery"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(serde_json::json!([{
                    "filename": "durable.png",
                    "timestamp": 1,
                    "metadata": {
                        "prompt": "runpod transport proof",
                        "job_id": "durable-runpod-1",
                        "model": "flux-dev:q4",
                        "seed": 7,
                        "steps": 1,
                        "guidance": 1.0,
                        "width": 64,
                        "height": 64,
                        "version": "test"
                    }
                }])),
            )
            .expect(1)
            .mount(&server)
            .await;
        let artifact = generate_with_runpod_transport(
            &mold_core::MoldClient::new(&server.uri()),
            &transport_request(),
        )
        .await
        .expect("durable admission returns the artifact");
        assert_eq!(artifact.filename, "durable.png");
        assert_eq!(artifact.bytes, [1_u8, 2, 3]);
        assert_eq!(
            artifact.metadata.job_id.as_deref(),
            Some("durable-runpod-1")
        );
        assert_eq!(artifact.request.prompt, "runpod transport proof");
        let requests = server.received_requests().await.unwrap();
        assert!(!requests.iter().any(|request| {
            matches!(request.url.path(), "/api/generate" | "/api/generate/stream")
        }));
    }

    fn policy_create_options(model: Option<&str>) -> CreateOptions {
        CreateOptions {
            name: None,
            gpu: None,
            datacenter: None,
            cloud: CloudType::Secure,
            volume_gb: 50,
            disk_gb: 20,
            image_tag: None,
            model: model.map(str::to_owned),
            hf_token: false,
            network_volume_id: None,
            dry_run: false,
            json: false,
        }
    }

    struct MoldHomeGuard(Option<String>);

    impl Drop for MoldHomeGuard {
        fn drop(&mut self) {
            match self.0.take() {
                Some(value) => std::env::set_var("MOLD_HOME", value),
                None => std::env::remove_var("MOLD_HOME"),
            }
        }
    }

    fn pin_mold_home(path: &std::path::Path) -> MoldHomeGuard {
        let previous = std::env::var("MOLD_HOME").ok();
        std::env::set_var("MOLD_HOME", path);
        MoldHomeGuard(previous)
    }

    #[allow(clippy::await_holding_lock)]
    #[tokio::test]
    async fn public_create_and_run_gate_before_reap_or_local_mutation() {
        let _lock = crate::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let temp = tempfile::tempdir().unwrap();
        let mold_home = temp.path().join("mold-home");
        let output_dir = temp.path().join("outputs");
        let _home = pin_mold_home(&mold_home);

        let error = run_create(policy_create_options(Some("MiniMax-H3-FL2VA")))
            .await
            .expect_err("create must reject before reaping or provider setup");
        assert!(error
            .to_string()
            .contains(mold_core::MINIMAX_H3_AUTHORIZATION_REQUIRED));
        assert!(!mold_home.exists());

        let run = RunOptions {
            prompt: "test".into(),
            model: Some("MiniMaxH3Scheduler".into()),
            output_dir: output_dir.clone(),
            keep: false,
            seed: None,
            steps: None,
            width: None,
            height: None,
            create: policy_create_options(None),
            wait_ready_timeout_secs: 1,
        };
        let error = run_run(run)
            .await
            .expect_err("run must reject before reaping or provider setup");
        assert!(error
            .to_string()
            .contains(mold_core::MINIMAX_H3_AUTHORIZATION_REQUIRED));
        assert!(!mold_home.exists());
        assert!(!output_dir.exists());
    }

    #[tokio::test]
    async fn create_and_run_models_are_rejected_before_provider_admission() {
        let mut config = Config::default();
        let create = policy_create_options(Some("hf:MiniMaxAI/MiniMax-H3"));
        let error = require_create_model_activation(&mut config, &create)
            .await
            .expect_err("RunPod create preload must be compliance-gated");
        assert!(error
            .to_string()
            .contains(mold_core::MINIMAX_H3_AUTHORIZATION_REQUIRED));

        config.default_model = "MiniMaxH3Scheduler".into();
        let run = RunOptions {
            prompt: "test".into(),
            model: None,
            output_dir: std::path::PathBuf::from("/tmp/mold-runpod-policy"),
            keep: false,
            seed: None,
            steps: None,
            width: None,
            height: None,
            create: policy_create_options(None),
            wait_ready_timeout_secs: 1,
        };
        let error = require_run_model_activation(&mut config, &run)
            .await
            .expect_err("RunPod run default must be compliance-gated");
        assert!(error
            .to_string()
            .contains(mold_core::MINIMAX_H3_AUTHORIZATION_REQUIRED));

        config.default_model = "flux2-klein:q8".into();
        let run = RunOptions {
            model: Some("flux2-klein:q8".into()),
            create: policy_create_options(Some("MiniMax-H3-FL2VA")),
            ..run
        };
        let error = require_run_model_activation(&mut config, &run)
            .await
            .expect_err("distinct RunPod preload must also be compliance-gated");
        assert!(error
            .to_string()
            .contains(mold_core::MINIMAX_H3_AUTHORIZATION_REQUIRED));
    }

    #[test]
    fn gpu_id_normalization() {
        assert_eq!(normalize_gpu_id("4090"), "NVIDIA GeForce RTX 4090");
        assert_eq!(normalize_gpu_id("rtx 4090"), "NVIDIA GeForce RTX 4090");
        assert_eq!(normalize_gpu_id("5090"), "NVIDIA GeForce RTX 5090");
        assert_eq!(normalize_gpu_id("a100"), "NVIDIA A100 80GB PCIe");
        assert_eq!(normalize_gpu_id("L40S"), "NVIDIA L40S");
        assert_eq!(
            normalize_gpu_id("NVIDIA GeForce RTX 4090"),
            "NVIDIA GeForce RTX 4090"
        );
    }

    #[test]
    fn published_runpod_gpu_identity_rejects_grace_in_either_field() {
        assert!(ensure_published_runpod_gpu_identity("NVIDIA GB200 A100", "A100 PCIe").is_err());
        assert!(
            ensure_published_runpod_gpu_identity("NVIDIA A100 80GB PCIe", "NVIDIA GB200").is_err()
        );
        assert!(ensure_published_runpod_gpu_identity("NVIDIA B200", "NVIDIA B200").is_ok());
    }

    #[test]
    fn authoritative_provider_id_also_owns_distribution_image_selection() {
        assert_eq!(
            ensure_published_runpod_gpu_identity(
                "NVIDIA A100 80GB PCIe",
                "NVIDIA GeForce RTX 5090"
            )
            .unwrap(),
            "NVIDIA A100 80GB PCIe"
        );
    }

    #[test]
    fn absent_provider_id_retains_legacy_display_name_fallback() {
        let gpu: GpuType = serde_json::from_value(serde_json::json!({
            "displayName": "A100 PCIe"
        }))
        .unwrap();
        assert_eq!(
            authoritative_runpod_gpu_type_id(&gpu),
            "NVIDIA A100 80GB PCIe"
        );
    }

    #[test]
    fn alternate_provider_gpu_id_precedes_display_name_fallback() {
        let gpu: GpuType = serde_json::from_value(serde_json::json!({
            "gpuId": "  NVIDIA A100-SXM4-80GB  ",
            "displayName": "RTX 5090"
        }))
        .unwrap();
        assert_eq!(
            authoritative_runpod_gpu_type_id(&gpu),
            "NVIDIA A100-SXM4-80GB"
        );
    }

    #[tokio::test]
    async fn auto_gpu_resolution_uses_provider_memory_instead_of_display_label_capacity() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": {
                    "gpuTypes": [{
                        "id": "NVIDIA GeForce RTX 5090",
                        "displayName": "A100 PCIe",
                        "memoryInGb": 16,
                        "secureCloud": true,
                        "communityCloud": false
                    }],
                    "dataCenters": [{
                        "gpuAvailability": [{
                            "displayName": "A100 PCIe",
                            "stockStatus": "High"
                        }]
                    }]
                }
            })))
            .mount(&server)
            .await;
        let client = RunPodClient::new(server.uri(), "test-key");
        let opts = CreateOptions {
            name: None,
            gpu: None,
            datacenter: None,
            cloud: CloudType::Secure,
            volume_gb: 50,
            disk_gb: 20,
            image_tag: Some("latest".into()),
            model: None,
            hf_token: false,
            network_volume_id: None,
            dry_run: true,
            json: true,
        };

        let error = build_create_request(&opts, &client, &Config::default())
            .await
            .unwrap_err();
        assert!(
            error.to_string().contains("no GPUs with"),
            "unexpected error: {error:#}"
        );
    }

    #[tokio::test]
    async fn auto_gpu_resolution_preserves_present_zero_provider_memory() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": {
                    "gpuTypes": [{
                        "id": "NVIDIA GeForce RTX 5090",
                        "displayName": "A100 PCIe",
                        "memoryInGb": 0,
                        "secureCloud": true,
                        "communityCloud": false
                    }],
                    "dataCenters": [{
                        "gpuAvailability": [{
                            "displayName": "A100 PCIe",
                            "stockStatus": "High"
                        }]
                    }]
                }
            })))
            .mount(&server)
            .await;
        let client = RunPodClient::new(server.uri(), "test-key");
        let opts = CreateOptions {
            name: None,
            gpu: None,
            datacenter: None,
            cloud: CloudType::Secure,
            volume_gb: 50,
            disk_gb: 20,
            image_tag: Some("latest".into()),
            model: None,
            hf_token: false,
            network_volume_id: None,
            dry_run: true,
            json: true,
        };

        let error = build_create_request(&opts, &client, &Config::default())
            .await
            .unwrap_err();
        assert!(
            error.to_string().contains("no GPUs with"),
            "unexpected error: {error:#}"
        );
    }

    #[tokio::test]
    async fn create_rejects_inventory_gpu_missing_from_live_pod_schema() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": {
                    "gpuTypes": [{
                        "id": "unsupported-provider-id",
                        "displayName": "RTX 5090",
                        "memoryInGb": 32,
                        "secureCloud": true,
                        "communityCloud": false
                    }],
                    "dataCenters": [{
                        "gpuAvailability": [{
                            "displayName": "RTX 5090",
                            "stockStatus": "High"
                        }]
                    }]
                }
            })))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "components": { "schemas": { "PodCreateInput": { "properties": {
                    "gpuTypeIds": { "items": { "enum": ["different-supported-id"] } }
                } } } }
            })))
            .mount(&server)
            .await;
        let client = RunPodClient::new(server.uri(), "test-key");
        let opts = CreateOptions {
            name: None,
            gpu: None,
            datacenter: None,
            cloud: CloudType::Secure,
            volume_gb: 50,
            disk_gb: 20,
            image_tag: Some("latest".into()),
            model: None,
            hf_token: false,
            network_volume_id: None,
            dry_run: true,
            json: true,
        };

        let error = build_create_request(&opts, &client, &Config::default())
            .await
            .unwrap_err();
        assert!(
            error.to_string().contains("not currently accepted"),
            "unexpected error: {error:#}"
        );
    }

    #[tokio::test]
    async fn create_uses_exact_live_schema_identity_after_normalized_inventory_match() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": {
                    "gpuTypes": [{
                        "id": "  nvidia  geforce rtx 5090 ",
                        "displayName": "RTX 5090",
                        "memoryInGb": 32,
                        "secureCloud": true,
                        "communityCloud": false
                    }],
                    "dataCenters": [{
                        "gpuAvailability": [{
                            "displayName": "RTX 5090",
                            "stockStatus": "High"
                        }]
                    }]
                }
            })))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "components": { "schemas": { "PodCreateInput": { "properties": {
                    "gpuTypeIds": { "items": { "enum": ["NVIDIA GeForce RTX 5090"] } }
                } } } }
            })))
            .mount(&server)
            .await;
        let client = RunPodClient::new(server.uri(), "test-key");
        let opts = CreateOptions {
            name: None,
            gpu: None,
            datacenter: None,
            cloud: CloudType::Secure,
            volume_gb: 50,
            disk_gb: 20,
            image_tag: Some("latest".into()),
            model: None,
            hf_token: false,
            network_volume_id: None,
            dry_run: true,
            json: true,
        };

        let request = build_create_request(&opts, &client, &Config::default())
            .await
            .unwrap();
        assert_eq!(
            request.gpu_type_ids,
            ["NVIDIA GeForce RTX 5090".to_string()]
        );
    }

    #[tokio::test]
    async fn explicit_blank_gpu_identity_is_rejected_before_network_access() {
        let server = MockServer::start().await;
        let client = RunPodClient::new(server.uri(), "test-key");
        let opts = CreateOptions {
            name: None,
            gpu: Some(" \t ".into()),
            datacenter: None,
            cloud: CloudType::Secure,
            volume_gb: 50,
            disk_gb: 20,
            image_tag: Some("latest".into()),
            model: None,
            hf_token: false,
            network_volume_id: None,
            dry_run: true,
            json: true,
        };

        let error = build_create_request(&opts, &client, &Config::default())
            .await
            .unwrap_err();
        assert!(
            error.to_string().contains("GPU type cannot be blank"),
            "unexpected error: {error:#}"
        );
        assert!(server.received_requests().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn auto_gpu_resolution_preserves_authoritative_provider_id_for_platform_validation() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": {
                    "gpuTypes": [{
                        "id": "NVIDIA GB200",
                        "displayName": "A100 PCIe",
                        "memoryInGb": 80,
                        "secureCloud": true,
                        "communityCloud": false
                    }],
                    "dataCenters": [{
                        "gpuAvailability": [{
                            "displayName": "A100 PCIe",
                            "stockStatus": "High"
                        }]
                    }]
                }
            })))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "components": { "schemas": { "PodCreateInput": { "properties": {
                    "gpuTypeIds": { "items": { "enum": ["NVIDIA GB200"] } }
                } } } }
            })))
            .mount(&server)
            .await;
        let client = RunPodClient::new(server.uri(), "test-key");
        let opts = CreateOptions {
            name: None,
            gpu: None,
            datacenter: None,
            cloud: CloudType::Secure,
            volume_gb: 50,
            disk_gb: 20,
            image_tag: Some("latest".into()),
            model: None,
            hf_token: false,
            network_volume_id: None,
            dry_run: true,
            json: true,
        };

        let error = build_create_request(&opts, &client, &Config::default())
            .await
            .unwrap_err();

        assert!(
            error.to_string().contains("linux/arm64"),
            "unexpected error: {error:#}"
        );
    }

    #[test]
    fn warm_pod_network_volume_must_match_requested_selection() {
        assert!(!network_volume_selection_mismatch(None, None));
        assert!(!network_volume_selection_mismatch(
            Some("nv-1"),
            Some("nv-1")
        ));
        assert!(network_volume_selection_mismatch(None, Some("nv-1")));
        assert!(network_volume_selection_mismatch(Some("nv-1"), None));
        assert!(network_volume_selection_mismatch(
            Some("nv-1"),
            Some("nv-2")
        ));
    }

    #[test]
    fn matching_network_volume_supersedes_conflicting_warm_datacenter_pin() {
        assert_eq!(
            warm_datacenter_requirement(Some("OTHER-DC"), Some("CONFIG-DC"), Some("nv-1")),
            None
        );
        assert_eq!(
            warm_datacenter_requirement(Some("CLI-DC"), Some("CONFIG-DC"), None),
            Some("CLI-DC".into())
        );
    }

    #[test]
    fn machine_gpu_type_id_is_an_allocation_signal() {
        let pod: Pod = serde_json::from_value(serde_json::json!({
            "id": "pod-live-shape",
            "desiredStatus": "RUNNING",
            "uptimeSeconds": 0,
            "machine": { "gpuTypeId": "NVIDIA GeForce RTX 4090" }
        }))
        .unwrap();

        assert!(pod_is_scheduled(&pod));
    }

    #[test]
    fn pod_detail_shows_top_level_gpu_without_machine() {
        let pod: Pod = serde_json::from_value(serde_json::json!({
            "id": "pod-top-level-gpu",
            "gpu": { "displayName": "NVIDIA H100 NVL" }
        }))
        .unwrap();

        assert_eq!(
            pod_detail_gpu_line(&pod).as_deref(),
            Some("NVIDIA H100 NVL")
        );
    }

    #[test]
    fn only_not_found_clears_a_missing_tracked_pod() {
        let not_found = anyhow::Error::new(MoldError::RunPodNotFound("gone".into()));
        let transient = anyhow::Error::new(MoldError::RunPod("service unavailable".into()));
        let auth = anyhow::Error::new(MoldError::RunPodAuth("expired".into()));

        assert!(tracked_pod_is_absent(&not_found));
        assert!(!tracked_pod_is_absent(&transient));
        assert!(!tracked_pod_is_absent(&auth));
    }

    #[tokio::test]
    async fn failed_delete_does_not_finalize_tracked_state() {
        let server = MockServer::start().await;
        Mock::given(method("DELETE"))
            .respond_with(ResponseTemplate::new(503))
            .mount(&server)
            .await;
        let client = RunPodClient::new(server.uri(), "test-key");
        let finalized = Arc::new(AtomicBool::new(false));
        let finalized_in_callback = Arc::clone(&finalized);

        let error = delete_then_finalize(&client, "still-billing", move || {
            finalized_in_callback.store(true, Ordering::SeqCst);
            Ok(())
        })
        .await
        .unwrap_err();

        assert!(!finalized.load(Ordering::SeqCst));
        assert!(error.to_string().contains("remains tracked"));
        assert!(error.to_string().contains("may still be billing"));
    }

    #[test]
    fn network_volume_datacenter_overrides_cli_and_config_pins() {
        assert_eq!(
            authoritative_datacenter(Some("CLI-DC"), Some("CONFIG-DC"), Some("VOLUME-DC")),
            Some("VOLUME-DC".to_string())
        );
        assert_eq!(
            authoritative_datacenter(Some("CLI-DC"), Some("CONFIG-DC"), None),
            Some("CLI-DC".to_string())
        );
        assert_eq!(
            authoritative_datacenter(None, Some("CONFIG-DC"), None),
            Some("CONFIG-DC".to_string())
        );
    }

    #[test]
    fn network_volume_forces_secure_placement_without_workspace_disk() {
        let volume = NetworkVolume {
            id: "nv-1".into(),
            name: "models".into(),
            data_center_id: "VOLUME-DC".into(),
            size: 100,
        };
        assert_eq!(
            network_volume_placement(Some(&volume), Some("OTHER-DC".into()), "COMMUNITY", 80),
            (Some("VOLUME-DC".into()), "SECURE".into(), 0)
        );
    }

    #[test]
    fn friendly_gpu_roundtrip() {
        assert_eq!(friendly_gpu_name("NVIDIA GeForce RTX 4090"), "RTX 4090");
        assert_eq!(friendly_gpu_name("NVIDIA GeForce RTX 5090"), "RTX 5090");
        assert_eq!(friendly_gpu_name("NVIDIA A100 80GB PCIe"), "A100 PCIe");
        assert_eq!(friendly_gpu_name("NVIDIA L40S"), "L40S");
    }

    #[test]
    fn since_parsing() {
        assert_eq!(since_to_epoch(None).unwrap(), 0);
        assert_eq!(since_to_epoch(Some("")).unwrap(), 0);
        let now = now_epoch();
        let hour_ago = since_to_epoch(Some("1h")).unwrap();
        assert!(hour_ago <= now);
        assert!(hour_ago >= now.saturating_sub(3601));
        assert!(since_to_epoch(Some("7d")).unwrap() <= now.saturating_sub(86400 * 7 - 1));
        assert!(since_to_epoch(Some("2w")).unwrap() <= now.saturating_sub(86400 * 14 - 1));
        assert!(since_to_epoch(Some("bad")).is_err());
    }

    #[test]
    fn stock_coloring_does_not_panic() {
        let _ = color_stock("High");
        let _ = color_stock("Medium");
        let _ = color_stock("Low");
        let _ = color_stock("");
    }

    #[test]
    fn completion_candidates_nonempty() {
        assert!(!complete_gpu_id().is_empty());
        assert!(!complete_dc_id().is_empty());
        assert!(complete_cloud_type().len() == 2);
    }

    #[test]
    fn gpu_vram_lookup_covers_all_known_families() {
        assert_eq!(gpu_vram_gb("RTX 3090"), Some(24));
        assert_eq!(gpu_vram_gb("RTX 4090"), Some(24));
        assert_eq!(gpu_vram_gb("RTX 5090"), Some(32));
        assert_eq!(gpu_vram_gb("L40"), Some(48));
        assert_eq!(gpu_vram_gb("L40S"), Some(48));
        assert_eq!(gpu_vram_gb("RTX A6000"), Some(48));
        assert_eq!(gpu_vram_gb("NVIDIA A100 80GB PCIe"), Some(80));
        assert_eq!(gpu_vram_gb("NVIDIA A100-SXM4-80GB"), Some(80));
        assert_eq!(gpu_vram_gb("NVIDIA H100 80GB HBM3"), Some(80));
        assert_eq!(gpu_vram_gb("NVIDIA H100 NVL"), Some(80));
        // Case-insensitive
        assert_eq!(gpu_vram_gb("rtx 4090"), Some(24));
        // Unknown GPU → None
        assert_eq!(gpu_vram_gb("T4"), None);
    }

    #[test]
    fn estimated_vram_sizes_scale_with_model() {
        // Unknown model: None.
        assert!(estimated_vram_need_gb("nonexistent-model:fp16").is_none());
        // Tiny floor: small models still estimate at least 12GB for
        // encoder + activations + latents.
        if let Some(need) = estimated_vram_need_gb("flux2-klein:q4") {
            assert!(need >= 12, "small FLUX q4 should floor at 12GB, got {need}");
        }
        // LTX-2.3 22B fp8 is ~29GB weights → should land in mid-40s GB
        // range once we multiply by 1.8×. Definitely > 24 (so 4090 fails).
        if let Some(need) = estimated_vram_need_gb("ltx-2.3-22b-dev:fp8") {
            assert!(
                need > 24,
                "LTX-2.3 22B must not fit on a 24GB card, got {need}"
            );
        }
    }

    #[test]
    fn resolve_hf_token_respects_priority() {
        // Clean slate.
        let prev = std::env::var("HF_TOKEN").ok();
        std::env::remove_var("HF_TOKEN");

        // Non-gated model, no flag → no token.
        let opts = CreateOptions {
            name: None,
            gpu: None,
            datacenter: None,
            cloud: CloudType::Secure,
            volume_gb: 50,
            disk_gb: 20,
            image_tag: None,
            model: Some("sd15:fp16".into()),
            hf_token: false,
            network_volume_id: None,
            dry_run: false,
            json: false,
        };
        assert!(resolve_hf_token(&opts).is_none());

        // Explicit --hf-token with no local env → RunPod secret template.
        let mut opts_flag = opts.clone();
        opts_flag.hf_token = true;
        assert_eq!(
            resolve_hf_token(&opts_flag),
            Some("{{ RUNPOD_SECRET_HF_TOKEN }}".into())
        );

        // Local env wins when set, even with the flag on.
        std::env::set_var("HF_TOKEN", "hf_localvalue");
        assert_eq!(resolve_hf_token(&opts_flag), Some("hf_localvalue".into()));

        // Gated model auto-enables token even without the flag.
        let mut gated_opts = opts.clone();
        gated_opts.model = Some("flux-dev:q4".into());
        assert_eq!(resolve_hf_token(&gated_opts), Some("hf_localvalue".into()));

        // Clean up.
        std::env::remove_var("HF_TOKEN");
        if let Some(v) = prev {
            std::env::set_var("HF_TOKEN", v);
        }
    }

    #[test]
    fn wait_outcome_variants_are_exhaustive() {
        // Compile-time sanity: enum must cover Scheduled / Cancelled / Failed.
        let _ = |o: WaitOutcome| match o {
            WaitOutcome::Scheduled(_) => {}
            WaitOutcome::Cancelled => {}
            WaitOutcome::Failed(_) => {}
        };
    }

    #[test]
    fn is_interesting_gpu_matches_the_curated_list() {
        for name in [
            "NVIDIA GeForce RTX 4090",
            "NVIDIA GeForce RTX 5090",
            "NVIDIA GeForce RTX 3090",
            "NVIDIA L40S",
            "NVIDIA L40",
            "NVIDIA A100 80GB PCIe",
            "NVIDIA H100 NVL",
            "NVIDIA RTX A6000",
        ] {
            assert!(is_interesting_gpu(name), "{name} should be interesting");
        }
        for name in ["NVIDIA T4", "NVIDIA RTX 2060", "NVIDIA V100"] {
            assert!(!is_interesting_gpu(name), "{name} should be filtered");
        }
    }

    #[test]
    fn cloud_type_str_roundtrip_is_case_insensitive_and_exhaustive() {
        use std::str::FromStr;
        assert!(matches!(
            CloudType::from_str("secure").unwrap(),
            CloudType::Secure
        ));
        assert!(matches!(
            CloudType::from_str("SECURE").unwrap(),
            CloudType::Secure
        ));
        assert!(matches!(
            CloudType::from_str("Community").unwrap(),
            CloudType::Community
        ));
        assert_eq!(CloudType::Secure.as_str(), "SECURE");
        assert_eq!(CloudType::Community.as_str(), "COMMUNITY");
        assert!(CloudType::from_str("hybrid").is_err());
        assert!(CloudType::from_str("").is_err());
    }

    #[test]
    fn since_to_epoch_supports_every_unit_and_rejects_garbage() {
        let now = now_epoch();
        // Each unit subtracts a known amount; allow 2s of test-runtime slop.
        let approx = |actual: u64, expected_offset: u64| {
            let want = now.saturating_sub(expected_offset);
            actual <= want && actual >= want.saturating_sub(2)
        };
        assert!(approx(since_to_epoch(Some("30s")).unwrap(), 30));
        assert!(approx(since_to_epoch(Some("5m")).unwrap(), 300));
        assert!(approx(since_to_epoch(Some("2h")).unwrap(), 7200));
        assert!(approx(since_to_epoch(Some("3d")).unwrap(), 3 * 86400));
        assert!(approx(since_to_epoch(Some("1w")).unwrap(), 7 * 86400));
        // Unknown unit and unparseable number both error.
        assert!(since_to_epoch(Some("12x")).is_err());
        assert!(since_to_epoch(Some("abch")).is_err());
    }

    #[test]
    fn runpod_state_serde_roundtrip_preserves_optional_fields() {
        let state = RunPodState {
            last_pod_id: Some("abc123".into()),
            last_pod_created_at: Some(1_700_000_000),
            last_pod_last_used_at: Some(1_700_000_500),
            last_pod_gpu: Some("RTX 4090".into()),
            last_pod_cost_per_hr: Some(0.34),
        };
        let json = serde_json::to_string(&state).unwrap();
        let back: RunPodState = serde_json::from_str(&json).unwrap();
        assert_eq!(back.last_pod_id.as_deref(), Some("abc123"));
        assert_eq!(back.last_pod_cost_per_hr, Some(0.34));

        // Empty default round-trips and an empty file decodes to default.
        let default = RunPodState::default();
        let blob = serde_json::to_string(&default).unwrap();
        let back: RunPodState = serde_json::from_str(&blob).unwrap();
        assert!(back.last_pod_id.is_none());
        assert!(back.last_pod_cost_per_hr.is_none());
    }

    #[test]
    fn history_entry_serde_handles_legacy_rows_without_model_or_prompt() {
        // New-format entry with model/prompt populated — round-trips cleanly.
        let entry = HistoryEntry {
            pod_id: "pod-1".into(),
            created_at: 100,
            deleted_at: Some(900),
            cost_per_hr: 1.25,
            gpu: "RTX 4090".into(),
            model: Some("flux-dev:q8".into()),
            prompt: Some("a cat".into()),
        };
        let json = serde_json::to_string(&entry).unwrap();
        let back: HistoryEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(back.pod_id, "pod-1");
        assert_eq!(back.deleted_at, Some(900));
        assert_eq!(back.model.as_deref(), Some("flux-dev:q8"));

        // Legacy row predating model/prompt fields still decodes thanks to
        // serde(default). Pre-T2 logs have these as null/missing.
        let legacy = r#"{"pod_id":"pod-old","created_at":1,"deleted_at":null,"cost_per_hr":0.69,"gpu":"RTX 3090"}"#;
        let back: HistoryEntry = serde_json::from_str(legacy).unwrap();
        assert_eq!(back.pod_id, "pod-old");
        assert!(back.model.is_none());
        assert!(back.prompt.is_none());
    }

    #[test]
    fn friendly_to_gpu_id_falls_back_to_input_for_unknown() {
        // Known mappings are already covered by `friendly_gpu_roundtrip`.
        // Anything we don't recognise must pass through unchanged so users
        // can still pin GPUs we haven't listed yet.
        assert_eq!(
            friendly_to_gpu_id("NVIDIA Mystery GPU"),
            "NVIDIA Mystery GPU"
        );
        assert_eq!(friendly_to_gpu_id(""), "");
    }

    #[test]
    fn normalize_gpu_id_passes_through_unrecognised_aliases() {
        // Unknown shorthands fall through unchanged — better than guessing.
        assert_eq!(normalize_gpu_id("v100"), "v100");
        assert_eq!(normalize_gpu_id("rtx 2060"), "rtx 2060");
        // Anything containing "NVIDIA" is treated as the canonical id.
        assert_eq!(
            normalize_gpu_id("NVIDIA GeForce RTX 6090"),
            "NVIDIA GeForce RTX 6090"
        );
    }

    // Silence unused-import warnings when tests compile but don't use every import.
    #[allow(dead_code)]
    fn _use_imports(_: GpuType) {}
}
