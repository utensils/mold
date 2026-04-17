//! `mold runpod` — native RunPod pod management.
//!
//! Wraps `mold_core::runpod::RunPodClient` so users can manage cloud GPU pods
//! end-to-end from the mold CLI: list/create/stop/delete pods, stream logs,
//! check spend, and (via `run`) generate images on a freshly-provisioned pod
//! with one command.

use anyhow::{bail, Context, Result};
use clap_complete::engine::CompletionCandidate;
use colored::Colorize;
use mold_core::config::Config;
use mold_core::error::MoldError;
use mold_core::runpod::{
    image_tag_for_gpu, CreatePodRequest, GpuType, Pod, RunPodClient, API_KEY_ENV, DEFAULT_ENDPOINT,
};

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
        "{:<28}{:<6}{:<20}{:<10}",
        "GPU".bold(),
        "VRAM".bold(),
        "stock".bold(),
        "secure".bold()
    );
    for g in filtered {
        let stock = g.stock_status.as_deref().unwrap_or("—");
        let stock_colored = color_stock(stock);
        let secure = if g.secure_cloud { "yes" } else { "no" };
        println!(
            "{:<28}{:<6}{:<20}{:<10}",
            g.display_name,
            format!("{}G", g.memory_in_gb),
            stock_colored,
            secure
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

/// `mold runpod list [--json]` — list pods.
pub async fn run_list(json: bool) -> Result<()> {
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
        let gpu = p
            .machine
            .as_ref()
            .and_then(|m| m.gpu_display_name.clone())
            .unwrap_or_else(|| "—".into());
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
    if let Some(machine) = &pod.machine {
        let gpu = machine.gpu_display_name.clone().unwrap_or_default();
        let loc = machine.location.clone().unwrap_or_default();
        println!("  gpu: {gpu} ({loc})");
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

/// `mold runpod usage [--since 7d] [--json]` — show spend summary.
pub async fn run_usage(since: Option<String>, json: bool) -> Result<()> {
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
        let gpu = p
            .machine
            .as_ref()
            .and_then(|m| m.gpu_display_name.clone())
            .unwrap_or_else(|| "—".into());
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

/// Build a `CreatePodRequest` from resolved defaults.
pub async fn build_create_request(
    opts: &CreateOptions,
    client: &RunPodClient,
    config: &Config,
) -> Result<CreatePodRequest> {
    // Resolve GPU — either user-supplied, config default, or cheapest available.
    let (gpu_name, gpu_display) = resolve_gpu(opts, client, config).await?;

    // Resolve datacenter — either user-supplied, config default, or first with stock.
    let dc = resolve_datacenter(opts, client, config, &gpu_display).await?;

    // Resolve image tag.
    let image_tag = opts
        .image_tag
        .clone()
        .unwrap_or_else(|| image_tag_for_gpu(&gpu_display).to_string());
    let image = format!("ghcr.io/utensils/mold:{image_tag}");

    // Resolve volume + network volume.
    let volume_id = opts
        .network_volume_id
        .clone()
        .or_else(|| config.runpod.default_network_volume_id.clone());

    // Build env.
    let mut env = serde_json::Map::new();
    if let Some(model) = opts.model.clone() {
        env.insert("MOLD_DEFAULT_MODEL".into(), model.into());
    }
    env.insert("MOLD_LOG".into(), "info".into());
    if opts.hf_token {
        env.insert("HF_TOKEN".into(), "{{ RUNPOD_SECRET_HF_TOKEN }}".into());
    }

    let name = opts
        .name
        .clone()
        .unwrap_or_else(|| format!("mold-{}", short_timestamp()));
    Ok(CreatePodRequest {
        name,
        image_name: image,
        gpu_type_ids: vec![gpu_name],
        cloud_type: opts.cloud.as_str().to_string(),
        data_center_ids: dc.map(|id| vec![id]),
        gpu_count: 1,
        container_disk_in_gb: opts.disk_gb,
        volume_in_gb: opts.volume_gb,
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
        return Ok((resolved.clone(), friendly_gpu_name(&resolved)));
    }
    if let Some(gpu) = config.runpod.default_gpu.clone() {
        let resolved = normalize_gpu_id(&gpu);
        return Ok((resolved.clone(), friendly_gpu_name(&resolved)));
    }
    // Smart default — query stock and pick the best available.
    let gpus = client.gpu_types().await?;
    // Preference order: 4090 > 5090 > L40S > A100 (cheapest → most headroom)
    for preferred in ["RTX 4090", "RTX 5090", "L40S", "A100 PCIe", "A100 SXM"] {
        if let Some(g) = gpus.iter().find(|g| g.display_name == preferred) {
            let stock = g.stock_status.as_deref().unwrap_or("");
            if matches!(stock, "High" | "Medium") {
                return Ok((friendly_to_gpu_id(&g.display_name), g.display_name.clone()));
            }
        }
    }
    // Fallback: anything with High stock.
    for g in &gpus {
        if g.stock_status.as_deref() == Some("High") && is_interesting_gpu(&g.display_name) {
            return Ok((friendly_to_gpu_id(&g.display_name), g.display_name.clone()));
        }
    }
    bail!("no GPUs with High or Medium stock available — try explicit --gpu")
}

async fn resolve_datacenter(
    opts: &CreateOptions,
    _client: &RunPodClient,
    config: &Config,
    _gpu_display: &str,
) -> Result<Option<String>> {
    // Only honor explicit pins — user-supplied --dc or config default.
    //
    // Why not auto-pick: RunPod's REST /pods endpoint enforces an enum of
    // acceptable `dataCenterIds` that is narrower than what GraphQL's
    // `dataCenters` query exposes. Auto-picking based on GraphQL stockStatus
    // would hit 400 schema errors for many DCs. When no DC is pinned, we
    // leave `dataCenterIds` unset and let RunPod's scheduler choose — it
    // knows which DCs are currently schedulable. Retry logic in
    // `ensure_pod` handles the fallback loop if that fails.
    if let Some(dc) = opts.datacenter.clone() {
        return Ok(Some(dc));
    }
    if let Some(dc) = config.runpod.default_datacenter.clone() {
        return Ok(Some(dc));
    }
    Ok(None)
}

/// Convert RunPod's `displayName` (e.g. `"RTX 5090"`) to the `gpuId` the REST
/// API actually accepts (e.g. `"NVIDIA GeForce RTX 5090"`).
pub fn friendly_to_gpu_id(display: &str) -> String {
    // RunPod `/gputypes` returns these as `displayName`, but the create
    // endpoint expects the full NVIDIA name. The mapping is hardcoded because
    // RunPod's REST API doesn't expose a lookup endpoint.
    match display {
        "RTX 4090" => "NVIDIA GeForce RTX 4090",
        "RTX 5090" => "NVIDIA GeForce RTX 5090",
        "RTX 3090" => "NVIDIA GeForce RTX 3090",
        "L40S" => "NVIDIA L40S",
        "L40" => "NVIDIA L40",
        "A100 PCIe" => "NVIDIA A100 80GB PCIe",
        "A100 SXM" => "NVIDIA A100-SXM4-80GB",
        "H100 SXM" => "NVIDIA H100 80GB HBM3",
        "H100 NVL" => "NVIDIA H100 NVL",
        "RTX A6000" => "NVIDIA RTX A6000",
        other => other,
    }
    .to_string()
}

fn normalize_gpu_id(user: &str) -> String {
    // Allow users to pass the short alias (e.g. "4090", "5090", "a100") or
    // the full NVIDIA name. If we don't recognize it, pass through unchanged.
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

/// `mold runpod create` — create a new pod.
pub async fn run_create(opts: CreateOptions) -> Result<()> {
    let client = match build_client() {
        Ok(c) => c,
        Err(e) => {
            explain_runpod_error(&e);
            return Err(AlreadyReported.into());
        }
    };
    let config = Config::load_or_default();
    let req = build_create_request(&opts, &client, &config).await?;

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

    // Persist warm-pod state + history entry.
    let gpu_display = pod
        .machine
        .as_ref()
        .and_then(|m| m.gpu_display_name.clone())
        .unwrap_or_else(|| friendly_gpu_name(&req.gpu_type_ids[0]));
    let mut state = load_state();
    state.last_pod_id = Some(pod.id.clone());
    state.last_pod_created_at = Some(now_epoch());
    state.last_pod_last_used_at = Some(now_epoch());
    state.last_pod_gpu = Some(gpu_display.clone());
    state.last_pod_cost_per_hr = Some(pod.cost_per_hr);
    let _ = save_state(&state);
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
        println!(
            "  {} {}",
            theme::prefix_hint(),
            format!("export MOLD_HOST=https://{}-7680.proxy.runpod.net", pod.id).bold()
        );
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
pub async fn run_delete(pod_id: String, force: bool, json: bool) -> Result<()> {
    if !force && !confirm(&format!("Delete pod {pod_id}? (billing stops)"))? {
        println!("{} cancelled", theme::icon_neutral());
        return Ok(());
    }
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

fn confirm(msg: &str) -> Result<bool> {
    use std::io::{self, BufRead as _, Write as _};
    // If stdin isn't a TTY, default to no.
    if !std::io::stdin().is_terminal() {
        eprintln!(
            "{} {msg} (non-interactive stdin — refusing)",
            theme::prefix_warning()
        );
        return Ok(false);
    }
    eprint!("{msg} [y/N] ");
    io::stderr().flush().ok();
    let mut line = String::new();
    io::stdin().lock().read_line(&mut line)?;
    Ok(matches!(line.trim().to_lowercase().as_str(), "y" | "yes"))
}

use std::io::IsTerminal as _;

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
    println!("export MOLD_HOST=https://{pod_id}-7680.proxy.runpod.net");
    Ok(())
}

/// `mold runpod logs <pod-id> [--follow]` — print pod logs.
pub async fn run_logs(pod_id: String, follow: bool) -> Result<()> {
    let client = match build_client() {
        Ok(c) => c,
        Err(e) => {
            explain_runpod_error(&e);
            return Err(AlreadyReported.into());
        }
    };
    let initial = match client.pod_logs(&pod_id).await {
        Ok(s) => s,
        Err(e) => {
            explain_runpod_error(&e);
            return Err(AlreadyReported.into());
        }
    };
    print!("{initial}");
    if !follow {
        return Ok(());
    }
    // Crude delta-polling: track the prefix we last printed and emit only the
    // tail. Poll at 2s. Stops on Ctrl-C (global handler) or network error.
    let mut last = initial;
    loop {
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
        let next = match client.pod_logs(&pod_id).await {
            Ok(s) => s,
            Err(_) => continue,
        };
        if next.starts_with(&last) {
            let delta = &next[last.len()..];
            if !delta.is_empty() {
                print!("{delta}");
            }
        } else {
            // log file rotated — reprint
            print!("\n{next}");
        }
        last = next;
    }
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
    let client = match build_client() {
        Ok(c) => c,
        Err(e) => {
            explain_runpod_error(&e);
            return Err(AlreadyReported.into());
        }
    };
    let config = Config::load_or_default();

    // Pick or create a pod.
    let pod = ensure_pod(&client, &config, &opts).await?;
    println!(
        "{} using pod {} on {}",
        theme::icon_ok(),
        pod.id.bold(),
        pod.machine
            .as_ref()
            .and_then(|m| m.gpu_display_name.clone())
            .unwrap_or_default()
    );

    // Wait for proxy readiness.
    let mold_host = format!("https://{}-7680.proxy.runpod.net", pod.id);
    wait_for_mold(&mold_host, opts.wait_ready_timeout_secs).await?;

    // Delegate to the existing run pipeline via MoldClient.
    std::env::set_var("MOLD_HOST", &mold_host);
    std::fs::create_dir_all(&opts.output_dir)
        .with_context(|| format!("create output dir {}", opts.output_dir.display()))?;
    let filename = format!("runpod-{}-{}.png", pod.id, short_timestamp());
    let output_path = opts.output_dir.join(&filename);

    // Build a minimal request and stream.
    let model = opts
        .model
        .clone()
        .or_else(|| Some(config.default_model.clone()))
        .unwrap_or_else(|| "flux2-klein:q8".into());
    let req = mold_core::GenerateRequest {
        prompt: opts.prompt.clone(),
        negative_prompt: None,
        model,
        width: opts.width.unwrap_or(config.default_width),
        height: opts.height.unwrap_or(config.default_height),
        steps: opts.steps.unwrap_or(config.default_steps),
        guidance: 0.0,
        seed: opts.seed,
        batch_size: 1,
        output_format: mold_core::OutputFormat::Png,
        embed_metadata: None,
        scheduler: None,
        source_image: None,
        edit_images: None,
        strength: 0.75,
        mask_image: None,
        control_image: None,
        control_model: None,
        control_scale: 1.0,
        expand: None,
        original_prompt: None,
        lora: None,
        frames: None,
        fps: None,
        upscale_model: None,
        gif_preview: false,
        enable_audio: None,
        audio_file: None,
        source_video: None,
        keyframes: None,
        pipeline: None,
        loras: None,
        retake_range: None,
        spatial_upscale: None,
        temporal_upscale: None,
    };
    let http = mold_core::MoldClient::new(&mold_host);

    println!(
        "{} generating → {}",
        theme::icon_info(),
        output_path.display().to_string().cyan()
    );

    // Stream progress via SSE so RunPod's 100s Cloudflare proxy timeout
    // doesn't kill the model-pull phase.
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<mold_core::types::SseProgressEvent>();
    let progress_task = tokio::spawn(async move {
        let pb = indicatif::ProgressBar::new_spinner();
        pb.set_style(
            indicatif::ProgressStyle::with_template(&format!(
                "{{spinner:.{}}} {{msg}}",
                theme::SPINNER_STYLE
            ))
            .unwrap(),
        );
        pb.enable_steady_tick(std::time::Duration::from_millis(80));
        while let Some(ev) = rx.recv().await {
            let line = format_progress_event(&ev);
            pb.set_message(line);
        }
        pb.finish_and_clear();
    });

    let maybe_resp = http
        .generate_stream(&req, tx)
        .await
        .with_context(|| "generation failed")?;
    let _ = progress_task.await;

    let resp = match maybe_resp {
        Some(r) => r,
        None => http
            .generate(req)
            .await
            .with_context(|| "generation failed (non-stream fallback)")?,
    };

    if let Some(video) = resp.video {
        let vid_path = opts.output_dir.join(format!(
            "runpod-{}-{}.{}",
            pod.id,
            short_timestamp(),
            extension_for_video(video.format)
        ));
        std::fs::write(&vid_path, &video.data)?;
        println!("{} saved {}", theme::icon_done(), vid_path.display());
    } else if let Some(img) = resp.images.first() {
        std::fs::write(&output_path, &img.data)?;
        println!(
            "{} saved {} ({:.1}s on seed {})",
            theme::icon_done(),
            output_path.display(),
            resp.generation_time_ms as f64 / 1000.0,
            resp.seed_used,
        );
    }

    // Update state + history.
    let mut state = load_state();
    state.last_pod_id = Some(pod.id.clone());
    state.last_pod_last_used_at = Some(now_epoch());
    let _ = save_state(&state);

    if !opts.keep && config.runpod.auto_teardown {
        println!(
            "{} auto-teardown enabled — deleting pod",
            theme::icon_info()
        );
        let _ = client.delete_pod(&pod.id).await;
        mark_history_deleted(&pod.id);
        let mut state = load_state();
        state.last_pod_id = None;
        let _ = save_state(&state);
    } else if opts.keep {
        println!(
            "{} pod {} kept running (--keep). Delete with {}",
            theme::icon_info(),
            pod.id,
            format!("mold runpod delete {}", pod.id).bold()
        );
    } else {
        println!(
            "{} pod {} left warm for reuse. Idle reap in {} min.",
            theme::icon_info(),
            pod.id,
            config.runpod.auto_teardown_idle_mins
        );
    }
    Ok(())
}

fn format_progress_event(ev: &mold_core::types::SseProgressEvent) -> String {
    use mold_core::types::SseProgressEvent as E;
    match ev {
        E::StageStart { name } => format!("stage: {name}"),
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
        } => format!("✓ [{file_index}/{total_files}] {filename}"),
        E::PullComplete { model } => format!("pull complete: {model}"),
        E::Queued { position } => format!("queued (#{position})"),
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
    const K: f64 = 1024.0;
    let f = b as f64;
    if f < K {
        format!("{b}B")
    } else if f < K * K {
        format!("{:.1}K", f / K)
    } else if f < K * K * K {
        format!("{:.1}M", f / (K * K))
    } else {
        format!("{:.1}G", f / (K * K * K))
    }
}

fn extension_for_video(fmt: mold_core::OutputFormat) -> &'static str {
    match fmt {
        mold_core::OutputFormat::Mp4 => "mp4",
        mold_core::OutputFormat::Gif => "gif",
        mold_core::OutputFormat::Apng => "apng",
        mold_core::OutputFormat::Webp => "webp",
        _ => "bin",
    }
}

async fn ensure_pod(client: &RunPodClient, config: &Config, opts: &RunOptions) -> Result<Pod> {
    // 1. Warm pod? Only reuse if actually scheduled on a machine.
    let state = load_state();
    if let Some(id) = state.last_pod_id.clone() {
        if let Ok(pod) = client.get_pod(&id).await {
            let scheduled = pod.runtime.is_some()
                && pod
                    .machine
                    .as_ref()
                    .and_then(|m| m.gpu_display_name.as_deref())
                    .is_some_and(|s| !s.is_empty());
            if pod.desired_status == "RUNNING" && scheduled {
                return Ok(pod);
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
    let user_pinned = create_opts.datacenter.is_some();
    if let Some(dc) = &create_opts.datacenter {
        candidates.push(dc.clone());
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
    let _ = user_pinned;

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

        match wait_for_schedule(client, &pod.id, 90).await {
            Ok(scheduled) => {
                println!(
                    "{} scheduled on {} ({})",
                    theme::icon_ok(),
                    scheduled
                        .machine
                        .as_ref()
                        .and_then(|m| m.gpu_display_name.clone())
                        .unwrap_or_default(),
                    scheduled
                        .machine
                        .as_ref()
                        .and_then(|m| m.location.clone())
                        .unwrap_or_default(),
                );
                return Ok(scheduled);
            }
            Err(e) => {
                eprintln!(
                    "{} {} didn't schedule — deleting and trying next",
                    theme::icon_warn(),
                    pod.id
                );
                let _ = client.delete_pod(&pod.id).await;
                mark_history_deleted(&pod.id);
                let mut state = load_state();
                state.last_pod_id = None;
                let _ = save_state(&state);
                last_err = Some(e);
            }
        }
    }
    Err(last_err.unwrap_or_else(|| {
        anyhow::anyhow!("no datacenters with schedulable capacity — try explicit --dc or --gpu")
    }))
}

/// Poll the pod until it has been scheduled on a real machine.
/// "Scheduled" means `runtime` is non-null AND `machine.gpuDisplayName` is set.
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
    let start = Instant::now();
    loop {
        match client.get_pod(pod_id).await {
            Ok(pod) => {
                let scheduled = pod.runtime.is_some()
                    && pod
                        .machine
                        .as_ref()
                        .and_then(|m| m.gpu_display_name.as_deref())
                        .is_some_and(|s| !s.is_empty());
                if scheduled {
                    pb.finish_and_clear();
                    return Ok(pod);
                }
                pb.set_message(format!(
                    "status={} uptime={}s",
                    pod.desired_status, pod.uptime_seconds
                ));
            }
            Err(e) => {
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
    // Live query (synchronous via tokio::runtime::Handle::block_on is
    // problematic in async contexts — skip if we can't)
    if let Ok(client) = build_client() {
        if let Ok(rt) = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
        {
            if let Ok(pods) = rt.block_on(client.list_pods()) {
                for p in pods {
                    if !out.iter().any(|c| c.get_value().to_string_lossy() == p.id) {
                        let display = format!("{} ({})", p.id, p.name.clone().unwrap_or_default());
                        out.push(CompletionCandidate::new(p.id).help(Some(display.into())));
                    }
                }
            }
        }
    }
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

pub fn complete_cloud_type() -> Vec<CompletionCandidate> {
    vec![
        CompletionCandidate::new("secure"),
        CompletionCandidate::new("community"),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

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

    // Silence unused-import warnings when tests compile but don't use every import.
    #[allow(dead_code)]
    fn _use_imports(_: GpuType) {}
}
