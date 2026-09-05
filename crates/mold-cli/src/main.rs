mod catalog_bridge;
mod commands;
mod control;
mod errors;
mod fs_util;
mod metadata_db;
mod output;
mod procinfo;
mod skill;
#[cfg(test)]
mod test_support;
mod theme;
mod ui;

use clap::{builder::ValueHint, CommandFactory, Parser, Subcommand};
use clap_complete::engine::{ArgValueCandidates, CompletionCandidate};
use mold_core::{OutputFormat, Scheduler};

/// Value parser for OutputFormat with tab-completion candidates.
fn output_format_parser(formats: &'static [&'static str]) -> clap::builder::ValueParser {
    let parser = clap::builder::TypedValueParser::map(
        clap::builder::PossibleValuesParser::new(formats),
        |s: String| s.parse::<OutputFormat>().unwrap(),
    );
    clap::builder::ValueParser::new(parser)
}

/// Value parser for `mold library export --format`. Delegates to the wire
/// type's own `FromStr` so the CLI can never accept a container the export
/// endpoint would refuse.
fn mesh_export_format_parser(raw: &str) -> Result<mold_core::MeshExportFormat, String> {
    raw.parse()
}

/// Value parser for `--title`: applies the shared print-title contract
/// (trim, no control characters, at most 120 characters) at parse time so a
/// bad title is refused before any server or GPU work. An empty or
/// whitespace-only title is a parse error too — `--title ""` is a mistake, not
/// an untitled print.
fn print_title_parser(raw: &str) -> Result<String, String> {
    match mold_core::validate_print_title(raw) {
        Ok(Some(title)) => Ok(title),
        Ok(None) => Err("title must not be empty".to_string()),
        Err(error) => Err(error),
    }
}

/// Value parser for `--tag`: applies the shared tag contract (trim, collapse
/// interior whitespace, no control characters, at most 64 characters) at
/// parse time, so a bad tag is refused before any server or GPU work. An
/// empty tag is a parse error rather than a silent no-op — a user typing
/// `--tag ""` meant something.
fn tag_parser(raw: &str) -> Result<String, String> {
    match mold_core::normalize_tag_name(raw) {
        Ok(Some(tag)) => Ok(tag),
        Ok(None) => Err("tag must not be empty".to_string()),
        Err(error) => Err(error),
    }
}

/// Value parser for `--collection`: the shared collection-name contract.
/// Returns the normalized display name; the slug it merges on is derived
/// server-side (and locally) from exactly this value.
fn collection_name_parser(raw: &str) -> Result<String, String> {
    mold_core::validate_collection_name(raw).map(|(name, _slug)| name)
}

fn library_limit_parser(raw: &str) -> Result<usize, String> {
    let value = raw
        .parse::<usize>()
        .map_err(|_| "limit must be an integer from 1 to 1000".to_string())?;
    if !(1..=1000).contains(&value) {
        return Err("limit must be from 1 to 1000".to_string());
    }
    Ok(value)
}

#[derive(Clone, clap::ValueEnum)]
enum LogFormat {
    Text,
    Json,
}

#[derive(Clone, clap::ValueEnum)]
pub(crate) enum Ltx2SpatialUpscaleArg {
    #[value(name = "x1.5")]
    X1_5,
    #[value(name = "x2")]
    X2,
}

#[derive(Clone, clap::ValueEnum)]
pub(crate) enum Ltx2TemporalUpscaleArg {
    #[value(name = "x2")]
    X2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum Ltx2PipelineArg {
    #[value(name = "one-stage")]
    OneStage,
    #[value(name = "two-stage")]
    TwoStage,
    #[value(name = "two-stage-hq")]
    TwoStageHq,
    #[value(name = "distilled")]
    Distilled,
    #[value(name = "ic-lora")]
    IcLora,
    #[value(name = "keyframe")]
    Keyframe,
    // Canonically `a2-vid`, matching the wire enum every other surface uses.
    // The original `a2vid` spelling stays accepted so existing scripts keep
    // working.
    #[value(name = "a2-vid", alias = "a2vid")]
    A2Vid,
    #[value(name = "retake")]
    Retake,
    #[value(name = "lip-dub", alias = "lipdub")]
    LipDub,
    /// Text-to-audio: audio-only generation, no video.
    #[value(name = "t2a")]
    T2a,
}

/// Sentinel error: the command already printed diagnostics to stderr.
/// The main handler should just exit(1) without printing anything extra.
#[derive(Debug)]
struct AlreadyReported;

impl std::fmt::Display for AlreadyReported {
    fn fmt(&self, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}

impl std::error::Error for AlreadyReported {}

#[derive(Parser)]
#[command(
    name = "mold",
    about = "Local AI image generation — FLUX, SD1.5, SDXL & Z-Image diffusion models on your GPU",
    after_long_help = "\
Quick start:
  mold pull flux-schnell:q8        Download a model
  mold run \"a cat on a skateboard\"  Generate an image

Run 'mold <command> --help' for more information on a command.

Report bugs: https://github.com/utensils/mold/issues"
)]
#[command(version = mold_core::build_info::FULL_VERSION, propagate_version = true)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

/// `mold run`'s 3-D controls.
///
/// A flattened `Args` struct rather than five more fields on
/// `Commands::Run`. That variant is already enormous — around ninety fields —
/// and clap's generated `from_arg_matches` constructs it through several
/// frames by value; in a debug build, five more inline `Option`s pushed the
/// first CLI parse test past the 2 MiB test-thread stack and aborted the
/// process. Flattening keeps the group in its own frame, and it reads better
/// besides: these five only ever make sense together.
#[derive(clap::Args, Debug, Clone, Default)]
struct MeshArgs {
    /// Resolution of the query grid a 3-D model's occupancy field is
    /// evaluated on. Higher captures finer detail; cost is CUBIC, so 384 is
    /// roughly eight times 192. Defaults to 256.
    #[arg(long, value_name = "N", help_heading = "3D")]
    octree: Option<u32>,

    /// Iso-level at which the surface is extracted (0.0-1.0).
    /// Defaults to 0.6.
    #[arg(long, value_name = "T", help_heading = "3D")]
    mesh_threshold: Option<f32>,

    /// Decimate the mesh to approximately this many triangles.
    /// Omitted keeps the raw surface-net output.
    #[arg(long, value_name = "N", help_heading = "3D")]
    target_faces: Option<u32>,

    /// Generate PBR textures as well as geometry. Requires the paint bundle;
    /// without it the request is refused rather than answered with a bare
    /// white mesh.
    #[arg(long, help_heading = "3D")]
    texture: bool,

    /// Edge length of the generated texture atlas (1024, 2048 or 4096).
    /// Requires --texture.
    #[arg(long, value_name = "N", requires = "texture", help_heading = "3D")]
    texture_resolution: Option<u32>,
}

impl MeshArgs {
    fn into_flags(self) -> commands::run::MeshFlags {
        commands::run::MeshFlags {
            octree: self.octree,
            threshold: self.mesh_threshold,
            target_faces: self.target_faces,
            texture: self.texture,
            texture_resolution: self.texture_resolution,
        }
    }
}

/// Subcommands of `mold licenses`.
#[derive(Subcommand)]
enum LicensesAction {
    /// Record acceptance of pinned third-party terms WITHOUT downloading
    ///
    /// Consent and acquisition are different acts: this agrees to the terms so
    /// a later pull is unblocked, rather than transferring gigabytes now.
    /// Acceptance is recorded on the machine that would run the pull — the
    /// server at MOLD_HOST when one answers, otherwise this machine.
    Accept {
        /// License id(s), as printed by `mold licenses`
        #[arg(required = true, num_args = 1..)]
        ids: Vec<String>,

        /// Record in THIS machine's Mold data root instead of the server's
        #[arg(long)]
        local: bool,
    },
}

#[derive(Subcommand)]
enum ConfigAction {
    /// List all configuration values
    List {
        /// Output as JSON object
        #[arg(long)]
        json: bool,
    },
    /// Get a configuration value by key
    Get {
        /// Config key (e.g. server_port, expand.backend, models.flux-dev:q4.default_steps)
        #[arg(add = ArgValueCandidates::new(commands::config::complete_config_key))]
        key: String,
        /// Output raw value only (no decoration), for scripting
        #[arg(long)]
        raw: bool,
    },
    /// Set a configuration value
    Set {
        /// Config key (e.g. server_port, expand.backend)
        #[arg(add = ArgValueCandidates::new(commands::config::complete_config_key))]
        key: String,
        /// Value to set (use "none" to clear optional fields)
        value: String,
    },
    /// Show the config file path
    Path,
    /// Open config file in $EDITOR
    Edit,
    /// Show which surface (file, db, env) owns a given key
    Where {
        /// Config key to inspect
        #[arg(add = ArgValueCandidates::new(commands::config::complete_config_key))]
        key: String,
    },
    /// Drop a DB-backed key so the next read falls back to config.toml / env / default.
    ///
    /// Rejects TOML-only keys with a helpful error — those are edited by
    /// `mold config set` or by editing `config.toml` directly.
    Reset {
        /// Config key to reset. Use `--all` instead of a key to drop every
        /// DB row.
        #[arg(
            required_unless_present = "all",
            add = ArgValueCandidates::new(commands::config::complete_config_key)
        )]
        key: Option<String>,
        /// Drop every DB row for the active profile.
        #[arg(long, conflicts_with = "key")]
        all: bool,
        /// Skip the confirmation prompt when using `--all`.
        #[arg(long)]
        yes: bool,
    },
}

#[derive(Subcommand)]
enum GpuAction {
    /// List all runtime-visible devices.
    List {
        /// Output the stable-ID device inventory as JSON.
        #[arg(long)]
        json: bool,
    },
    /// Stop assigning new work to a device; active work drains first.
    Disable {
        /// Opaque stable ID (preferred) or current display ordinal.
        #[arg(add = ArgValueCandidates::new(commands::gpu::complete_device_id))]
        device: String,
    },
    /// Return a disabled or draining device to service.
    Enable {
        /// Opaque stable ID (preferred) or current display ordinal.
        #[arg(add = ArgValueCandidates::new(commands::gpu::complete_device_id))]
        device: String,
    },
}

#[derive(Subcommand)]
enum RunpodAction {
    /// Check RunPod auth, endpoint, and account info
    Doctor,
    /// List available GPU types
    Gpus {
        /// Show all GPUs (not just commonly-used ones)
        #[arg(long)]
        all: bool,
        /// Output as JSON
        #[arg(long)]
        json: bool,
    },
    /// List datacenters; `--gpu <name>` filters by GPU stock
    Datacenters {
        /// Filter by GPU display name (e.g. "RTX 4090")
        #[arg(long, add = ArgValueCandidates::new(commands::runpod::complete_gpu_id))]
        gpu: Option<String>,
        /// Output as JSON
        #[arg(long)]
        json: bool,
    },
    /// Manage persistent RunPod network volumes
    #[command(alias = "volumes", alias = "volume")]
    NetworkVolume {
        #[command(subcommand)]
        action: RunpodNetworkVolumeAction,
    },
    /// List pods in your account
    List {
        #[arg(long)]
        json: bool,
    },
    /// Show details for a single pod
    Get {
        /// Pod id
        #[arg(add = ArgValueCandidates::new(commands::runpod::complete_pod_id))]
        pod_id: String,
        #[arg(long)]
        json: bool,
    },
    /// Create a new pod (with smart defaults if fields are omitted)
    #[command(after_long_help = "\
Examples:
  mold runpod create                         # smart defaults, first available stock
  mold runpod create --gpu 5090              # specific GPU
  mold runpod create --gpu 4090 --dc US-IL-1 # pin to a datacenter
  mold runpod create --model flux-dev:q4     # preload a model via MOLD_DEFAULT_MODEL
  mold runpod create --dry-run               # print plan without creating")]
    Create {
        /// Pod name (auto-generated if omitted)
        #[arg(long)]
        name: Option<String>,
        /// GPU (e.g. 4090, 5090, a100, l40s, h100, or full NVIDIA name)
        #[arg(long, add = ArgValueCandidates::new(commands::runpod::complete_gpu_id))]
        gpu: Option<String>,
        /// Datacenter id (e.g. EUR-IS-2, US-IL-1)
        #[arg(long = "dc", add = ArgValueCandidates::new(commands::runpod::complete_dc_id))]
        datacenter: Option<String>,
        /// Attach this persistent network volume (forces Secure Cloud and its datacenter)
        #[arg(long, add = ArgValueCandidates::new(commands::runpod::complete_network_volume_id))]
        network_volume: Option<String>,
        /// Cloud tier: secure or community
        #[arg(long, default_value = "secure", add = ArgValueCandidates::new(commands::runpod::complete_cloud_type))]
        cloud: String,
        /// Container disk size in GB
        #[arg(long, default_value_t = 20)]
        disk: u32,
        /// Volume size in GB (mounted at /workspace)
        #[arg(long, default_value_t = 50)]
        volume: u32,
        /// Override the image tag (e.g. latest, latest-sm120)
        #[arg(long)]
        image_tag: Option<String>,
        /// Preload this model via MOLD_DEFAULT_MODEL
        #[arg(long)]
        model: Option<String>,
        /// Wire HF_TOKEN={{ RUNPOD_SECRET_HF_TOKEN }} into the pod env
        #[arg(long)]
        hf_token: bool,
        /// Print the request plan and exit without creating
        #[arg(long)]
        dry_run: bool,
        /// Output created pod as JSON
        #[arg(long)]
        json: bool,
    },
    /// Stop a pod (billing paused, storage retained)
    Stop {
        #[arg(add = ArgValueCandidates::new(commands::runpod::complete_pod_id))]
        pod_id: String,
        #[arg(long)]
        json: bool,
    },
    /// Start a stopped pod
    Start {
        #[arg(add = ArgValueCandidates::new(commands::runpod::complete_pod_id))]
        pod_id: String,
        #[arg(long)]
        json: bool,
    },
    /// Delete a pod (irreversible, no confirmation)
    Delete {
        #[arg(add = ArgValueCandidates::new(commands::runpod::complete_pod_id))]
        pod_id: String,
        /// No-op retained for backward compatibility — delete is always non-interactive.
        #[arg(long, short = 'f', hide = true)]
        force: bool,
        #[arg(long)]
        json: bool,
    },
    /// Print `export MOLD_HOST=…` for the given pod (shell-evalable)
    #[command(after_long_help = "\
Example:
  eval \"$(mold runpod connect <pod-id>)\"")]
    Connect {
        #[arg(add = ArgValueCandidates::new(commands::runpod::complete_pod_id))]
        pod_id: String,
        /// Also verify the pod is reachable before printing
        #[arg(long)]
        check: bool,
    },
    /// Validate a pod and print the RunPod console logs handoff
    Logs {
        #[arg(add = ArgValueCandidates::new(commands::runpod::complete_pod_id))]
        pod_id: String,
        /// Deprecated: RunPod exposes live logs only in its web console
        #[arg(long, short = 'f')]
        follow: bool,
    },
    /// Show spend summary: balance, active pods, historical spend
    Usage {
        /// Historical window (e.g. 7d, 24h, 2w)
        #[arg(long)]
        since: Option<String>,
        #[arg(long)]
        json: bool,
    },
    /// Generate on a fresh or warm RunPod pod, save to ./mold-outputs/
    #[command(after_long_help = "\
Examples:
  mold runpod run \"a cat on a skateboard\"
  mold runpod run \"a sunset\" --model flux2-klein:q8
  mold runpod run \"a robot\" --keep            # leave pod running after
  mold runpod run \"a sunset\" --gpu 5090       # force a GPU choice")]
    Run {
        /// Text prompt
        prompt: String,
        /// Target model (e.g. flux2-klein:q8); defaults to config.default_model
        #[arg(short, long, add = ArgValueCandidates::new(commands::run::complete_model_name))]
        model: Option<String>,
        /// Output directory (default ./mold-outputs)
        #[arg(short, long, default_value = "./mold-outputs", value_hint = ValueHint::DirPath)]
        output_dir: std::path::PathBuf,
        /// Keep the pod running after generation (otherwise left warm)
        #[arg(long)]
        keep: bool,
        /// Seed
        #[arg(long)]
        seed: Option<u64>,
        /// Steps
        #[arg(long)]
        steps: Option<u32>,
        /// Image width
        #[arg(long)]
        width: Option<u32>,
        /// Image height
        #[arg(long)]
        height: Option<u32>,
        /// GPU override (4090, 5090, a100, …)
        #[arg(long, add = ArgValueCandidates::new(commands::runpod::complete_gpu_id))]
        gpu: Option<String>,
        /// Datacenter override
        #[arg(long = "dc", add = ArgValueCandidates::new(commands::runpod::complete_dc_id))]
        datacenter: Option<String>,
        /// Attach this persistent network volume (forces Secure Cloud and its datacenter)
        #[arg(long, add = ArgValueCandidates::new(commands::runpod::complete_network_volume_id))]
        network_volume: Option<String>,
        /// Pod-ready timeout in seconds
        #[arg(long, default_value_t = 600)]
        wait_timeout: u64,
        /// Force HF_TOKEN passthrough even for non-gated models (auto-enabled
        /// for gated models). Uses local `HF_TOKEN` env if set, else the
        /// RunPod secret `HF_TOKEN`.
        #[arg(long)]
        hf_token: bool,
    },
}

#[derive(Subcommand)]
enum RunpodNetworkVolumeAction {
    /// List network volumes
    List {
        #[arg(long)]
        json: bool,
    },
    /// Show one network volume
    Get {
        #[arg(add = ArgValueCandidates::new(commands::runpod::complete_network_volume_id))]
        volume_id: String,
        #[arg(long)]
        json: bool,
    },
    /// Create persistent storage in a Secure Cloud datacenter
    Create {
        #[arg(long)]
        name: String,
        /// Size in GB (10-3999; live RunPod production bound)
        #[arg(long)]
        size: u32,
        /// Datacenter id where the volume will live
        #[arg(long = "dc", add = ArgValueCandidates::new(commands::runpod::complete_dc_id))]
        datacenter: String,
        #[arg(long)]
        json: bool,
    },
    /// Rename or grow a network volume (sizes cannot shrink)
    Update {
        #[arg(add = ArgValueCandidates::new(commands::runpod::complete_network_volume_id))]
        volume_id: String,
        #[arg(long)]
        name: Option<String>,
        /// New total size in GB (must be larger than current size)
        #[arg(long)]
        size: Option<u32>,
        #[arg(long)]
        json: bool,
    },
    /// Permanently delete a network volume and all of its data
    Delete {
        #[arg(add = ArgValueCandidates::new(commands::runpod::complete_network_volume_id))]
        volume_id: String,
        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand)]
enum LambdaAction {
    /// Check Lambda auth, endpoint, and account basics
    Doctor,
    /// List live Lambda GPU capacity and selected mold image tags
    Availability {
        #[arg(long)]
        json: bool,
    },
    /// Deploy or repair a private tunneled mold server on Lambda Cloud
    Deploy {
        /// Lambda instance type name, e.g. gpu_1x_a10
        #[arg(long)]
        instance_type: Option<String>,
        /// Lambda region name, e.g. us-west-1
        #[arg(long)]
        region: Option<String>,
        /// Force a new instance instead of reusing state
        #[arg(long)]
        new: bool,
        /// Print launch request and exit
        #[arg(long)]
        dry_run: bool,
        /// Emit machine-readable phase events
        #[arg(long)]
        json: bool,
        /// Copy local HF_TOKEN/CIVITAI_TOKEN to the remote service env
        #[arg(long)]
        forward_secrets: bool,
        /// Enqueue this model download after the tunnel is ready
        #[arg(long, add = ArgValueCandidates::new(commands::run::complete_model_name))]
        model: Option<String>,
        /// Open the local tunneled web UI in a browser
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        open_browser: bool,
    },
    /// Show the state-tracked Lambda instance
    Status {
        #[arg(long)]
        json: bool,
    },
    /// Print remote mold service logs over SSH
    Logs {
        #[arg(long, short = 'f')]
        follow: bool,
    },
    /// Start or refresh the local SSH tunnel for the state-tracked instance
    Tunnel {
        #[arg(long)]
        local_port: Option<u16>,
    },
    /// SSH into the state-tracked instance
    Ssh,
    /// List Lambda filesystems and usage
    Filesystems {
        #[arg(long)]
        json: bool,
    },
    /// Terminate the state-tracked instance and kill its local tunnel
    Terminate {
        #[arg(long)]
        json: bool,
    },
    /// Delete mold-managed Lambda resources after typed confirmation
    Reset {
        #[arg(long = "to-zero")]
        to_zero: bool,
        #[arg(long)]
        confirm: Option<String>,
        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand)]
enum ServerAction {
    /// Start the server as a background daemon
    #[command(after_long_help = "\
Examples:
  mold server start                Start on default port 7680
  mold server start --port 8080    Custom port")]
    Start {
        /// Server port
        #[arg(long, env = "MOLD_PORT", default_value_t = 7680)]
        port: u16,
        /// Bind address
        #[arg(long, default_value = "0.0.0.0")]
        bind: String,
        /// Override the models directory for this process
        #[arg(long, env = "MOLD_MODELS_DIR")]
        models_dir: Option<String>,
        /// Enable rotated file logging to ~/.mold/logs/
        #[arg(long, default_value_t = true)]
        log_file: bool,
        /// Disable mDNS/DNS-SD advertising and server-assisted browsing
        #[cfg(feature = "mdns")]
        #[arg(long)]
        no_mdns: bool,
    },
    /// Show status of the managed server, or of `--host` when one is named
    #[command(after_long_help = "\
Examples:
  mold server status                       This machine's managed daemon
  mold server status --host plato          A remote server (PID/logs are local-only)
  MOLD_HOST=plato mold server status       Same, from the environment")]
    Status {
        /// Report on this server instead of the local managed daemon
        #[arg(long, env = "MOLD_HOST", help_heading = "Server")]
        host: Option<String>,
    },
    /// Discover mold servers advertised on the local network via mDNS
    #[cfg(feature = "mdns")]
    #[command(after_long_help = "\
Examples:
  mold server discover                 Browse for ~3s and print a table
  mold server discover --json          Machine-readable output
  mold server discover --probe         Also measure /health latency

Advertising and server-assisted browsing are on by default when the server is
built with the `mdns` feature; disable both per-server with
`mold serve --no-mdns` or `MOLD_MDNS=0`.")]
    Discover {
        /// Seconds to browse the network before reporting
        #[arg(long, default_value_t = 3)]
        timeout_secs: u64,
        /// Emit JSON instead of a table
        #[arg(long)]
        json: bool,
        /// Probe each server's /health + /api/status and report latency
        #[arg(long)]
        probe: bool,
    },
    /// Stop the managed server
    Stop,
}

#[derive(Subcommand)]
enum ChainSub {
    /// Parse and normalise a TOML script without submitting.
    #[command(after_long_help = "\
Examples:
  mold chain validate shot.toml")]
    Validate {
        /// Path to the TOML chain script.
        #[arg(value_hint = ValueHint::FilePath)]
        path: std::path::PathBuf,
    },
}

#[derive(clap::Subcommand)]
pub enum JobsAction {
    List {
        #[arg(long)]
        json: bool,
    },
    Show {
        id: String,
        #[arg(long)]
        json: bool,
    },
    Resume {
        id: String,
    },
    Retake {
        id: String,
        #[arg(long)]
        stage: u32,
        #[arg(long, value_enum, default_value = "cascade")]
        mode: RetakeModeArg,
        #[arg(long)]
        seed_offset: Option<u64>,
        #[arg(long)]
        prompt: Option<String>,
    },
    Cancel {
        id: String,
    },
    Delete {
        id: String,
        #[arg(long)]
        yes: bool,
    },
    Gc,
}

#[derive(clap::ValueEnum, Clone, Copy)]
pub enum RetakeModeArg {
    Cascade,
    Splice,
}

#[derive(clap::Subcommand)]
pub enum QueueAction {
    /// List queued, running, and held jobs
    ///
    /// Columns: job id, state (running step counter, `Next up`, `#N in
    /// line`, or the scheduler's own actionable reason), model, batch, the
    /// submitted prompt, and when the row was admitted. Held rows are listed
    /// again beneath with the server's error sentence and whether a retry is
    /// allowed.
    List {
        /// Show only held rows
        #[arg(long)]
        held: bool,
        /// Print the raw `GET /api/queue` document as JSON
        #[arg(long)]
        json: bool,
    },
    /// Show one job in full, with its plan entry and batch progress
    Show {
        /// Job id as shown by `mold queue list`
        #[arg(value_name = "JOB-ID")]
        job_id: String,
        /// Print the raw server documents as JSON
        #[arg(long)]
        json: bool,
    },
    /// Cancel jobs by id, the whole waiting queue, or one batch
    Cancel {
        /// Job ids as shown by `mold queue list`
        #[arg(value_name = "JOB-ID")]
        job_ids: Vec<String>,
        /// Cancel every still-queued job; running work is left alone
        #[arg(long, conflicts_with_all = ["job_ids", "batch"])]
        all: bool,
        /// Cancel every non-terminal child of one batch
        #[arg(long, value_name = "BATCH-ID", conflicts_with = "job_ids")]
        batch: Option<String>,
        /// Skip the confirmation prompt for `--all`
        #[arg(long, short = 'y')]
        yes: bool,
    },
    /// Return held jobs to the durable queue
    ///
    /// Only an explicitly retryable hold resumes; one that needs operator
    /// repair is refused by name rather than skipped.
    Retry {
        /// Job ids as shown by `mold queue list --held`
        #[arg(value_name = "JOB-ID")]
        job_ids: Vec<String>,
        /// Retry every retryable hold
        #[arg(long, conflicts_with = "job_ids")]
        held: bool,
    },
    /// Move one queued job to a new place in line
    Move {
        /// Job id as shown by `mold queue list`
        #[arg(value_name = "JOB-ID")]
        job_id: String,
        /// New 0-based position; a value past the tail is clamped by the host
        #[arg(long, value_name = "POSITION")]
        to: usize,
    },
    /// Pause one waiting job, or omit JOB-ID to hold host-wide dispatch
    Pause {
        #[arg(value_name = "JOB-ID")]
        job_id: Option<String>,
    },
    /// Resume one paused job, or omit JOB-ID to resume host-wide dispatch
    Resume {
        #[arg(value_name = "JOB-ID")]
        job_id: Option<String>,
    },
    /// Run the held-row and settled-batch retention sweeps now
    Sweep,
}

#[derive(Subcommand)]
enum VideoUpscaleAction {
    /// Create a durable Framewise upscale from one Library video
    Create {
        #[arg(value_name = "LIBRARY-FILENAME")]
        source: String,
        #[arg(short, long, default_value = "real-esrgan-x4plus:fp16")]
        model: String,
        #[arg(long, env = "MOLD_UPSCALE_TILE_SIZE")]
        tile_size: Option<u32>,
        #[arg(long, env = "MOLD_HOST")]
        host: Option<String>,
        /// Follow the job through terminal publication
        #[arg(long)]
        wait: bool,
    },
    /// List durable Framewise upscale jobs
    List {
        #[arg(long, env = "MOLD_HOST")]
        host: Option<String>,
    },
    /// Print one job as JSON
    Status {
        id: String,
        #[arg(long, env = "MOLD_HOST")]
        host: Option<String>,
    },
    /// Pause after the current frame boundary
    Pause {
        id: String,
        #[arg(long, env = "MOLD_HOST")]
        host: Option<String>,
    },
    /// Resume from the last completed frame checkpoint
    Resume {
        id: String,
        #[arg(long, env = "MOLD_HOST")]
        host: Option<String>,
    },
    /// Cancel without replacing or publishing source media
    Cancel {
        id: String,
        #[arg(long, env = "MOLD_HOST")]
        host: Option<String>,
    },
}

#[derive(clap::Subcommand)]
pub enum TrashAction {
    /// List trashed prints with their purge countdowns
    ///
    /// Columns: filename, title, when the print was trashed, when the
    /// retention sweep purges it (`kept` = retention is keep-forever,
    /// `due` = the next sweep removes it), and size.
    List {
        /// Print the raw `GET /api/gallery?view=trash` rows as JSON
        #[arg(long)]
        json: bool,
    },
    /// Restore trashed prints to the live gallery
    Restore {
        /// Gallery filenames as shown by `mold trash list`
        #[arg(required = true, value_name = "FILENAME")]
        filenames: Vec<String>,
    },
    /// Permanently delete every trashed print on the server
    Empty {
        /// Skip the confirmation prompt
        #[arg(long, short = 'y')]
        yes: bool,
    },
    /// Run the retention sweep now
    ///
    /// Purges trashed prints older than the host's
    /// `gallery.trash_retention_days` and reports how many remain. The
    /// server also sweeps hourly and at startup.
    Sweep,
}

#[derive(clap::Subcommand)]
pub enum LibraryTagAction {
    /// List tags and their print counts
    List {
        #[arg(long)]
        json: bool,
    },
    /// Add one or more tags to existing prints
    Add {
        #[arg(required = true, value_name = "FILENAME")]
        filenames: Vec<String>,
        #[arg(long = "tag", required = true, value_name = "TAG", value_parser = tag_parser)]
        tags: Vec<String>,
    },
    /// Remove one or more tags from existing prints
    Remove {
        #[arg(required = true, value_name = "FILENAME")]
        filenames: Vec<String>,
        #[arg(long = "tag", required = true, value_name = "TAG", value_parser = tag_parser)]
        tags: Vec<String>,
    },
    /// Rename a tag everywhere it is used
    Rename {
        #[arg(value_name = "OLD", value_parser = tag_parser)]
        old: String,
        #[arg(value_name = "NEW", value_parser = tag_parser)]
        new: String,
    },
    /// Delete a tag and detach it from every print
    Delete {
        #[arg(value_name = "TAG", value_parser = tag_parser)]
        tag: String,
        #[arg(long, short = 'y')]
        yes: bool,
    },
}

#[derive(clap::Subcommand)]
pub enum LibraryCollectionAction {
    /// List collections and their print counts
    List {
        #[arg(long)]
        json: bool,
    },
    /// Show one collection and its ordered member filenames
    Show {
        #[arg(value_name = "NAME-OR-SLUG")]
        collection: String,
        #[arg(long)]
        json: bool,
    },
    /// Create a collection
    Create {
        #[arg(value_name = "NAME", value_parser = collection_name_parser)]
        name: String,
        #[arg(long, value_name = "TEXT")]
        description: Option<String>,
    },
    /// Update a collection's name, description, cover, or visibility
    Update {
        #[arg(value_name = "NAME-OR-SLUG")]
        collection: String,
        #[arg(long, value_name = "TEXT", value_parser = collection_name_parser)]
        name: Option<String>,
        #[arg(long, value_name = "TEXT", conflicts_with = "clear_description")]
        description: Option<String>,
        #[arg(long, conflicts_with = "description")]
        clear_description: bool,
        #[arg(long, value_name = "FILENAME", conflicts_with = "clear_cover")]
        cover: Option<String>,
        #[arg(long, conflicts_with = "cover")]
        clear_cover: bool,
        #[arg(long, conflicts_with = "visible")]
        hidden: bool,
        #[arg(long, conflicts_with = "hidden")]
        visible: bool,
    },
    /// Delete a collection without deleting its prints
    Delete {
        #[arg(value_name = "NAME-OR-SLUG")]
        collection: String,
        #[arg(long, short = 'y')]
        yes: bool,
    },
    /// Add existing prints to a collection
    Add {
        #[arg(value_name = "NAME-OR-SLUG")]
        collection: String,
        #[arg(required = true, value_name = "FILENAME")]
        filenames: Vec<String>,
    },
    /// Remove existing prints from a collection
    Remove {
        #[arg(value_name = "NAME-OR-SLUG")]
        collection: String,
        #[arg(required = true, value_name = "FILENAME")]
        filenames: Vec<String>,
    },
}

#[derive(clap::Subcommand)]
pub enum LibraryAction {
    /// List and filter live Library prints
    List {
        #[arg(long, value_name = "TEXT")]
        query: Option<String>,
        #[arg(long = "tag", value_name = "TAG", value_parser = tag_parser)]
        tags: Vec<String>,
        #[arg(long, value_name = "NAME-OR-SLUG")]
        collection: Option<String>,
        #[arg(long)]
        favorite: bool,
        #[arg(long, value_parser = output_format_parser(&[
            "png", "jpeg", "jpg", "gif", "apng", "webp", "mp4", "wav", "glb",
        ]))]
        format: Option<OutputFormat>,
        #[arg(long, default_value_t = 50, value_parser = library_limit_parser)]
        limit: usize,
        #[arg(long, default_value_t = 0)]
        offset: usize,
        #[arg(long)]
        json: bool,
    },
    /// Show one print's metadata and optionally preview it inline
    Show {
        #[arg(value_name = "FILENAME")]
        filename: String,
        #[arg(long, conflicts_with = "preview")]
        json: bool,
        #[arg(long, conflicts_with = "json")]
        preview: bool,
    },
    /// Open the protocol-aware terminal Library grid
    Grid {
        #[arg(long, value_name = "URL", conflicts_with = "local")]
        host: Option<String>,
        #[arg(long, conflicts_with = "host")]
        local: bool,
    },
    /// Set or clear one existing print's title
    Title {
        #[arg(value_name = "FILENAME")]
        filename: String,
        #[arg(value_name = "TEXT", required_unless_present = "clear", conflicts_with = "clear", value_parser = print_title_parser)]
        title: Option<String>,
        #[arg(long, conflicts_with = "title")]
        clear: bool,
    },
    /// Mark existing prints as favorites
    Favorite {
        #[arg(required = true, value_name = "FILENAME")]
        filenames: Vec<String>,
    },
    /// Remove the favorite mark from existing prints
    Unfavorite {
        #[arg(required = true, value_name = "FILENAME")]
        filenames: Vec<String>,
    },
    /// Manage tags on existing prints
    Tag {
        #[command(subcommand)]
        action: LibraryTagAction,
    },
    /// Manage collections and membership
    Collection {
        #[command(subcommand)]
        action: LibraryCollectionAction,
    },
    /// Move live prints into the recoverable gallery trash
    Trash {
        #[arg(required = true, value_name = "FILENAME")]
        filenames: Vec<String>,
    },
    /// Export one stored 3-D print as OBJ, STL, PLY, or a turntable GIF/APNG/WebP
    ///
    /// The gallery keeps its `.glb`; this writes a converted copy beside it or
    /// wherever `--output` names. Each geometry container loses something the
    /// stored glTF carries — OBJ has no materials, STL has no shared vertices
    /// or UVs — which is why none of them is a generation target. An animated
    /// format is a TURNTABLE: the gallery poster's view rendered around the
    /// mesh, framed once for the whole sweep.
    ///
    /// On a host that advertises `capabilities.mesh.export_geometry`, a
    /// geometry container is written print-ready: STL and PLY are scaled to
    /// 100 mm on their longest axis, turned `z` up and rested on the floor,
    /// and OBJ keeps its model units and `y` up. `--size-mm`, `--up-axis` and
    /// `--origin` override that, and are refused outright against a host that
    /// does not advertise the block, which would silently ignore them.
    #[command(after_long_help = "\
Examples:
  mold library export mold-hunyuan3d-1700000000000.glb --format stl
  mold library export chair.glb --format obj -o ~/prints/chair.obj
  mold library export chair.glb --format ply --output -   Write to stdout
  mold library export chair.glb --format stl --size-mm 120
  mold library export chair.glb --format stl --up-axis y --origin center
  mold library export chair.glb --format gif              36 frames, 10 fps, 512 px, loops
  mold library export chair.glb --format gif --playback bounce --repeat once
  mold library export chair.glb --format webp --frames 72 --fps 24 --max-dimension 768")]
    Export {
        #[arg(value_name = "FILENAME")]
        filename: String,
        /// Container: glb, obj, stl, or ply. glb downloads the stored file
        /// unchanged; the rest are transcodes.
        #[arg(long, value_name = "FORMAT", value_parser = mesh_export_format_parser)]
        format: mold_core::MeshExportFormat,
        /// Where to write the converted file. Defaults to the print's stem
        /// with the new extension in the current directory; `-` writes to
        /// stdout.
        #[arg(long, short = 'o', value_name = "PATH")]
        output: Option<String>,
        #[command(flatten)]
        turntable: TurntableArgs,
        #[command(flatten)]
        geometry: GeometryArgs,
    },
}

/// Turntable controls for `mold library export --format gif|apng|webp`. The
/// names are the gallery video export's own (`playback`, `repeat`,
/// `max_dimension`) plus the two only a render has, and every one is optional
/// so an absent flag means the server's default. Ignored by a geometry
/// export.
#[derive(clap::Args, Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TurntableArgs {
    /// GIF only: `loop` is one seamless full turn; `bounce` sweeps half a
    /// turn and plays it back.
    #[arg(long, value_name = "loop|bounce", help_heading = "Turntable")]
    playback: Option<mold_core::MeshTurntablePlayback>,
    /// GIF only: `forever` loops; `once` plays through and rests on the final
    /// frame.
    #[arg(long, value_name = "forever|once", help_heading = "Turntable")]
    repeat: Option<mold_core::MeshTurntableRepeat>,
    /// Frame edge in pixels, 240 to 2048 (default 512, the poster's size).
    #[arg(long, value_name = "PIXELS", help_heading = "Turntable")]
    max_dimension: Option<u32>,
    /// Views rendered around the mesh, 8 to 180 (default 36, a 10° step).
    #[arg(long, value_name = "N", help_heading = "Turntable")]
    frames: Option<u32>,
    /// Playback rate, 1 to 30 (default 10).
    #[arg(long, value_name = "N", help_heading = "Turntable")]
    fps: Option<u32>,
}

impl From<TurntableArgs> for mold_core::MeshTurntableOptions {
    fn from(args: TurntableArgs) -> Self {
        Self {
            playback: args.playback,
            repeat: args.repeat,
            max_dimension: args.max_dimension,
            frames: args.frames,
            fps: args.fps,
        }
    }
}

/// Geometry controls for `mold library export --format obj|stl|ply`.
///
/// Every one is optional, and on a host that advertises
/// `capabilities.mesh.export_geometry` an absent flag means the FORMAT's own
/// default rather than "unchanged": the stored `.glb` is normalized model
/// space, so a verbatim transcode reaches a slicer as a two-millimetre blob
/// lying on its side. STL and PLY default to 100 mm on the longest axis, `z`
/// up, resting on the floor; OBJ keeps its model units and `y` up, because
/// every tool that reads OBJ converts the axis itself and treats one unit as
/// one metre. Refused on `glb` and on a turntable, which have no geometry to
/// shape, and against a host without the block, which would drop them.
#[derive(clap::Args, Debug, Clone, Copy, Default, PartialEq)]
pub struct GeometryArgs {
    /// Longest bounding-box axis in millimetres, 1 to 1000 (default 100 for
    /// stl and ply; obj stays in model units).
    #[arg(long, value_name = "MM", help_heading = "Geometry")]
    size_mm: Option<f64>,
    /// Which world axis points up (default z for stl and ply, y for obj).
    #[arg(long, value_name = "y|z", help_heading = "Geometry")]
    up_axis: Option<mold_core::MeshUpAxis>,
    /// Where the origin sits: `floor` rests the mesh on the ground plane,
    /// `center` puts the bounding-box centre on it (default floor).
    #[arg(long, value_name = "center|floor", help_heading = "Geometry")]
    origin: Option<mold_core::MeshExportOrigin>,
}

impl From<GeometryArgs> for mold_core::MeshGeometryOptions {
    fn from(args: GeometryArgs) -> Self {
        Self {
            size_mm: args.size_mm,
            up_axis: args.up_axis,
            origin: args.origin,
        }
    }
}

#[derive(Subcommand)]
#[allow(clippy::large_enum_variant)]
enum Commands {
    /// Generate images from a text prompt
    ///
    /// First positional arg is treated as MODEL if it matches a known model name.
    /// Remaining args are the prompt.
    #[command(after_long_help = "\
Examples:
  mold run \"a cat on a skateboard\"
  mold run flux-dev:q4 \"a sunset over mountains\"
  mold run \"a cat\" --seed 42 --steps 20 -o cat.png
  mold run \"a cat\" | viu -
  echo \"a dog\" | mold run flux-schnell")]
    Run {
        /// Model name (e.g. flux-dev:q4, flux-schnell)
        #[arg(add = ArgValueCandidates::new(commands::run::complete_model_name))]
        model_or_prompt: Option<String>,

        /// Prompt text (remaining words after model)
        prompt_rest: Vec<String>,

        /// Output file path
        #[arg(short, long, help_heading = "Output", value_hint = ValueHint::FilePath)]
        output: Option<String>,

        /// Output format (defaults to PNG for images, MP4 for video, GLB for
        /// 3-D). `obj` is deliberately absent: mold never STORES an OBJ,
        /// because one carries neither materials nor textures on its own —
        /// it exists only as a gallery export transcode.
        #[arg(long, help_heading = "Output",
              value_parser = output_format_parser(&["png", "jpeg", "jpg", "gif", "apng", "webp", "mp4", "wav", "glb"]))]
        format: Option<OutputFormat>,

        /// Disable embedded generation metadata in PNG output for this run
        #[arg(long, help_heading = "Output")]
        no_metadata: bool,

        /// Print title (up to 120 characters). Embedded in the output
        /// metadata, seeded into the gallery row, and folded into the default
        /// filename as a slug: mold-<model>-<timestamp>~<slug>.<ext>
        #[arg(long, help_heading = "Output", value_name = "TEXT", value_parser = print_title_parser)]
        title: Option<String>,

        /// File the print under a tag. Repeatable, up to 20 tags
        /// (1-64 characters each). Tags are matched case-insensitively.
        #[arg(long = "tag", help_heading = "Output", value_name = "TAG", value_parser = tag_parser)]
        tags: Vec<String>,

        /// File the print into a collection, creating it if it does not
        /// exist yet. Collections merge across machines by name.
        #[arg(long, help_heading = "Output", value_name = "NAME", value_parser = collection_name_parser)]
        collection: Option<String>,

        /// Do not add the title as a tag, whatever
        /// `generate.auto_tag_title` says.
        #[arg(long, help_heading = "Output")]
        no_auto_tag: bool,

        /// Display generated image(s) inline in the terminal after generation
        #[arg(long, env = "MOLD_PREVIEW", help_heading = "Output")]
        preview: bool,

        /// Image width — defaults to model config value
        #[arg(long, help_heading = "Image")]
        width: Option<u32>,

        /// Image height — defaults to model config value
        #[arg(long, help_heading = "Image")]
        height: Option<u32>,

        /// Number of inference steps — defaults to model config value
        #[arg(long, help_heading = "Image")]
        steps: Option<u32>,

        /// Guidance scale — defaults to model config value
        #[arg(long, help_heading = "Image")]
        guidance: Option<f64>,

        /// Random seed
        #[arg(long, help_heading = "Image")]
        seed: Option<u64>,

        /// Number of images to generate
        #[arg(long, default_value = "1", help_heading = "Image", value_parser = clap::value_parser!(u32).range(1..))]
        batch: u32,

        #[command(flatten)]
        mesh: MeshArgs,

        /// Number of video frames to generate (video models only, e.g. ltx-video).
        /// Implies video output mode; release builds default to MP4.
        ///
        /// For LTX-2 distilled, values above 97 automatically chain multiple
        /// clips at render time (see `--clip-frames` / `--motion-tail`).
        #[arg(long, help_heading = "Video")]
        frames: Option<u32>,

        /// Let a qualified LTX-2.5 duration head choose a 1–20 second clip.
        /// This deliberately omits `frames` from the request.
        #[arg(long, conflicts_with_all = ["frames", "duration"], help_heading = "Video")]
        predict_duration: bool,

        /// Video frames per second for output encoding.
        /// Defaults to the selected model's frame rate.
        #[arg(long, help_heading = "Video")]
        fps: Option<u32>,

        /// MiniMax H3 duration in seconds (4 through 15). Resolves to the
        /// nearest exact 17n+5 frame count on H3's fixed 24 FPS clock.
        #[arg(long, conflicts_with = "frames", help_heading = "Video")]
        duration: Option<f64>,

        /// Per-clip frame cap for chained video. When --frames exceeds this,
        /// the CLI splits into multiple chained clips stitched at render time.
        /// Defaults to the model's native cap (97 for LTX-2 distilled).
        #[arg(long, value_name = "N", help_heading = "Video")]
        clip_frames: Option<u32>,

        /// Motion-tail overlap between chained clips in pixel frames. Each clip
        /// after the first reuses this many trailing latents from the prior
        /// clip, trimming the duplicated pixel frames at stitch time. 0 disables
        /// latent carryover (simple concat). Default 17 — three LTX-2 latent
        /// frames of carryover at the 8× causal temporal compression (causal-
        /// first slot + two continuation slots, ≈0.7 s at 24 fps), enough hard-
        /// pinned pixel context to keep scene identity coherent across clips.
        #[arg(long, value_name = "N", default_value_t = 17, help_heading = "Video")]
        motion_tail: u32,

        /// Force synchronized audio for LTX-2 family generation. Audio is
        /// already on by default wherever the model renders it, so this only
        /// matters against a non-MP4 container.
        #[arg(long, help_heading = "Video", conflicts_with = "no_audio")]
        audio: bool,

        /// Render an LTX-2 clip silent. Audio is on by default for this
        /// family, so this is how you turn it off.
        #[arg(long, help_heading = "Video", conflicts_with = "audio")]
        no_audio: bool,

        /// Skip the LTX-2 audio branch entirely (#1037). Output-changing:
        /// the branch feeds the video stream, so this is never a default.
        /// Conflicts with --audio and --audio-file.
        #[arg(long, help_heading = "Video", conflicts_with_all = ["audio", "audio_file"])]
        video_only: bool,

        /// Conditioning audio file for audio-to-video generation.
        #[arg(long, help_heading = "Video", value_hint = ValueHint::FilePath)]
        audio_file: Option<String>,

        /// Source video for retake / video-to-video workflows.
        #[arg(long, help_heading = "Video", value_hint = ValueHint::FilePath)]
        video: Option<String>,

        /// Continue an existing video. The output is that clip followed by
        /// newly generated frames; --frames sets the rendered continuation
        /// length, of which --extend-overlap reproduces the source tail.
        #[arg(
            long,
            help_heading = "Video",
            value_hint = ValueHint::FilePath,
            conflicts_with_all = ["video", "image", "keyframe"],
        )]
        extend: Option<String>,

        /// Pixel frames of the source tail used as motion context for --extend.
        /// Must sit on the family's VAE temporal grid — 8k+1 for LTX-2, 4k+1
        /// for Wan — and be strictly less than --frames. Wan carries exactly
        /// one frame (its seam is image conditioning, not a latent tail), so 1
        /// is the only value it accepts.
        #[arg(long, help_heading = "Video", value_name = "N", requires = "extend")]
        extend_overlap: Option<u32>,

        /// Keyframe conditioning in the form <frame:path>. Repeat for multiple keyframes.
        #[arg(long, help_heading = "Video")]
        keyframe: Vec<String>,

        /// Closing frame for a Wan first/last-frame render. Pairs with the
        /// --image opening frame: the two travel as the clip's endpoint
        /// keyframes and replace the plain source image, so it is mutually
        /// exclusive with hand-written --keyframe entries.
        #[arg(
            long,
            help_heading = "Video",
            value_hint = ValueHint::FilePath,
            requires = "image",
            conflicts_with = "keyframe",
        )]
        last_image: Option<String>,

        /// MiniMax H3 FL2VA opening-frame condition.
        #[arg(
            long,
            conflicts_with = "image",
            help_heading = "MiniMax H3",
            value_hint = ValueHint::FilePath
        )]
        first_frame: Option<std::path::PathBuf>,

        /// MiniMax H3 FL2VA closing-frame condition. It is bound to the final
        /// frame after --duration/--frames resolves onto H3's exact frame grid.
        #[arg(long, help_heading = "MiniMax H3", value_hint = ValueHint::FilePath)]
        last_frame: Option<std::path::PathBuf>,

        /// Ordered reference image, repeatable in semantic order. A bare PATH
        /// is an image; MiniMax H3 Ref2VA also takes video=PATH and
        /// audio=PATH, whose files use authenticated, request-bound streaming
        /// upload and are never placed inline in JSON.
        #[arg(long, value_name = "PATH", help_heading = "Reference images")]
        reference: Vec<commands::h3::ReferenceArg>,

        /// LTX-2 pipeline mode.
        #[arg(long, help_heading = "Video", value_enum)]
        pipeline: Option<Ltx2PipelineArg>,

        /// Official LTX-2 IC-LoRA reference control.
        #[arg(long, value_name = "ID", help_heading = "Video", requires = "video")]
        ic_lora_control: Option<String>,

        /// Also write the render as an OpenEXR sequence in this directory, in
        /// scene-referred linear HDR. Requires `--ic-lora-control hdr`, whose
        /// adapter is what makes the render HDR. The ordinary video is still
        /// written and is what lands in the gallery.
        #[arg(long, value_name = "DIR", help_heading = "Video")]
        hdr_exr_dir: Option<String>,

        /// Write EXR samples at full 32-bit float instead of 16-bit half.
        #[arg(long, help_heading = "Video", requires = "hdr_exr_dir")]
        hdr_exr_full_float: bool,

        /// Retake time range in the form <start:end> seconds.
        #[arg(long, help_heading = "Video")]
        retake: Option<String>,

        /// Spatial upscaling mode for LTX-2.3.
        #[arg(long, help_heading = "Video", value_enum)]
        spatial_upscale: Option<Ltx2SpatialUpscaleArg>,

        /// Temporal upscaling mode for LTX-2.3.
        #[arg(long, help_heading = "Video", value_enum)]
        temporal_upscale: Option<Ltx2TemporalUpscaleArg>,

        /// LTX-2 spatiotemporal guidance (STG) scale. 0 disables STG.
        ///
        /// Overrides the pipeline's own constant (1.0 for two-stage /
        /// keyframe / a2-vid, 0 for two-stage-hq). Higher values add motion
        /// structure and detail at the cost of stability.
        #[arg(long, value_name = "SCALE", help_heading = "Video")]
        stg_scale: Option<f64>,

        /// Transformer blocks perturbed for STG (comma-separated, e.g. 28,29).
        ///
        /// Defaults to the checkpoint's own block (29 for LTX-2 19B, 28 for
        /// LTX-2.3 22B). Deeper blocks are weaker; earlier blocks are stronger.
        #[arg(long, value_name = "BLOCKS", help_heading = "Video")]
        stg_blocks: Option<String>,

        /// LTX-2 CFG-rescale factor between 0 and 1.
        ///
        /// Rescales the guided prediction toward the conditional prediction's
        /// standard deviation. Raise it when high guidance washes out contrast.
        #[arg(long, value_name = "SCALE", help_heading = "Video")]
        rescale_scale: Option<f64>,

        /// LTX-2 cross-modality (audio ↔ video) guidance scale. 1 disables it.
        #[arg(long, value_name = "SCALE", help_heading = "Video")]
        modality_scale: Option<f64>,

        /// Apply LTX-2 guidance on every Nth+1 step only (0 = every step).
        ///
        /// Trades a little prompt adherence for a shorter denoise: each
        /// skipped step takes the conditional prediction directly.
        #[arg(long, value_name = "N", help_heading = "Video")]
        guidance_skip_step: Option<u32>,

        /// Wan sample solver: unipc (default), euler, or dpm++ — upstream's
        /// --sample_solver. euler is the solver the 4-step Lightning distills
        /// were tuned for; dpm++ matches upstream's alternative grid.
        /// (MOLD_WAN_SOLVER is deliberately NOT a clap env binding: it would
        /// inject a wan-only field into every family's request. The wan
        /// engine reads it itself, so the env still applies to wan renders.)
        #[arg(
            long,
            value_parser = ["unipc", "euler", "dpm++", "dpmpp", "dpm-pp"],
            help_heading = "Video",
            conflicts_with = "scheduler"
        )]
        sample_solver: Option<String>,

        /// Wan flow shift (upstream --sample_shift): the family's primary
        /// quality/character knob. Overrides the per-tier default for this
        /// run. Upstream ships 3.0-16 per task; Lightning wants 5, upstream
        /// quality A14B T2V wants 12, ComfyUI templates ship 8.
        /// (MOLD_WAN_SHIFT reaches wan renders engine-side, same as above.)
        #[arg(long, value_name = "SHIFT", help_heading = "Video")]
        sample_shift: Option<f64>,

        /// Wan Lightning distill strength: `high=X,low=Y` (either half
        /// optional) or one number for both experts. The community's
        /// reduced-motion mitigation runs high=1.5..2.0 with low=1.0.
        #[arg(long, value_name = "SPEC", help_heading = "Video")]
        distill_strength: Option<String>,

        /// Camera-control LoRA preset name or .safetensors path.
        ///
        /// Preset aliases (dolly-in, dolly-left, dolly-out, dolly-right,
        /// jib-down, jib-up, static) currently resolve only Lightricks' LTX-2
        /// 19B LoRAs; LTX-2.3 has no published presets yet, so use an explicit
        /// .safetensors path for LTX-2.3.
        #[arg(long, help_heading = "Video")]
        camera_control: Option<String>,

        /// Server URL to connect to
        #[arg(long, env = "MOLD_HOST", help_heading = "Server")]
        host: Option<String>,

        /// Skip server and run inference locally (requires GPU features)
        #[arg(long, help_heading = "Server")]
        local: bool,

        /// Prompt(s). Repeat for a multi-stage uniform chain with smooth
        /// transitions. For heterogeneous stages, use --script.
        #[arg(long, help_heading = "Image")]
        prompt: Vec<String>,

        /// Per-clip frame cap for multi-prompt sugar. Clamped to the model's
        /// per-clip cap. Only used when --prompt is repeated. Defaults to 97
        /// (the LTX-2 19B/22B distilled cap).
        #[arg(long, value_name = "N", help_heading = "Video")]
        frames_per_clip: Option<u32>,

        /// Path to a `mold.chain.v1` TOML script. When set, every other
        /// generation flag is ignored except `--output`, `--local`, `--host`,
        /// and `--dry-run`.
        #[arg(long, value_name = "PATH", help_heading = "Server")]
        script: Option<std::path::PathBuf>,

        /// Parse and normalise the script without submitting. Prints the
        /// canonical stage list and estimated total frames to stdout.
        #[arg(long, help_heading = "Server")]
        dry_run: bool,

        /// GPUs for local generation: all, none, ordinals, or stable cuda:/metal:/GPU-/MIG- IDs
        #[arg(long, env = "MOLD_GPUS", help_heading = "Advanced")]
        gpus: Option<String>,

        /// T5 encoder variant: auto (default), fp16, q8, q6, q5, q4, q3
        #[arg(long, help_heading = "Advanced")]
        t5_variant: Option<String>,

        /// Qwen3 text encoder variant (Z-Image): auto (default), bf16, q8, q6, iq4, q3
        #[arg(long, help_heading = "Advanced")]
        qwen3_variant: Option<String>,

        /// Qwen2.5-VL text encoder variant (Qwen-Image): auto, bf16, q8, q6, q5, q4, q3, q2
        #[arg(
            long,
            env = "MOLD_QWEN2_VARIANT",
            value_parser = ["auto", "bf16", "q8", "q6", "q5", "q4", "q3", "q2"],
            help_heading = "Advanced"
        )]
        qwen2_variant: Option<String>,

        /// Qwen2.5-VL text encoder mode (Qwen-Image): auto, gpu, cpu-stage, cpu
        /// `auto` respects the selected variant; BF16 on Metal stages after encoding, CUDA defaults stay unchanged.
        #[arg(
            long,
            env = "MOLD_QWEN2_TEXT_ENCODER_MODE",
            value_parser = ["auto", "gpu", "cpu-stage", "cpu"],
            help_heading = "Advanced"
        )]
        qwen2_text_encoder_mode: Option<String>,

        /// Scheduler algorithm for UNet models: ddim, euler-ancestral, uni-pc,
        /// or edm-dpm-pp-2m (Playground v2.5 only)
        /// Ignored by flow-matching models (FLUX, SD3, Z-Image, Flux.2, Qwen-Image).
        #[arg(long, env = "MOLD_SCHEDULER", help_heading = "Advanced")]
        scheduler: Option<Scheduler>,

        /// Enable CFG++ (manifold-projection guidance, Chung et al. 2024).
        /// Lowers usable CFG to ~1.5–2.5 and reduces guidance artifacts.
        /// Supported on SD3, SDXL, and SD1.5 (DDIM scheduler only — others
        /// fall back to standard CFG with a warn). Ignored by guidance-
        /// distilled families (FLUX, Z-Image, Flux.2) and whenever guidance
        /// does not activate CFG.
        #[arg(long, env = "MOLD_CFG_PLUS", help_heading = "Advanced")]
        cfg_plus: bool,

        /// Keep all model components loaded simultaneously (faster but uses more memory).
        /// By default, components are loaded and unloaded sequentially to reduce peak memory.
        #[arg(long, help_heading = "Advanced")]
        eager: bool,

        /// Stream transformer blocks between CPU and GPU one at a time.
        /// Reduces VRAM from ~24GB to ~2-4GB for large models (3-5x slower).
        /// Auto-enabled when VRAM is insufficient. Force with MOLD_OFFLOAD=1.
        #[arg(long, help_heading = "Advanced")]
        offload: bool,

        /// LTX-2 only: split spatial work into overlapping tiles.
        /// `auto` (default) refines and decodes in tiles only past the
        /// 2048-px span the checkpoints were trained on; `off` never tiles;
        /// `<px>` or `<px>:<overlap>` (multiples of 32) forces that tile size.
        /// Equivalent to MOLD_LTX2_SPATIAL_TILE, which `mold serve` reads.
        #[arg(
            long,
            value_name = "off|auto|PX[:OVERLAP]",
            env = "MOLD_LTX2_SPATIAL_TILE",
            help_heading = "Advanced"
        )]
        spatial_tile: Option<String>,

        /// Place all text encoders (T5/CLIP/Qwen) on a specific device.
        /// Accepts `auto` (default), `cpu`, `gpu` (= `gpu:0`), or `gpu:N`.
        /// Applied to every model family. CLI flag overrides
        /// `MOLD_PLACE_TEXT_ENCODERS` and any config-file placement.
        #[arg(long = "device-text-encoders", help_heading = "Placement")]
        device_text_encoders: Option<String>,

        /// Place the transformer (FLUX / Flux.2 / Z-Image / Qwen-Image only).
        /// Accepts `auto`, `cpu`, `gpu`, `gpu:N`. Overrides `MOLD_PLACE_TRANSFORMER`.
        #[arg(long = "device-transformer", help_heading = "Placement")]
        device_transformer: Option<String>,

        /// Place the VAE (FLUX / Flux.2 / Z-Image / Qwen-Image only).
        /// Accepts `auto`, `cpu`, `gpu`, `gpu:N`. Overrides `MOLD_PLACE_VAE`.
        #[arg(long = "device-vae", help_heading = "Placement")]
        device_vae: Option<String>,

        /// Place the T5 text encoder (FLUX only). Accepts `auto`, `cpu`,
        /// `gpu`, `gpu:N`. Overrides `MOLD_PLACE_T5`.
        #[arg(long = "device-t5", help_heading = "Placement")]
        device_t5: Option<String>,

        /// Place CLIP-L (FLUX only). Accepts `auto`, `cpu`, `gpu`, `gpu:N`.
        /// Overrides `MOLD_PLACE_CLIP_L`.
        #[arg(long = "device-clip-l", help_heading = "Placement")]
        device_clip_l: Option<String>,

        /// Place CLIP-G. Accepts `auto`, `cpu`, `gpu`, `gpu:N`.
        /// Overrides `MOLD_PLACE_CLIP_G`.
        #[arg(long = "device-clip-g", help_heading = "Placement")]
        device_clip_g: Option<String>,

        /// Place the Qwen text encoder (Flux.2 / Z-Image / Qwen-Image).
        /// Accepts `auto`, `cpu`, `gpu`, `gpu:N`. Overrides `MOLD_PLACE_QWEN`.
        #[arg(long = "device-qwen", help_heading = "Placement")]
        device_qwen: Option<String>,

        /// LoRA adapter safetensors file path. Repeat for multiple LTX-2 adapters.
        /// Suffix `@high` or `@low` to bind an adapter to one Wan 2.2 A14B
        /// expert; the community publishes those as non-interchangeable pairs.
        #[arg(long, help_heading = "LoRA", value_hint = ValueHint::FilePath)]
        lora: Vec<String>,

        /// LoRA effect strength (0.0 = none, 1.0 = full, up to 2.0)
        #[arg(long, default_value = "1.0", help_heading = "LoRA")]
        lora_scale: f64,

        /// Source image(s). Repeat for multi-image edit models; use `-` for stdin on single-image families only.
        #[arg(short = 'i', long, help_heading = "img2img", value_hint = ValueHint::FilePath)]
        image: Vec<String>,

        /// img2img/I2V strength (default: 0.75). SD-family img2img: higher
        /// = more change (1.0 = full noise). LTX-2 I2V: higher = more
        /// source preservation (1.0 pins the opening frame).
        #[arg(long, help_heading = "img2img")]
        strength: Option<f64>,

        /// Mask image for inpainting (file path; white = repaint, black = preserve)
        #[arg(long, requires = "image", help_heading = "img2img", value_hint = ValueHint::FilePath)]
        mask: Option<String>,

        /// Reference photograph to preserve the face of (PuLID). Repeat up to
        /// 4 times to average several references of the same person.
        /// Supported across FLUX.1 (`mold pull pulid-flux`) and SDXL except
        /// SDXL Turbo (`mold pull pulid-sdxl`), on a server built with the
        /// `pulid` feature. Either bundle needs
        /// `--accept-license insightface-antelopev2`; a machine that already
        /// has one pulls only the other's adapter.
        #[arg(
            long,
            conflicts_with_all = ["image", "lora"],
            help_heading = "Identity",
            value_hint = ValueHint::FilePath
        )]
        id_image: Vec<std::path::PathBuf>,

        /// Identity strength, 0.0-3.0 (default: 1.0). Exactly 0.0 renders the
        /// unconditioned print — nothing is pulled, loaded, or extracted.
        #[arg(long, requires = "id_image", help_heading = "Identity")]
        id_weight: Option<f64>,

        /// First denoise step identity is applied from (default: 0). Must be
        /// below --steps.
        #[arg(long, requires = "id_image", help_heading = "Identity")]
        id_start_step: Option<u32>,

        /// True classifier-free guidance scale, 1.0-10.0 (default: 1.0 = off).
        /// Above 1.0 each step from --cfg-start-step runs a second forward
        /// over --negative-prompt and the unconditional identity. Requires
        /// --id-image; upstream recommends lowering --guidance to 1.0 with it.
        /// FLUX only — it exists because FLUX [dev] is guidance-distilled, and
        /// on SDXL --guidance already IS the classifier-free scale.
        #[arg(long, requires = "id_image", help_heading = "Identity")]
        true_cfg: Option<f64>,

        /// First denoise step the true-CFG negative branch runs at
        /// (default: 1). Must be below --steps.
        #[arg(long, requires = "true_cfg", help_heading = "Identity")]
        cfg_start_step: Option<u32>,

        /// Control image for ControlNet conditioning (file path, e.g. edges.png)
        #[arg(long, help_heading = "ControlNet", value_hint = ValueHint::FilePath)]
        control: Option<String>,

        /// ControlNet model name (e.g. controlnet-canny-sd15)
        #[arg(long, requires = "control", help_heading = "ControlNet")]
        control_model: Option<String>,

        /// ControlNet conditioning scale (0.0 = no effect, 1.0 = full, up to 2.0)
        #[arg(long, default_value = "1.0", help_heading = "ControlNet")]
        control_scale: f64,

        /// Negative prompt — what to avoid generating (CFG-based models: SD1.5, SDXL, SD3, Wuerstchen, and the undistilled flux2-klein-base tiers)
        #[arg(short = 'n', long, help_heading = "Image")]
        negative_prompt: Option<String>,

        /// Disable every default negative prompt — config-file defaults and
        /// the model's tuned default (wan) — by sending an explicit empty
        /// unconditional
        #[arg(long, help_heading = "Image")]
        no_negative: bool,

        /// Enable LLM-powered prompt expansion
        #[arg(long, env = "MOLD_EXPAND", help_heading = "Expansion")]
        expand: bool,

        /// Disable prompt expansion (overrides config/env default)
        #[arg(long, conflicts_with = "expand", help_heading = "Expansion")]
        no_expand: bool,

        /// Expansion backend: "local" for built-in GGUF, or an OpenAI-compatible API URL
        #[arg(long, env = "MOLD_EXPAND_BACKEND", help_heading = "Expansion")]
        expand_backend: Option<String>,

        /// LLM model for expansion (local or API model name)
        #[arg(long, env = "MOLD_EXPAND_MODEL", help_heading = "Expansion")]
        expand_model: Option<String>,

        /// Upscale generated images with this model (e.g. real-esrgan-x4plus:fp16)
        #[arg(long, help_heading = "Upscale", add = ArgValueCandidates::new(commands::upscale::complete_upscaler_model))]
        upscale: Option<String>,
    },

    /// Install the mold Agent Skill for AI coding agents
    ///
    /// Renders an agent-compatible managed skill bundle and installs it for
    /// Claude Code, OpenAI Codex CLI, Pi, OpenClaw, GitHub Copilot CLI, Cursor,
    /// Gemini CLI, Amp, Goose, or the generic Agent Skills directory.
    Skill(skill::SkillArgs),

    /// Start the inference server
    #[command(after_long_help = "\
Examples:
  mold serve
  mold serve --port 8080
  mold serve --bind 127.0.0.1 --port 9000
  MOLD_PORT=8080 mold serve

Clients connect via MOLD_HOST=http://<addr>:<port>

For gated or private Hugging Face repos, export HF_TOKEN in the server
environment before starting mold serve.")]
    Serve {
        /// Server port
        #[arg(long, env = "MOLD_PORT", default_value_t = 7680)]
        port: u16,

        /// Bind address
        #[arg(long, default_value = "0.0.0.0")]
        bind: String,

        /// Models directory
        #[arg(long, env = "MOLD_MODELS_DIR", value_hint = ValueHint::DirPath)]
        models_dir: Option<String>,

        /// Log output format
        #[arg(long, default_value = "json")]
        log_format: LogFormat,

        /// Write logs to file (~/.mold/logs/)
        #[arg(long)]
        log_file: bool,

        /// GPUs to use: all, none, ordinals, or stable cuda:/metal:/GPU-/MIG- IDs
        #[arg(long, env = "MOLD_GPUS")]
        gpus: Option<String>,

        /// Jobs hydrated into the runtime window; the durable backlog is uncapped
        #[arg(long, env = "MOLD_QUEUE_SIZE", default_value_t = 200)]
        queue_size: usize,

        /// Also start the Discord bot in this process
        #[cfg(feature = "discord")]
        #[arg(long)]
        discord: bool,

        /// Disable mDNS/DNS-SD advertising and server-assisted browsing
        #[cfg(feature = "mdns")]
        #[arg(long)]
        no_mdns: bool,
    },

    /// Start a stdio MCP server for LM Studio and other MCP hosts
    #[command(after_long_help = "\
Examples:
  mold mcp
  mold mcp --host http://localhost:7680

Use with LM Studio by adding a stdio MCP entry whose command is the mold binary
and whose args are [\"mcp\", \"--host\", \"http://localhost:7680\"]. Run
`mold serve` separately before calling generation tools.")]
    Mcp {
        /// Server URL to connect to
        #[arg(long, env = "MOLD_HOST")]
        host: Option<String>,
    },

    /// Manage a background mold server daemon (start, stop, status, discover)
    #[command(after_long_help = "\
Examples:
  mold server start              Start background server on port 7680
  mold server start --port 8080  Custom port
  mold server status             Check if server is running
  mold server stop               Stop the server
  mold server discover           Find mold servers on the local network (mDNS)")]
    Server {
        #[command(subcommand)]
        action: ServerAction,
    },

    /// Script-mode chain authoring tools
    #[command(after_long_help = "\
Examples:
  mold chain validate shot.toml    Parse and normalise a TOML chain script")]
    Chain {
        #[command(subcommand)]
        action: ChainSub,
    },

    /// Manage durable chained video jobs
    Jobs {
        #[command(subcommand)]
        action: JobsAction,
    },

    /// Inspect and control the generation queue on a running server
    #[command(after_long_help = "\
Examples:
  mold queue list                      Everything queued, running, or held
  mold queue list --held               Only the parked rows, with their errors
  mold queue show job-abc123           One job in full, with its batch progress
  mold queue cancel job-abc123
  mold queue cancel --all --yes        Clear the backlog; running work finishes
  mold queue cancel --batch batch-7    Cancel one batch's remaining children
  mold queue retry --held              Resume every retryable hold
  mold queue move job-abc123 --to 0    Send a job to the head of the line
  mold queue pause job-abc123          Pause only one waiting job
  mold queue pause / mold queue resume Host-wide dispatch gate
  mold queue sweep                     Run the retention sweeps now

Talks to the server at MOLD_HOST (MOLD_API_KEY when configured). There is
no local fallback: a queue belongs to one serving host. `list`, `show`, and
the waiting vocabulary (`Next up`, `#N in line`) are the same policy the web,
desktop, and iPhone surfaces render.")]
    Queue {
        #[command(subcommand)]
        action: QueueAction,
    },

    /// Browse and organize existing Library prints on a running server
    #[command(after_long_help = "\
Examples:
  mold library list --tag portrait --favorite
  mold library show mold-flux-dev-1700000000000.png --preview
  mold library tag add a.png b.png --tag portrait --tag selected
  mold library collection add Portfolio a.png b.png
  mold library collection remove Portfolio b.png
  mold library trash a.png
  mold library grid

Non-grid commands target MOLD_HOST (with MOLD_API_KEY when configured) and
never fall back to direct filesystem access. The grid opens the existing Mold
TUI Library; an unreachable explicit host is an error, not a local fallback.")]
    Library {
        #[command(subcommand)]
        action: LibraryAction,
    },

    /// Inspect, restore, or empty the gallery trash on a running server
    #[command(after_long_help = "\
Examples:
  mold trash list                      Trashed prints with purge countdowns
  mold trash restore mold-flux-dev-q4-1700000000000.png
  mold trash empty --yes               Purge everything without confirming
  mold trash sweep                     Run the retention sweep now

Talks to the server at MOLD_HOST (MOLD_API_KEY when configured). There is
no local fallback: the trash belongs to that host's gallery. Retention is
the host's `gallery.trash_retention_days` config key (default 30; 0 keeps
trashed prints forever).")]
    Trash {
        #[command(subcommand)]
        action: TrashAction,
    },

    /// Download model weights via the running server, or locally if no server is reachable
    #[command(after_long_help = "\
Examples:
  mold pull flux-schnell:q8
  mold pull sdxl-turbo:fp16

If MOLD_HOST is reachable, the download happens on that server.
If no server is reachable, mold pulls locally.

For gated or private Hugging Face repos, export HF_TOKEN=hf_... before pulling.
When using a remote server, HF_TOKEN must be set in the server process
environment.

Run 'mold list' to see all available models.")]
    Pull {
        /// Model name to download
        #[arg(add = ArgValueCandidates::new(commands::run::complete_model_name))]
        model: String,

        /// Skip SHA-256 verification after download
        #[arg(long)]
        skip_verify: bool,

        /// Record acceptance of a third-party model license before pulling
        /// (e.g. `insightface-antelopev2`). Some auxiliary assets are
        /// published under terms mold will not accept on your behalf. Repeat
        /// the flag for a bundle covered by more than one agreement.
        #[arg(long, value_name = "ID", action = clap::ArgAction::Append)]
        accept_license: Vec<String>,
    },

    /// Show third-party model licenses and whether they have been accepted
    #[command(after_long_help = "\
Acceptance is recorded per Mold data root, so this reports the machine that
runs the pull: the server at MOLD_HOST when one answers, otherwise this
machine. Accept terms without downloading anything:

  mold licenses accept tencent-hunyuan3d-2.0

or accept as part of the pull it unblocks:

  mold pull pulid-flux --accept-license insightface-antelopev2")]
    Licenses {
        #[command(subcommand)]
        action: Option<LicensesAction>,

        /// Read this machine's own acceptances instead of asking the server
        #[arg(long)]
        local: bool,
    },

    /// Remove downloaded model(s) and their unique files
    #[command(
        alias = "remove",
        after_long_help = "\
Examples:
  mold rm flux-dev:q4
  mold rm flux-dev:q4 sdxl-turbo:fp16 --force

Files shared between models (e.g. VAE, CLIP) are kept until no model references them."
    )]
    Rm {
        /// Model name(s) to remove
        #[arg(required = true, num_args = 1..)]
        #[arg(add = ArgValueCandidates::new(commands::rm::complete_installed_model_name))]
        models: Vec<String>,

        /// Skip confirmation prompt
        #[arg(short, long)]
        force: bool,
    },

    /// List locally available models — shows installed models with disk usage, plus models available to pull
    #[command(alias = "ls")]
    List,

    /// Show disk usage overview for models, output, logs, and shared components
    #[command(after_long_help = "\
Examples:
  mold stats               Show disk usage summary
  mold stats --json        Machine-readable output")]
    Stats {
        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Clean up orphaned files, stale downloads, and old output images
    ///
    /// Dry-run by default — shows what would be removed without deleting anything.
    /// Use --force to actually delete files.
    #[command(after_long_help = "\
Examples:
  mold clean                          Dry-run: show what would be cleaned
  mold clean --force                  Actually delete orphaned/stale files
  mold clean --older-than 30d         Include output images older than 30 days
  mold clean --older-than 7d --force  Delete old output images

Detects:
  - Stale .pulling markers from interrupted downloads (>6 hours old)
  - Orphaned shared files not referenced by any installed model
  - hf-cache transient files (locks, partial downloads, dangling symlinks)
  - Output images older than the specified age (with --older-than)")]
    Clean {
        /// Actually delete files (default is dry-run)
        #[arg(long)]
        force: bool,

        /// Clean output images older than this duration (e.g. 30d, 7d, 24h, 12h)
        #[arg(long, value_name = "DURATION")]
        older_than: Option<String>,
    },

    /// Show detailed model information, or installation overview when no model is given
    #[command(after_long_help = "\
Examples:
  mold info                          Installation overview
  mold info flux-dev:q4              Model details
  mold info sdxl-turbo:fp16 --verify Verify file integrity")]
    Info {
        /// Model name (e.g. flux-dev:q4). Omit for installation overview.
        #[arg(add = ArgValueCandidates::new(commands::run::complete_model_name))]
        model: Option<String>,

        /// Verify file integrity via SHA-256 checksums (requires a model name)
        #[arg(long)]
        verify: bool,
    },

    /// Get or set the default model
    ///
    /// With no argument, shows the current default model and how it was resolved.
    /// With a model name, sets it as the default in the config file.
    #[command(after_long_help = "\
Examples:
  mold default                   Show current default model
  mold default flux-dev:q4       Set default to flux-dev:q4
  mold default sdxl-turbo        Set default (bare name auto-resolves)

The default model is used by 'mold run' when no model is specified.
The MOLD_DEFAULT_MODEL env var takes precedence over the config file.")]
    Default {
        /// Model name to set as default (e.g. flux-dev:q4). Omit to show current default.
        #[arg(add = ArgValueCandidates::new(commands::default::complete_model_name))]
        model: Option<String>,
    },

    /// View and edit configuration settings
    ///
    /// Get, set, and list all config.toml settings using dot-notation keys.
    #[command(after_long_help = "\
Examples:
  mold config list                                  Show all settings
  mold config get server_port                       Get a single value
  mold config get server_port --raw                 Raw value for scripting
  mold config set server_port 8080                  Set a value
  mold config set expand.enabled true               Nested key
  mold config set output_dir none                   Clear optional field
  mold config set models.flux-dev:q4.default_steps 30   Per-model setting
  mold config list --json                           Machine-readable output
  mold config path                                  Config file location
  mold config edit                                  Open in $EDITOR")]
    Config {
        /// Operate on the named profile (v6 settings scoping). Overrides
        /// the `MOLD_PROFILE` env var for the duration of this command.
        #[arg(long, global = true, value_name = "NAME")]
        profile: Option<String>,
        #[command(subcommand)]
        action: ConfigAction,
    },

    /// Manage RunPod cloud GPU pods end-to-end
    #[command(after_long_help = "\
Set up once:
  mold config set runpod.api_key <key>              Save key to config
  export RUNPOD_API_KEY=<key>                       Or use env var
  mold runpod doctor                                Verify auth

Generate on a fresh pod (the killer feature):
  mold runpod run \"a cat on a skateboard\"           Creates pod → generates → saves
  mold runpod run \"a sunset\" --keep                Leave pod up for reuse

Manage pods manually:
  mold runpod gpus                                  List GPU stock
  mold runpod create --gpu 5090                     Create a pod
  mold runpod list                                  List active pods
  mold runpod connect <pod-id>                      Print export MOLD_HOST=…
  mold runpod delete <pod-id>                       Tear down")]
    Runpod {
        #[command(subcommand)]
        action: RunpodAction,
    },

    /// Deploy and manage private mold servers on Lambda Cloud
    #[command(after_long_help = "\
Set up once:
  mold config set lambda.api_key <key>
  mold lambda doctor

Deploy a private tunneled web UI:
  mold lambda availability
  mold lambda deploy --instance-type gpu_1x_a10 --region us-west-1

Clean up:
  mold lambda terminate
  mold lambda reset --to-zero")]
    Lambda {
        #[command(subcommand)]
        action: LambdaAction,
    },

    /// Preview LLM prompt expansion without generating
    ///
    /// Expand a short prompt into detailed generation prompts using an LLM.
    /// Useful for previewing what --expand will produce.
    #[command(after_long_help = "\
Examples:
  mold expand \"a cat\"
  mold expand \"a cat\" --model flux-schnell
  mold expand \"she turns\" --model ltx-2-19b-distilled:fp8 --task image-to-video
  mold expand \"cyberpunk city\" --variations 5
  mold expand \"a cat\" --variations 3 --json
  mold expand \"the balloon lifts off\" --model wan22-i2v-a14b:q5 --frames 81 --fps 16 --reference image:first-frame")]
    Expand {
        /// Text prompt to expand
        prompt: String,

        /// Target diffusion model (used for model-aware prompt style)
        #[arg(short, long, add = ArgValueCandidates::new(commands::run::complete_model_name))]
        model: Option<String>,

        /// Number of prompt variations to generate
        #[arg(long, default_value = "1")]
        variations: usize,

        /// Output as JSON array
        #[arg(long)]
        json: bool,

        /// Expansion backend override
        #[arg(long)]
        backend: Option<String>,

        /// LLM model name override
        #[arg(long)]
        expand_model: Option<String>,

        /// Resolved conditioning policy for previewing without attached media
        #[arg(long, value_name = "TASK")]
        task: Option<String>,

        /// Canvas width the prompt targets (context for the expander)
        #[arg(long, value_name = "PX")]
        width: Option<u32>,

        /// Canvas height the prompt targets (context for the expander)
        #[arg(long, value_name = "PX")]
        height: Option<u32>,

        /// Frame count the prompt targets (video families)
        #[arg(long, value_name = "N")]
        frames: Option<u32>,

        /// Frames per second the prompt targets (video families)
        #[arg(long, value_name = "N")]
        fps: Option<u32>,

        /// Frames per clip when the run auto-chains past one clip
        #[arg(long, value_name = "N")]
        clip_frames: Option<u32>,

        /// Attached reference to describe, in order: image|video|audio[:role]
        /// where role is first-frame, last-frame, keyframe, source, identity,
        /// edit, or reference (repeatable)
        #[arg(long, value_name = "KIND[:ROLE]")]
        reference: Vec<String>,
    },

    /// Preview subject-preserving prompt alternatives without generating.
    #[command(after_long_help = "\
Examples:
  mold remix \"a cat astronaut\"
  mold remix \"she turns\" --model ltx-2-19b-distilled:fp8 --task image-to-video
  mold remix \"a lighthouse\" --dimensions camera,lighting --variations 4
  mold remix \"a cat\" --source original --root-prompt \"a cat\" --json")]
    Remix {
        /// Exact prompt to use as the remix source.
        source_prompt: String,
        /// Target generation model, used for family-aware policy.
        #[arg(short, long, add = ArgValueCandidates::new(commands::run::complete_model_name))]
        model: Option<String>,
        /// Exact number of alternatives to return.
        #[arg(long, default_value = "3")]
        variations: usize,
        /// Output the structured Remix response as JSON.
        #[arg(long)]
        json: bool,
        /// Expansion backend override.
        #[arg(long)]
        backend: Option<String>,
        /// LLM model override.
        #[arg(long)]
        expand_model: Option<String>,
        /// Resolved conditioning policy.
        #[arg(long, value_name = "TASK")]
        task: Option<String>,
        /// How the source relates to the visible prompt history.
        #[arg(long, default_value = "direct", value_parser = ["original", "current", "direct"])]
        source: String,
        /// Earliest known user prompt, when different from the remix source.
        #[arg(long)]
        root_prompt: Option<String>,
        /// Creative dimensions to vary; repeat or pass comma-separated values.
        #[arg(long, value_delimiter = ',')]
        dimensions: Vec<String>,
        /// Locked style constraint retained in every alternative.
        #[arg(long)]
        style: Option<String>,

        /// Canvas width the prompt targets (context for the expander)
        #[arg(long, value_name = "PX")]
        width: Option<u32>,

        /// Canvas height the prompt targets (context for the expander)
        #[arg(long, value_name = "PX")]
        height: Option<u32>,

        /// Frame count the prompt targets (video families)
        #[arg(long, value_name = "N")]
        frames: Option<u32>,

        /// Frames per second the prompt targets (video families)
        #[arg(long, value_name = "N")]
        fps: Option<u32>,

        /// Frames per clip when the run auto-chains past one clip
        #[arg(long, value_name = "N")]
        clip_frames: Option<u32>,

        /// Attached reference to describe, in order: image|video|audio[:role]
        /// where role is first-frame, last-frame, keyframe, source, identity,
        /// edit, or reference (repeatable)
        #[arg(long, value_name = "KIND[:ROLE]")]
        reference: Vec<String>,
    },

    /// Unload the current model from the server to free GPU memory
    #[command(
        after_long_help = "Requires a running server (mold serve). Use 'mold ps' to check status."
    )]
    Unload,

    /// Show server status and loaded models
    #[command(after_long_help = "Use 'mold unload' to free GPU memory when idle.")]
    Ps,

    /// Inspect and administer server compute devices
    Gpu {
        #[command(subcommand)]
        action: GpuAction,
    },

    /// Show version information
    Version,

    /// Inspect or administer this local machine (never redirects to MOLD_HOST)
    System {
        #[command(subcommand)]
        action: commands::system::SystemAction,
    },

    /// Update mold to the latest release from GitHub
    ///
    /// Checks for new releases, downloads the appropriate platform binary,
    /// verifies its SHA256 checksum, and replaces the current binary in-place.
    #[command(
        disable_version_flag = true,
        after_long_help = "\
Examples:
  mold update                   Update to latest release
  mold update --nightly         Update to latest rolling build from main
  mold update --check           Check for updates without installing
  mold update --version v0.7.0  Install a specific version
  mold update --force           Reinstall even if already up-to-date"
    )]
    Update {
        /// Only check for updates, don't install
        #[arg(long)]
        check: bool,

        /// Reinstall even if the current version matches
        #[arg(long)]
        force: bool,

        /// Install the latest rolling build from main
        #[arg(long, conflicts_with = "version")]
        nightly: bool,

        /// Install a specific version tag (e.g. v0.7.0)
        #[arg(long)]
        version: Option<String>,
    },

    /// Start the Discord bot (connects to a running mold server via MOLD_HOST)
    #[cfg(feature = "discord")]
    Discord,

    /// Launch the interactive terminal UI
    ///
    /// Full-featured TUI for image generation with live preview,
    /// model management, and gallery browsing.
    #[cfg(feature = "tui")]
    Tui {
        /// Server URL override
        #[arg(long, env = "MOLD_HOST")]
        host: Option<String>,

        /// Force local inference (no server connection)
        #[arg(long)]
        local: bool,
    },

    /// Generate shell completions (sources dynamic model-name completion)
    #[command(after_long_help = "\
Setup instructions:

  zsh (add to ~/.zshrc):
    source <(mold completions zsh)

  bash (add to ~/.bashrc):
    source <(mold completions bash)

  fish (persist to completions dir):
    mold completions fish | source
    mold completions fish > ~/.config/fish/completions/mold.fish

  elvish:
    eval (mold completions elvish | slurp)

  powershell (add to $PROFILE):
    mold completions powershell | Out-String | Invoke-Expression")]
    /// Upscale an image using a super-resolution model (Real-ESRGAN)
    ///
    /// Supports standalone upscaling of existing images and piped I/O.
    #[command(after_long_help = "\
Examples:
  mold upscale photo.png
  mold upscale photo.png -m real-esrgan-x4plus:fp16 -o photo_4x.png
  mold upscale - < input.png > output.png
  mold run \"a cat\" | mold upscale -")]
    Upscale {
        /// Input image file path (or - for stdin)
        image: String,

        /// Upscaler model name
        #[arg(short, long, add = ArgValueCandidates::new(commands::upscale::complete_upscaler_model))]
        model: Option<String>,

        /// Output file path (default: <input>_upscaled.<ext>)
        #[arg(short, long, value_hint = ValueHint::FilePath)]
        output: Option<String>,

        /// Output format
        #[arg(long, default_value_t = OutputFormat::Png,
              value_parser = output_format_parser(&["png", "jpeg", "jpg"]))]
        format: OutputFormat,

        /// Tile size for memory-efficient tiled inference (0 to disable)
        #[arg(long, env = "MOLD_UPSCALE_TILE_SIZE")]
        tile_size: Option<u32>,

        /// Server URL to connect to
        #[arg(long, env = "MOLD_HOST")]
        host: Option<String>,

        /// Skip server and run inference locally
        #[arg(long)]
        local: bool,

        /// Display upscaled image inline in the terminal after completion
        #[arg(long, env = "MOLD_PREVIEW")]
        preview: bool,
    },

    /// Durable per-frame Real-ESRGAN video upscale; temporal flicker may remain
    VideoUpscale {
        #[command(subcommand)]
        action: VideoUpscaleAction,
    },

    Completions {
        /// Shell to generate completions for (bash, zsh, fish, elvish, powershell)
        #[arg(add = ArgValueCandidates::new(complete_shell))]
        shell: String,
    },
}

fn complete_shell() -> Vec<CompletionCandidate> {
    ["bash", "zsh", "fish", "elvish", "powershell"]
        .into_iter()
        .map(CompletionCandidate::new)
        .collect()
}

/// Republish `--spatial-tile` as `MOLD_LTX2_SPATIAL_TILE`.
///
/// Local inference reads spatial tiling through the frozen runtime
/// environment, the same as every other memory knob, so a flag has to land
/// there rather than travel down the generate call chain. Clap already reads
/// the variable for the flag's default, so writing back is a no-op when the
/// user set the variable instead of the flag.
fn apply_spatial_tile_override(value: Option<&str>) {
    let Some(value) = value else {
        return;
    };
    if std::env::var("MOLD_LTX2_SPATIAL_TILE").as_deref() == Ok(value) {
        return;
    }
    // SAFETY: called once from the command dispatch before any inference
    // thread exists, matching how `serve` publishes `MOLD_HOST`.
    unsafe { std::env::set_var("MOLD_LTX2_SPATIAL_TILE", value) };
}

#[tokio::main]
async fn main() {
    // Keep positive compile-time exclusion provenance in every published CLI
    // artifact even after LTO and stripping. Release verification rejects a
    // missing marker as well as either compiled FlashAttention feature.
    std::hint::black_box(mold_inference::h3_attention_release_provenance_marker());

    // Install a panic hook that prints a friendly crash report with a link
    // to file an issue.  This only fires on Rust panics — segfaults from
    // FFI/CUDA are OS signals and bypass this hook entirely.
    std::panic::set_hook(Box::new(|info| {
        // Clear any in-progress line (progress bars, spinners)
        eprint!("\r\x1b[2K");
        eprintln!("\n{} mold crashed unexpectedly", theme::prefix_error());
        eprintln!();
        if let Some(msg) = info.payload().downcast_ref::<&str>() {
            eprintln!("  {msg}");
        } else if let Some(msg) = info.payload().downcast_ref::<String>() {
            eprintln!("  {msg}");
        }
        if let Some(loc) = info.location() {
            eprintln!("  at {}:{}:{}", loc.file(), loc.line(), loc.column());
        }
        eprintln!();
        eprintln!("  This is a bug. Please report it at:");
        eprintln!("  https://github.com/utensils/mold/issues");
        eprintln!();
        eprintln!("  Include the full output above and your 'mold version'.");
    }));

    // Reset SIGPIPE to default (terminate) so piping doesn't panic.
    // Rust ignores SIGPIPE by default, causing "broken pipe" panics when
    // stdout is a pipe and the reader closes (e.g. `mold run ... | head`).
    // NOTE: this disposition is wrong for long-running servers — a client
    // dropping mid-write would kill the process via SIGPIPE. `mold serve`
    // re-arms SIG_IGN in `mold_server::run_server()` (see issue #342).
    #[cfg(unix)]
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
    }

    // Handle Ctrl+C gracefully — exit immediately without letting background
    // threads (e.g. indicatif's ctrl-c cleanup thread) panic on RecvError.
    ctrlc::set_handler(move || {
        // Clear the line to remove any progress bar artifacts
        eprint!("\r\x1b[2K");
        std::process::exit(130); // 128 + SIGINT(2), standard Unix convention
    })
    .ok();

    if let Err(e) = run().await {
        // If the command already printed its own diagnostics, just exit.
        if e.downcast_ref::<AlreadyReported>().is_some() {
            std::process::exit(1);
        }

        let msg = format!("{e}");

        // Strip candle backtrace frames (numbered lines referencing candle/tokio internals).
        // Candle's Error::bt() embeds frame numbers in the Display output.
        let short = msg
            .lines()
            .take_while(|line| {
                let t = line.trim_start();
                !(t.len() > 2
                    && t.as_bytes()[0].is_ascii_digit()
                    && (t.contains("candle") || t.contains("tokio") || t.contains("at /")))
            })
            .collect::<Vec<_>>()
            .join("\n");
        let display = if short.is_empty() { &msg } else { &short };

        // Detect CUDA/Metal OOM and print a friendly message with suggestions.
        // Note: candle wraps Metal allocation failures as CUDA_ERROR_OUT_OF_MEMORY,
        // and Metal buffer creation failures as "Failed to create metal resource".
        // If the error came from a remote server, we must not label it with the
        // *client* platform's backend — see errors::RemoteInferenceError.
        if errors::is_oom_message(&msg) {
            let remote = e.downcast_ref::<errors::RemoteInferenceError>();
            let ctx = match remote {
                Some(r) => errors::OomContext::Remote { host: &r.host },
                None => errors::OomContext::Local,
            };
            let (label, hints) =
                errors::format_oom_message(&msg, ctx, cfg!(target_os = "macos"), display);
            eprintln!("{} {label}", theme::prefix_error());
            eprintln!();
            for line in &hints {
                if line.is_empty() {
                    eprintln!();
                } else {
                    eprintln!("  {line}");
                }
            }
            std::process::exit(1);
        }

        // Detect missing tensor errors (incompatible GGUF quantization format).
        if msg.contains("cannot find tensor") {
            eprintln!("{} {display}", theme::prefix_error());
            eprintln!();
            eprintln!("  The model file may be corrupted or uses an incompatible format.");
            eprintln!("  Try re-downloading: mold rm <model> && mold pull <model>");
            eprintln!("  Or try a different variant: mold list");
            std::process::exit(1);
        }

        // For all other errors, print the stripped message (no candle backtraces).
        // If the error is wrapped in `RemoteInferenceError`, iterate the inner
        // error's chain: our wrapper's `Display` forwards to `inner`, so using
        // `e.chain()` would print the same message twice — once as the top-level
        // `error:` line and again as the first `cause:` entry.
        let chain_source = e
            .downcast_ref::<errors::RemoteInferenceError>()
            .map(|r| &r.inner);
        eprintln!("{} {display}", theme::prefix_error());
        let chain = match chain_source {
            Some(inner) => inner.chain().skip(1),
            None => e.chain().skip(1),
        };
        for cause in chain {
            eprintln!("  {} {cause}", theme::prefix_cause());
        }
        std::process::exit(1);
    }
}

async fn run() -> anyhow::Result<()> {
    // Parse CLI first so we can set the log level based on the subcommand.
    clap_complete::CompleteEnv::with_factory(Cli::command).complete();
    let cli = Cli::parse();

    // Skill management is self-contained and must keep working even when a
    // configured external MOLD_HOME is offline or the inference stack cannot
    // initialize. It only reads/writes the explicitly selected skill paths.
    if let Commands::Skill(args) = &cli.command {
        return skill::run(args);
    }
    // Privileged machine administration must never open user Config/DB paths.
    if let Commands::System { action } = &cli.command {
        return commands::system::run(action);
    }

    // A missing saved root means its external drive is offline. Fail before
    // the DB, logger, model cache, or output paths can recreate that mount.
    mold_core::Config::ensure_saved_mold_dir_available()?;

    // Install the DB-backed `Config` overlay hook: first load runs the
    // one-shot config.toml → DB migration, every subsequent load picks
    // up authoritative user-preference values from the DB. Safe to run
    // before logging init because the hook no-ops when the DB is
    // disabled or unavailable.
    metadata_db::install_config_db_hooks();

    // Initialize tracing. `mold serve` uses the logging module for optional
    // file output; all other commands use stderr-only with warn level.
    let _log_guard = match &cli.command {
        Commands::Serve {
            log_format,
            log_file,
            ..
        } => {
            let config = mold_core::Config::load_or_default();
            let log_dir = config.resolved_log_dir();
            Some(mold_server::logging::init_tracing(
                *log_file,
                matches!(log_format, LogFormat::Json),
                &config.logging,
                "info",
                log_dir,
            ))
        }
        #[cfg(feature = "tui")]
        Commands::Tui { .. } => {
            // TUI owns the terminal — file-only logging, no stderr output.
            let config = mold_core::Config::load_or_default();
            let log_dir = config.resolved_log_dir();
            Some(mold_server::logging::init_tracing_file_only(
                &config.logging,
                "warn",
                log_dir,
            ))
        }
        _ => {
            let filter = tracing_subscriber::EnvFilter::try_from_env("MOLD_LOG")
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn"));
            tracing_subscriber::fmt()
                .with_env_filter(filter)
                .with_writer(std::io::stderr)
                .init();
            None
        }
    };

    match cli.command {
        Commands::Run {
            model_or_prompt,
            prompt_rest,
            output,
            width,
            height,
            steps,
            guidance,
            seed,
            batch,
            mesh,
            frames,
            predict_duration,
            fps,
            duration,
            clip_frames,
            motion_tail,
            audio,
            no_audio,
            video_only,
            audio_file,
            video,
            extend,
            extend_overlap,
            keyframe,
            last_image,
            first_frame,
            last_frame,
            reference,
            pipeline,
            ic_lora_control,
            hdr_exr_dir,
            hdr_exr_full_float,
            retake,
            spatial_upscale,
            temporal_upscale,
            stg_scale,
            stg_blocks,
            rescale_scale,
            modality_scale,
            guidance_skip_step,
            sample_solver,
            sample_shift,
            distill_strength,
            camera_control,
            host,
            format,
            no_metadata,
            title,
            tags,
            collection,
            no_auto_tag,
            preview,
            local,
            prompt,
            frames_per_clip,
            script,
            dry_run,
            gpus,
            t5_variant,
            qwen3_variant,
            qwen2_variant,
            qwen2_text_encoder_mode,
            scheduler,
            cfg_plus,
            eager,
            offload,
            spatial_tile,
            device_text_encoders,
            device_transformer,
            device_vae,
            device_t5,
            device_clip_l,
            device_clip_g,
            device_qwen,
            lora,
            lora_scale,
            image,
            strength,
            mask,
            id_image,
            id_weight,
            id_start_step,
            true_cfg,
            cfg_start_step,
            control,
            control_model,
            control_scale,
            negative_prompt,
            no_negative,
            expand,
            no_expand,
            expand_backend,
            expand_model,
            upscale: _upscale, // TODO: wire into generate pipeline for post-generation upscaling
        } => {
            apply_spatial_tile_override(spatial_tile.as_deref());

            // A chain request has no title slot yet; refuse rather than
            // silently dropping the flag.
            if title.is_some() && (script.is_some() || prompt.len() > 1) {
                anyhow::bail!(
                    "--title applies to single-clip runs; chain scripts and multi-prompt sequences do not carry a title yet"
                );
            }

            if let Some(ref path) = script {
                return commands::chain::run_from_script(
                    path,
                    host.clone(),
                    output.clone(),
                    local,
                    dry_run,
                    no_metadata,
                    preview,
                    gpus.clone(),
                    t5_variant.clone(),
                    qwen3_variant.clone(),
                    qwen2_variant.clone(),
                    qwen2_text_encoder_mode.clone(),
                    eager,
                    offload,
                )
                .await;
            }

            // Reject positional prompt + --prompt flag combo (Task 3.6).
            if !prompt.is_empty() {
                let config = mold_core::Config::load_or_default();
                let has_positional_prompt = match model_or_prompt.as_deref() {
                    Some(first) if mold_core::manifest::is_known_model(first, &config) => {
                        !prompt_rest.is_empty()
                    }
                    Some(_) => true, // first positional isn't a model → it's prompt text
                    None => !prompt_rest.is_empty(),
                };
                if has_positional_prompt {
                    anyhow::bail!(
                        "cannot combine positional prompt and --prompt; pick one or use --script"
                    );
                }
            }

            if prompt.len() > 1
                && (model_or_prompt
                    .as_deref()
                    .and_then(mold_core::minimax_h3::resolve_model_name)
                    .is_some()
                    || duration.is_some()
                    || first_frame.is_some()
                    || last_frame.is_some()
                    || !reference.is_empty())
            {
                anyhow::bail!(
                    "MiniMax H3 endpoint/reference authoring is single-clip; use one prompt"
                );
            }
            if prompt.len() > 1 {
                return commands::chain::run_from_sugar(
                    model_or_prompt.clone(),
                    prompt.clone(),
                    frames_per_clip,
                    motion_tail,
                    if audio {
                        Some(true)
                    } else if no_audio {
                        Some(false)
                    } else {
                        None
                    },
                    dry_run,
                    host.clone(),
                    output.clone(),
                    local,
                    no_metadata,
                    preview,
                    gpus.clone(),
                    t5_variant.clone(),
                    qwen3_variant.clone(),
                    qwen2_variant.clone(),
                    qwen2_text_encoder_mode.clone(),
                    eager,
                    offload,
                )
                .await;
            }
            // Fold single --prompt into prompt_rest when there's no positional prompt.
            let prompt_rest = if prompt.len() == 1 && prompt_rest.is_empty() {
                prompt.into_iter().collect()
            } else {
                prompt_rest
            };
            commands::run::run(
                model_or_prompt,
                prompt_rest,
                output,
                width,
                height,
                steps,
                guidance,
                seed,
                batch,
                frames,
                predict_duration,
                fps,
                duration,
                clip_frames,
                motion_tail,
                audio,
                no_audio,
                video_only,
                audio_file,
                video,
                extend,
                extend_overlap,
                keyframe,
                last_image,
                first_frame,
                last_frame,
                reference,
                pipeline,
                ic_lora_control,
                hdr_exr_dir,
                hdr_exr_full_float,
                retake,
                spatial_upscale,
                temporal_upscale,
                commands::run::GuidanceFlags {
                    stg_scale,
                    stg_blocks,
                    rescale_scale,
                    modality_scale,
                    skip_step: guidance_skip_step,
                },
                mesh.into_flags(),
                commands::run::WanFlags {
                    sample_solver,
                    sample_shift,
                    distill_strength,
                },
                camera_control,
                host,
                format,
                no_metadata,
                title,
                commands::generate::FilingOptions {
                    tags,
                    collection,
                    no_auto_tag,
                },
                preview,
                local,
                gpus,
                t5_variant,
                qwen3_variant,
                qwen2_variant,
                qwen2_text_encoder_mode,
                scheduler,
                cfg_plus,
                eager,
                offload,
                commands::run::PlacementFlags {
                    text_encoders: device_text_encoders,
                    transformer: device_transformer,
                    vae: device_vae,
                    t5: device_t5,
                    clip_l: device_clip_l,
                    clip_g: device_clip_g,
                    qwen: device_qwen,
                },
                lora,
                lora_scale,
                image,
                strength,
                mask,
                commands::identity::IdentityArgs {
                    id_images: id_image,
                    id_weight,
                    id_start_step,
                    true_cfg,
                    cfg_start_step,
                },
                control,
                control_model,
                control_scale,
                negative_prompt,
                no_negative,
                expand,
                no_expand,
                expand_backend,
                expand_model,
            )
            .await?;
        }
        Commands::Expand {
            prompt,
            model,
            variations,
            json,
            backend,
            expand_model,
            task,
            width,
            height,
            frames,
            fps,
            clip_frames,
            reference,
        } => {
            let context = commands::expand::context_from_flags(
                model.as_deref(),
                width,
                height,
                frames,
                fps,
                clip_frames,
                &reference,
            )?;
            commands::expand::run(
                &prompt,
                model.as_deref(),
                variations,
                json,
                backend.as_deref(),
                expand_model.as_deref(),
                task.as_deref(),
                context,
            )
            .await?;
        }
        Commands::Remix {
            source_prompt,
            model,
            variations,
            json,
            backend,
            expand_model,
            task,
            source,
            root_prompt,
            dimensions,
            style,
            width,
            height,
            frames,
            fps,
            clip_frames,
            reference,
        } => {
            let context = commands::expand::context_from_flags(
                model.as_deref(),
                width,
                height,
                frames,
                fps,
                clip_frames,
                &reference,
            )?;
            commands::remix::run(
                &source_prompt,
                model.as_deref(),
                variations,
                json,
                backend.as_deref(),
                expand_model.as_deref(),
                task.as_deref(),
                &source,
                root_prompt.as_deref(),
                &dimensions,
                style.as_deref(),
                context,
            )
            .await?;
        }
        Commands::Serve {
            port,
            bind,
            models_dir,
            gpus,
            queue_size,
            #[cfg(feature = "discord")]
            discord,
            #[cfg(feature = "mdns")]
            no_mdns,
            ..
        } => {
            #[cfg(feature = "discord")]
            let discord_enabled = discord;
            #[cfg(not(feature = "discord"))]
            let discord_enabled = false;

            // Opt out of advertising before the server reads MOLD_MDNS.
            #[cfg(feature = "mdns")]
            if no_mdns {
                // SAFETY: set on the main thread before run_server spawns tasks.
                unsafe { std::env::set_var("MOLD_MDNS", "0") };
            }

            commands::serve::run(port, &bind, models_dir, gpus, queue_size, discord_enabled)
                .await?;
        }
        Commands::Mcp { host } => {
            commands::mcp::run(host).await?;
        }
        Commands::Server { action } => match action {
            ServerAction::Start {
                port,
                bind,
                models_dir,
                log_file,
                #[cfg(feature = "mdns")]
                no_mdns,
            } => {
                commands::server::run_start(
                    port,
                    &bind,
                    models_dir,
                    log_file,
                    #[cfg(feature = "mdns")]
                    no_mdns,
                )
                .await?;
            }
            ServerAction::Status { host } => {
                commands::server::run_status(host).await?;
            }
            ServerAction::Stop => {
                commands::server::run_stop().await?;
            }
            #[cfg(feature = "mdns")]
            ServerAction::Discover {
                timeout_secs,
                json,
                probe,
            } => {
                commands::server::run_discover(timeout_secs, json, probe).await?;
            }
        },
        Commands::Chain { action } => match action {
            ChainSub::Validate { path } => commands::chain_validate::run(&path).await?,
        },
        Commands::Jobs { action } => {
            let config = mold_core::Config::load_or_default();
            commands::jobs::run(action, &config).await?;
        }
        Commands::Queue { action } => {
            commands::queue::run(action).await?;
        }
        Commands::Library { action } => {
            commands::library::run(action).await?;
        }
        Commands::Trash { action } => {
            commands::trash::run(action).await?;
        }
        Commands::Pull {
            model,
            skip_verify,
            accept_license,
        } => {
            // Passed through rather than recorded here: `pull::run` decides
            // whether the pull lands locally or on `MOLD_HOST`, and the
            // acceptance has to be written on whichever machine does the
            // downloading.
            let accept_licenses: Vec<String> = accept_license;
            let opts = mold_core::download::PullOptions { skip_verify };
            if model.starts_with("hf:") || model.starts_with("cv:") {
                match resolve_catalog_id(&model).await? {
                    CatalogIdResolution::Manifest(name) => {
                        commands::pull::run(&name, &opts, &accept_licenses).await?
                    }
                    CatalogIdResolution::Recipe(entry) => {
                        commands::pull::run_recipe(*entry, &opts).await?
                    }
                }
            } else {
                commands::pull::run(&model, &opts, &accept_licenses).await?;
            }
        }
        Commands::Licenses { action, local } => match action {
            Some(LicensesAction::Accept {
                ids,
                local: accept_local,
            }) => {
                // `--local` is accepted on either side of the subcommand
                // (`mold licenses --local accept X` and
                // `mold licenses accept X --local` read identically to a
                // user), and clap stores each in its own field. Honour
                // whichever was given: silently ignoring the parent flag
                // would record consent on MOLD_HOST after the user asked
                // for this machine.
                commands::licenses::accept(&ids, local || accept_local).await?;
            }
            None => {
                commands::licenses::run(local).await?;
            }
        },
        Commands::Rm { models, force } => {
            commands::rm::run(&models, force).await?;
        }
        Commands::List => {
            commands::list::run().await?;
        }
        Commands::Stats { json } => {
            commands::stats::run(json)?;
        }
        Commands::Clean { force, older_than } => {
            commands::clean::run(force, older_than.as_deref())?;
        }
        Commands::Info { model, verify } => {
            if let Some(model) = model {
                commands::info::run(&model, verify)?;
            } else {
                if verify {
                    eprintln!("{} --verify requires a model name", theme::prefix_error());
                    return Err(AlreadyReported.into());
                }
                commands::info::run_overview().await?;
            }
        }
        Commands::Default { model } => {
            commands::default::run(model.as_deref())?;
        }
        Commands::Config { action, profile } => {
            // Set `MOLD_PROFILE` for the duration of this command so every
            // `Settings::new(db)` / `ModelPrefs::load(db, …)` picks up the
            // flag automatically via `resolve_active_profile`.
            if let Some(ref p) = profile {
                std::env::set_var("MOLD_PROFILE", p);
            }
            match action {
                ConfigAction::List { json } => commands::config::run_list(json)?,
                ConfigAction::Get { key, raw } => commands::config::run_get(&key, raw)?,
                ConfigAction::Set { key, value } => commands::config::run_set(&key, &value)?,
                ConfigAction::Path => commands::config::run_path()?,
                ConfigAction::Edit => commands::config::run_edit()?,
                ConfigAction::Where { key } => commands::config::run_where(&key)?,
                ConfigAction::Reset { key, all, yes } => {
                    commands::config::run_reset(key.as_deref(), all, yes)?
                }
            }
        }
        Commands::Runpod { action } => match action {
            RunpodAction::Doctor => commands::runpod::run_doctor().await?,
            RunpodAction::Gpus { all, json } => commands::runpod::run_gpus(json, all).await?,
            RunpodAction::Datacenters { gpu, json } => {
                commands::runpod::run_datacenters(gpu, json).await?
            }
            RunpodAction::NetworkVolume { action } => match action {
                RunpodNetworkVolumeAction::List { json } => {
                    commands::runpod::run_network_volume_list(json).await?
                }
                RunpodNetworkVolumeAction::Get { volume_id, json } => {
                    commands::runpod::run_network_volume_get(volume_id, json).await?
                }
                RunpodNetworkVolumeAction::Create {
                    name,
                    size,
                    datacenter,
                    json,
                } => {
                    commands::runpod::run_network_volume_create(name, size, datacenter, json)
                        .await?
                }
                RunpodNetworkVolumeAction::Update {
                    volume_id,
                    name,
                    size,
                    json,
                } => {
                    commands::runpod::run_network_volume_update(volume_id, name, size, json).await?
                }
                RunpodNetworkVolumeAction::Delete { volume_id, json } => {
                    commands::runpod::run_network_volume_delete(volume_id, json).await?
                }
            },
            RunpodAction::List { json } => commands::runpod::run_list(json).await?,
            RunpodAction::Get { pod_id, json } => commands::runpod::run_get(pod_id, json).await?,
            RunpodAction::Create {
                name,
                gpu,
                datacenter,
                cloud,
                disk,
                volume,
                image_tag,
                model,
                hf_token,
                network_volume,
                dry_run,
                json,
            } => {
                use std::str::FromStr;
                let cloud_type = commands::runpod::CloudType::from_str(&cloud)?;
                let opts = commands::runpod::CreateOptions {
                    name,
                    gpu,
                    datacenter,
                    cloud: cloud_type,
                    volume_gb: volume,
                    disk_gb: disk,
                    image_tag,
                    model,
                    hf_token,
                    network_volume_id: network_volume,
                    dry_run,
                    json,
                };
                commands::runpod::run_create(opts).await?
            }
            RunpodAction::Stop { pod_id, json } => commands::runpod::run_stop(pod_id, json).await?,
            RunpodAction::Start { pod_id, json } => {
                commands::runpod::run_start(pod_id, json).await?
            }
            RunpodAction::Delete {
                pod_id,
                force,
                json,
            } => commands::runpod::run_delete(pod_id, force, json).await?,
            RunpodAction::Connect { pod_id, check } => {
                commands::runpod::run_connect(pod_id, check).await?
            }
            RunpodAction::Logs { pod_id, follow } => {
                commands::runpod::run_logs(pod_id, follow).await?
            }
            RunpodAction::Usage { since, json } => commands::runpod::run_usage(since, json).await?,
            RunpodAction::Run {
                prompt,
                model,
                output_dir,
                keep,
                seed,
                steps,
                width,
                height,
                gpu,
                datacenter,
                network_volume,
                wait_timeout,
                hf_token,
            } => {
                let create = commands::runpod::CreateOptions {
                    name: None,
                    gpu,
                    datacenter,
                    cloud: commands::runpod::CloudType::Secure,
                    volume_gb: 50,
                    disk_gb: 20,
                    image_tag: None,
                    model: model.clone(),
                    hf_token,
                    network_volume_id: network_volume,
                    dry_run: false,
                    json: false,
                };
                let opts = commands::runpod::RunOptions {
                    prompt,
                    model,
                    output_dir,
                    keep,
                    seed,
                    steps,
                    width,
                    height,
                    create,
                    wait_ready_timeout_secs: wait_timeout,
                };
                commands::runpod::run_run(opts).await?
            }
        },
        Commands::Lambda { action } => match action {
            LambdaAction::Doctor => commands::lambda::run_doctor().await?,
            LambdaAction::Availability { json } => commands::lambda::run_availability(json).await?,
            LambdaAction::Deploy {
                instance_type,
                region,
                new,
                dry_run,
                json,
                forward_secrets,
                model,
                open_browser,
            } => {
                commands::lambda::run_deploy(commands::lambda::DeployOptions {
                    instance_type,
                    region,
                    new,
                    dry_run,
                    json,
                    forward_secrets,
                    model,
                    open_browser,
                })
                .await?
            }
            LambdaAction::Status { json } => commands::lambda::run_status(json).await?,
            LambdaAction::Logs { follow } => commands::lambda::run_logs(follow).await?,
            LambdaAction::Tunnel { local_port } => commands::lambda::run_tunnel(local_port).await?,
            LambdaAction::Ssh => commands::lambda::run_ssh().await?,
            LambdaAction::Filesystems { json } => commands::lambda::run_filesystems(json).await?,
            LambdaAction::Terminate { json } => commands::lambda::run_terminate(json).await?,
            LambdaAction::Reset {
                to_zero,
                confirm,
                json,
            } => {
                if !to_zero {
                    anyhow::bail!("use `mold lambda reset --to-zero` to confirm the reset scope");
                }
                commands::lambda::run_reset_to_zero(confirm, json).await?
            }
        },
        Commands::Unload => {
            commands::unload::run().await?;
        }
        Commands::Ps => {
            commands::ps::run().await?;
        }
        Commands::Gpu { action } => match action {
            GpuAction::List { json } => commands::gpu::list(json).await?,
            GpuAction::Disable { device } => commands::gpu::set(&device, false).await?,
            GpuAction::Enable { device } => commands::gpu::set(&device, true).await?,
        },
        Commands::Version => {
            println!("mold {}", mold_core::build_info::version_string());
        }
        Commands::Update {
            check,
            force,
            nightly,
            version,
        } => {
            commands::update::run(check, force, nightly, version).await?;
        }
        #[cfg(feature = "discord")]
        Commands::Discord => {
            commands::discord::run().await?;
        }
        #[cfg(feature = "tui")]
        Commands::Tui { host, local } => {
            mold_tui::run_tui(host, local).await?;
        }
        Commands::Upscale {
            image,
            model,
            output,
            format,
            tile_size,
            host,
            local,
            preview,
        } => {
            commands::upscale::run(
                image, model, output, format, tile_size, host, local, preview,
            )
            .await?;
        }
        Commands::VideoUpscale { action } => match action {
            VideoUpscaleAction::Create {
                source,
                model,
                tile_size,
                host,
                wait,
            } => commands::video_upscale::create(source, model, tile_size, host, wait).await?,
            VideoUpscaleAction::List { host } => commands::video_upscale::list(host).await?,
            VideoUpscaleAction::Status { id, host } => {
                commands::video_upscale::status(id, host).await?
            }
            VideoUpscaleAction::Pause { id, host } => {
                commands::video_upscale::transition(id, "pause", host).await?
            }
            VideoUpscaleAction::Resume { id, host } => {
                commands::video_upscale::transition(id, "resume", host).await?
            }
            VideoUpscaleAction::Cancel { id, host } => {
                commands::video_upscale::transition(id, "cancel", host).await?
            }
        },
        Commands::Completions { shell } => {
            generate_completions(&shell)?;
        }
        Commands::Skill(_) => unreachable!("skill commands return before runtime initialization"),
        Commands::System { .. } => {
            unreachable!("system commands return before runtime initialization")
        }
    }

    Ok(())
}

/// Generate shell completion script.
///
/// For zsh: custom script that separates flags from positional candidates so
/// `mold run <TAB>` shows only model names, while `mold run --<TAB>` shows flags.
/// For other shells: delegates to clap_complete's dynamic registration.
fn generate_completions(shell: &str) -> anyhow::Result<()> {
    if shell == "zsh" {
        let bin = std::env::args()
            .next()
            .unwrap_or_else(|| "mold".to_string());
        print!(
            r##"#compdef mold
function _clap_dynamic_completer_mold() {{
    local _CLAP_COMPLETE_INDEX=$(expr $CURRENT - 1)
    local _CLAP_IFS=$'\n'

    # File-path flags: fall back to zsh native _files for tilde expansion,
    # directory traversal, and proper path completion.
    local prev_word="${{words[$(( CURRENT - 1 ))]}}"
    case "$prev_word" in
        --lora|--image|-i|--mask|--control|--output|-o)
            _files
            return
            ;;
        --control-model|--models-dir)
            _files -/
            return
            ;;
    esac

    local completions=("${{(@f)$( \
        _CLAP_IFS="$_CLAP_IFS" \
        _CLAP_COMPLETE_INDEX="$_CLAP_COMPLETE_INDEX" \
        COMPLETE="zsh" \
        {bin} -- "${{words[@]}}" 2>/dev/null \
    )}}")

    if [[ -n $completions ]]; then
        local -a flags=()
        local -a values=()
        local completion
        for completion in $completions; do
            local value="${{completion%%:*}}"
            if [[ "$value" == -* ]]; then
                flags+=("$completion")
            elif [[ "$value" == */ ]]; then
                local dir_no_slash="${{value%/}}"
                if [[ "$completion" == *:* ]]; then
                    local desc="${{completion#*:}}"
                    values+=("$dir_no_slash:$desc")
                else
                    values+=("$dir_no_slash")
                fi
            else
                values+=("$completion")
            fi
        done

        if [[ "${{words[$CURRENT]}}" == -* ]]; then
            [[ -n $flags ]] && _describe 'options' flags
        else
            [[ -n $values ]] && _describe 'values' values
        fi
    fi
}}

compdef _clap_dynamic_completer_mold mold
"##,
            bin = bin,
        );
        return Ok(());
    }

    let shells = clap_complete::env::Shells::builtins();
    let completer = match shells.completer(shell) {
        Some(c) => c,
        None => {
            let names: Vec<_> = shells.names().collect();
            anyhow::bail!(
                "unknown shell '{}', expected one of: {}",
                shell,
                names.join(", ")
            );
        }
    };
    let bin = std::env::args()
        .next()
        .unwrap_or_else(|| "mold".to_string());
    completer.write_registration("COMPLETE", "mold", "mold", &bin, &mut std::io::stdout())?;
    Ok(())
}

/// Result of resolving a `hf:` / `cv:` catalog id. HF entries map onto
/// an existing manifest model name (the manifest path takes it from
/// there); Civitai single-file checkpoints carry a recipe the CLI
/// fetches directly. `Recipe` boxes the entry because `CatalogEntry`
/// is several hundred bytes.
pub enum CatalogIdResolution {
    /// HF entry — pull through the existing manifest path with this name.
    Manifest(String),
    /// Civitai entry — pull companions then fetch each `download_recipe`
    /// file directly into `MOLD_MODELS_DIR/<sanitized-id>/`.
    Recipe(Box<mold_catalog::entry::CatalogEntry>),
}

/// Resolve a catalog ID via live HF/Civitai lookup. HF entries return
/// `Manifest(source_id)` (the existing manifest path knows what to do
/// with the repo id); Civitai entries return `Recipe(entry)` so the
/// caller can drive the recipe-fetcher with companion-first ordering.
pub async fn resolve_catalog_id(id: &str) -> anyhow::Result<CatalogIdResolution> {
    mold_core::require_model_activation(id, None)?;
    if let Some(repo_id) = id.strip_prefix("hf:") {
        // HF: defer to the manifest path. We don't need a live fetch to
        // know the repo id — that's what the user typed. The manifest
        // registry is the authority on which HF repos this build can
        // pull.
        return Ok(CatalogIdResolution::Manifest(repo_id.to_string()));
    }
    let entry = catalog_bridge::lookup_catalog_entry_live(id).await?;
    mold_core::require_model_activation(entry.id.as_str(), Some(entry.family.as_str()))?;
    mold_core::require_model_activation(&entry.name, Some(entry.family.as_str()))?;
    if !entry.supported {
        if mold_catalog::wan_a14b::entry_is_unpaired_a14b(&entry) {
            anyhow::bail!("{}", mold_catalog::wan_a14b::unpaired_reason(&entry.name));
        }
        anyhow::bail!("catalog entry is not supported by this build of mold");
    }
    Ok(CatalogIdResolution::Recipe(Box::new(entry)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::ENV_LOCK;
    use clap::Parser;

    /// Parse CLI args from a vector (simulates command-line invocation).
    fn parse(args: &[&str]) -> Cli {
        try_parse(args).unwrap_or_else(|error| panic!("{error}"))
    }

    /// Try to parse CLI args, returning the clap error on failure. The full
    /// clap tree no longer fits the default 2 MiB test-thread stack in a
    /// debug build, so parsing runs on a thread sized like the real `main`.
    fn try_parse(args: &[&str]) -> Result<Cli, clap::Error> {
        let argv = std::iter::once("mold".to_string())
            .chain(args.iter().map(|arg| (*arg).to_string()))
            .collect::<Vec<_>>();
        std::thread::Builder::new()
            .stack_size(64 << 20)
            .spawn(move || Cli::try_parse_from(argv))
            .expect("spawn parse thread")
            .join()
            .expect("parse thread panicked")
    }

    /// `--video-only` skips the audio branch, so asking for audio beside it
    /// — output or conditioning — is a contradiction clap refuses before a
    /// request exists.
    #[test]
    fn video_only_conflicts_with_audio_flags() {
        assert!(try_parse(&["run", "m", "p", "--video-only", "--audio"]).is_err());
        assert!(try_parse(&["run", "m", "p", "--video-only", "--audio-file", "a.wav"]).is_err());
        assert!(try_parse(&["run", "m", "p", "--video-only", "--no-audio"]).is_ok());
        assert!(try_parse(&["run", "m", "p", "--video-only"]).is_ok());
    }

    #[test]
    fn skill_install_requires_an_explicit_target_mode() {
        assert!(try_parse(&["skill", "install"]).is_err());
    }

    #[test]
    fn skill_install_accepts_agents_and_project_scope() {
        let cli = parse(&["skill", "install", "claude", "codex", "--project"]);
        assert!(matches!(cli.command, Commands::Skill(_)));
    }

    #[test]
    fn skill_install_target_modes_conflict() {
        assert!(try_parse(&["skill", "install", "codex", "--all"]).is_err());
        assert!(try_parse(&["skill", "install", "--detected", "--all"]).is_err());
    }

    #[test]
    fn skill_show_and_uninstall_parse() {
        assert!(matches!(
            parse(&["skill", "show"]).command,
            Commands::Skill(_)
        ));
        assert!(matches!(
            parse(&["skill", "uninstall", "--dir", "/tmp/project"]).command,
            Commands::Skill(_)
        ));
    }

    #[tokio::test]
    async fn raw_h3_catalog_id_remains_outside_the_pinned_manifest_path() {
        let error = resolve_catalog_id("hf:Comfy-Org/MiniMax-H3")
            .await
            .err()
            .expect("raw repositories must not bypass the pinned compact manifests");
        assert!(error
            .to_string()
            .contains(mold_core::MINIMAX_H3_AUTHORIZATION_REQUIRED));
    }

    #[test]
    fn run_parses_model_and_prompt() {
        let cli = parse(&["run", "flux-dev:q4", "a", "red", "apple"]);
        match cli.command {
            Commands::Run {
                model_or_prompt,
                prompt_rest,
                ..
            } => {
                assert_eq!(model_or_prompt.as_deref(), Some("flux-dev:q4"));
                assert_eq!(prompt_rest, vec!["a", "red", "apple"]);
            }
            _ => panic!("expected Run command"),
        }
    }

    #[test]
    fn serve_reads_explicit_none_from_mold_gpus() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let previous = std::env::var_os("MOLD_GPUS");
        std::env::set_var("MOLD_GPUS", "none");

        let cli = parse(&["serve"]);

        match previous {
            Some(value) => std::env::set_var("MOLD_GPUS", value),
            None => std::env::remove_var("MOLD_GPUS"),
        }
        match cli.command {
            Commands::Serve { gpus, .. } => assert_eq!(gpus.as_deref(), Some("none")),
            _ => panic!("expected Serve command"),
        }
    }

    #[test]
    fn run_seed_before_prompt() {
        let cli = parse(&["run", "model", "--seed", "42", "a", "cat"]);
        match cli.command {
            Commands::Run { seed, .. } => assert_eq!(seed, Some(42)),
            _ => panic!("expected Run"),
        }
    }

    /// The wan env fallbacks are deliberately NOT clap env bindings: a bound
    /// env would inject the wan-only fields into every family's request and
    /// admission would reject e.g. a FLUX run outright (codex review). The
    /// wan engine reads both vars itself, so they still reach wan renders.
    #[test]
    fn wan_env_fallbacks_do_not_populate_cli_flags() {
        let previous_solver = std::env::var("MOLD_WAN_SOLVER").ok();
        let previous_shift = std::env::var("MOLD_WAN_SHIFT").ok();
        std::env::set_var("MOLD_WAN_SOLVER", "euler");
        std::env::set_var("MOLD_WAN_SHIFT", "12.0");
        let cli = parse(&["run", "flux-schnell", "a", "cat"]);
        match previous_solver {
            Some(value) => std::env::set_var("MOLD_WAN_SOLVER", value),
            None => std::env::remove_var("MOLD_WAN_SOLVER"),
        }
        match previous_shift {
            Some(value) => std::env::set_var("MOLD_WAN_SHIFT", value),
            None => std::env::remove_var("MOLD_WAN_SHIFT"),
        }
        match cli.command {
            Commands::Run {
                sample_solver,
                sample_shift,
                ..
            } => {
                assert_eq!(sample_solver, None);
                assert_eq!(sample_shift, None);
            }
            _ => panic!("expected Run"),
        }
    }

    #[test]
    fn run_seed_after_prompt() {
        let cli = parse(&["run", "model", "a", "cat", "--seed", "42"]);
        match cli.command {
            Commands::Run { seed, .. } => assert_eq!(seed, Some(42)),
            _ => panic!("expected Run"),
        }
    }

    #[test]
    fn run_steps_after_prompt() {
        let cli = parse(&["run", "model", "a", "cat", "--steps", "20"]);
        match cli.command {
            Commands::Run { steps, .. } => assert_eq!(steps, Some(20)),
            _ => panic!("expected Run"),
        }
    }

    #[test]
    fn run_pipeline_uses_typed_value_enum() {
        let cli = parse(&[
            "run",
            "ltx-2-19b-distilled:fp8",
            "a clip",
            "--pipeline",
            "two-stage-hq",
        ]);
        match cli.command {
            Commands::Run { pipeline, .. } => {
                assert_eq!(pipeline, Some(Ltx2PipelineArg::TwoStageHq));
            }
            _ => panic!("expected Run"),
        }
    }

    #[test]
    fn run_pipeline_rejects_unknown_value_at_parse_time() {
        let err = match try_parse(&[
            "run",
            "ltx-2-19b-distilled:fp8",
            "a clip",
            "--pipeline",
            "unknown",
        ]) {
            Ok(_) => panic!("expected invalid pipeline value"),
            Err(err) => err,
        };
        let msg = err.to_string();
        assert!(msg.contains("invalid value 'unknown'"), "got: {msg}");
        assert!(msg.contains("two-stage-hq"), "got: {msg}");
    }

    #[test]
    fn run_width_height() {
        let cli = parse(&["run", "model", "--width", "512", "--height", "768", "test"]);
        match cli.command {
            Commands::Run { width, height, .. } => {
                assert_eq!(width, Some(512));
                assert_eq!(height, Some(768));
            }
            _ => panic!("expected Run"),
        }
    }

    #[test]
    fn run_guidance() {
        let cli = parse(&["run", "model", "test", "--guidance", "7.5"]);
        match cli.command {
            Commands::Run { guidance, .. } => assert_eq!(guidance, Some(7.5)),
            _ => panic!("expected Run"),
        }
    }

    #[test]
    fn run_audio_flags_conflict() {
        let err = try_parse(&["run", "ltx-2.3-22b-distilled:fp8", "--audio", "--no-audio"])
            .err()
            .expect("conflicting audio flags should fail");
        assert_eq!(err.kind(), clap::error::ErrorKind::ArgumentConflict);
    }

    #[test]
    fn run_qwen2_text_encoder_mode() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let cli = parse(&[
            "run",
            "qwen-image:q2",
            "test",
            "--qwen2-text-encoder-mode",
            "cpu-stage",
        ]);
        match cli.command {
            Commands::Run {
                qwen2_text_encoder_mode,
                ..
            } => assert_eq!(qwen2_text_encoder_mode.as_deref(), Some("cpu-stage")),
            _ => panic!("expected Run"),
        }
    }

    #[test]
    fn run_qwen2_variant() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let cli = parse(&["run", "qwen-image:q2", "test", "--qwen2-variant", "q6"]);
        match cli.command {
            Commands::Run { qwen2_variant, .. } => {
                assert_eq!(qwen2_variant.as_deref(), Some("q6"))
            }
            _ => panic!("expected Run"),
        }
    }

    #[test]
    fn run_format_jpeg() {
        let cli = parse(&["run", "model", "test", "--format", "jpeg"]);
        match cli.command {
            Commands::Run { format, .. } => assert_eq!(format, Some(OutputFormat::Jpeg)),
            _ => panic!("expected Run"),
        }
    }

    #[test]
    fn run_batch() {
        let cli = parse(&["run", "model", "test", "--batch", "4"]);
        match cli.command {
            Commands::Run { batch, .. } => assert_eq!(batch, 4),
            _ => panic!("expected Run"),
        }
    }

    #[test]
    fn run_output_flag() {
        let cli = parse(&["run", "model", "test", "-o", "/tmp/out.png"]);
        match cli.command {
            Commands::Run { output, .. } => assert_eq!(output.as_deref(), Some("/tmp/out.png")),
            _ => panic!("expected Run"),
        }
    }

    #[test]
    fn run_local_flag() {
        let cli = parse(&["run", "model", "test", "--local"]);
        match cli.command {
            Commands::Run { local, .. } => assert!(local),
            _ => panic!("expected Run"),
        }
    }

    #[test]
    fn run_eager_flag() {
        let cli = parse(&["run", "model", "test", "--eager"]);
        match cli.command {
            Commands::Run { eager, .. } => assert!(eager),
            _ => panic!("expected Run"),
        }
    }

    #[test]
    fn run_offload_flag() {
        let cli = parse(&["run", "model", "test", "--offload"]);
        match cli.command {
            Commands::Run { offload, .. } => assert!(offload),
            _ => panic!("expected Run"),
        }
    }

    #[test]
    fn run_spatial_tile_flag() {
        // Absent means auto, which the engine treats as "only past the span
        // the checkpoints were trained on".
        match parse(&["run", "model", "test"]).command {
            Commands::Run { spatial_tile, .. } => assert_eq!(spatial_tile, None),
            _ => panic!("expected Run"),
        }
        for value in ["off", "auto", "1280", "1280:256"] {
            match parse(&["run", "model", "test", "--spatial-tile", value]).command {
                Commands::Run { spatial_tile, .. } => {
                    assert_eq!(spatial_tile.as_deref(), Some(value))
                }
                _ => panic!("expected Run"),
            }
        }
    }

    #[test]
    fn run_all_flags_combined() {
        let cli = parse(&[
            "run",
            "model",
            "a complex prompt with many words",
            "--seed",
            "99",
            "--steps",
            "10",
            "--width",
            "512",
            "--height",
            "768",
            "--guidance",
            "4.0",
            "--format",
            "jpeg",
            "--batch",
            "2",
            "-o",
            "/tmp/test.jpg",
            "--local",
            "--eager",
        ]);
        match cli.command {
            Commands::Run {
                seed,
                steps,
                width,
                height,
                guidance,
                format,
                no_metadata,
                batch,
                output,
                local,
                eager,
                ..
            } => {
                assert_eq!(seed, Some(99));
                assert_eq!(steps, Some(10));
                assert_eq!(width, Some(512));
                assert_eq!(height, Some(768));
                assert_eq!(guidance, Some(4.0));
                assert_eq!(format, Some(OutputFormat::Jpeg));
                assert!(!no_metadata);
                assert_eq!(batch, 2);
                assert_eq!(output.as_deref(), Some("/tmp/test.jpg"));
                assert!(local);
                assert!(eager);
            }
            _ => panic!("expected Run"),
        }
    }

    #[test]
    fn run_defaults_when_no_flags() {
        let cli = parse(&["run", "model", "test"]);
        match cli.command {
            Commands::Run {
                seed,
                steps,
                width,
                height,
                guidance,
                format,
                no_metadata,
                batch,
                output,
                local,
                eager,
                ..
            } => {
                assert_eq!(seed, None);
                assert_eq!(steps, None);
                assert_eq!(width, None);
                assert_eq!(height, None);
                assert_eq!(guidance, None);
                assert_eq!(format, None);
                assert!(!no_metadata);
                assert_eq!(batch, 1);
                assert_eq!(output, None);
                assert!(!local);
                assert!(!eager);
            }
            _ => panic!("expected Run"),
        }
    }

    #[test]
    fn run_batch_zero_rejected() {
        let result = try_parse(&["run", "model", "test", "--batch", "0"]);
        assert!(result.is_err());
    }

    #[test]
    fn run_batch_large_accepted() {
        let cli = parse(&["run", "model", "test", "--batch", "100"]);
        match cli.command {
            Commands::Run { batch, .. } => assert_eq!(batch, 100),
            _ => panic!("expected Run"),
        }
    }

    #[test]
    fn run_format_invalid_rejected() {
        let result = try_parse(&["run", "model", "test", "--format", "bmp"]);
        assert!(result.is_err());
    }

    #[test]
    fn run_format_gif_accepted() {
        // "gif" is in the new format list [png, jpeg, gif, apng, webp, mp4]
        let cli = parse(&["run", "model", "test", "--format", "gif"]);
        match cli.command {
            Commands::Run { format, .. } => assert_eq!(format, Some(OutputFormat::Gif)),
            _ => panic!("expected Run"),
        }
    }

    #[test]
    fn run_no_metadata_flag() {
        let cli = parse(&["run", "model", "test", "--no-metadata"]);
        match cli.command {
            Commands::Run { no_metadata, .. } => assert!(no_metadata),
            _ => panic!("expected Run"),
        }
    }

    #[test]
    fn run_preview_flag() {
        let cli = parse(&["run", "model", "test", "--preview"]);
        match cli.command {
            Commands::Run { preview, .. } => assert!(preview),
            _ => panic!("expected Run"),
        }
    }

    #[test]
    fn run_preview_default_false() {
        let cli = parse(&["run", "model", "test"]);
        match cli.command {
            Commands::Run { preview, .. } => assert!(!preview),
            _ => panic!("expected Run"),
        }
    }

    #[test]
    fn run_image_flag() {
        let cli = parse(&["run", "model", "test", "--image", "photo.png"]);
        match cli.command {
            Commands::Run { image, .. } => assert_eq!(image, vec!["photo.png"]),
            _ => panic!("expected Run"),
        }
    }

    #[test]
    fn run_image_stdin() {
        let cli = parse(&["run", "model", "test", "--image", "-"]);
        match cli.command {
            Commands::Run { image, .. } => assert_eq!(image, vec!["-"]),
            _ => panic!("expected Run"),
        }
    }

    #[test]
    fn run_image_short_flag() {
        let cli = parse(&["run", "model", "test", "-i", "input.jpg"]);
        match cli.command {
            Commands::Run { image, .. } => assert_eq!(image, vec!["input.jpg"]),
            _ => panic!("expected Run"),
        }
    }

    #[test]
    fn run_strength_flag() {
        let cli = parse(&["run", "model", "test", "--strength", "0.5"]);
        match cli.command {
            Commands::Run { strength, .. } => assert_eq!(strength, Some(0.5)),
            _ => panic!("expected Run"),
        }
    }

    #[test]
    fn run_strength_default() {
        let cli = parse(&["run", "model", "test"]);
        match cli.command {
            Commands::Run { strength, .. } => assert_eq!(strength, None),
            _ => panic!("expected Run"),
        }
    }

    #[test]
    fn run_image_defaults_none() {
        let cli = parse(&["run", "model", "test"]);
        match cli.command {
            Commands::Run { image, .. } => assert!(image.is_empty()),
            _ => panic!("expected Run"),
        }
    }

    #[test]
    fn run_image_flag_repeats() {
        let cli = parse(&[
            "run", "model", "test", "--image", "a.png", "--image", "b.png",
        ]);
        match cli.command {
            Commands::Run { image, .. } => assert_eq!(image, vec!["a.png", "b.png"]),
            _ => panic!("expected Run"),
        }
    }

    #[test]
    fn default_no_args_parses() {
        let cli = parse(&["default"]);
        match cli.command {
            Commands::Default { model } => {
                assert!(model.is_none());
            }
            _ => panic!("expected Default"),
        }
    }

    #[test]
    fn default_with_model_parses() {
        let cli = parse(&["default", "flux-dev:q4"]);
        match cli.command {
            Commands::Default { model } => {
                assert_eq!(model.as_deref(), Some("flux-dev:q4"));
            }
            _ => panic!("expected Default"),
        }
    }

    #[test]
    fn info_no_args_parses() {
        let cli = parse(&["info"]);
        match cli.command {
            Commands::Info { model, verify } => {
                assert!(model.is_none());
                assert!(!verify);
            }
            _ => panic!("expected Info"),
        }
    }

    #[test]
    fn info_with_model_parses() {
        let cli = parse(&["info", "flux-schnell"]);
        match cli.command {
            Commands::Info { model, verify } => {
                assert_eq!(model.as_deref(), Some("flux-schnell"));
                assert!(!verify);
            }
            _ => panic!("expected Info"),
        }
    }

    #[test]
    fn info_verify_with_model_parses() {
        let cli = parse(&["info", "flux-schnell", "--verify"]);
        match cli.command {
            Commands::Info { model, verify } => {
                assert_eq!(model.as_deref(), Some("flux-schnell"));
                assert!(verify);
            }
            _ => panic!("expected Info"),
        }
    }

    // ── stats tests ─────────────────────────────────────────────────────

    #[test]
    fn stats_parses() {
        let cli = parse(&["stats"]);
        match cli.command {
            Commands::Stats { json } => assert!(!json),
            _ => panic!("expected Stats"),
        }
    }

    #[test]
    fn stats_json_flag() {
        let cli = parse(&["stats", "--json"]);
        match cli.command {
            Commands::Stats { json } => assert!(json),
            _ => panic!("expected Stats"),
        }
    }

    // ── queue tests ─────────────────────────────────────────────────────

    #[test]
    fn queue_list_parses_held_and_json_flags() {
        assert!(matches!(
            parse(&["queue", "list"]).command,
            Commands::Queue {
                action: QueueAction::List {
                    held: false,
                    json: false
                }
            }
        ));
        assert!(matches!(
            parse(&["queue", "list", "--held", "--json"]).command,
            Commands::Queue {
                action: QueueAction::List {
                    held: true,
                    json: true
                }
            }
        ));
    }

    #[test]
    fn queue_cancel_accepts_ids_all_or_a_batch_but_never_two_of_them() {
        match parse(&["queue", "cancel", "job-1", "job-2"]).command {
            Commands::Queue {
                action: QueueAction::Cancel { job_ids, all, .. },
            } => {
                assert_eq!(job_ids, vec!["job-1", "job-2"]);
                assert!(!all);
            }
            _ => panic!("expected Queue cancel"),
        }
        assert!(matches!(
            parse(&["queue", "cancel", "--all", "--yes"]).command,
            Commands::Queue {
                action: QueueAction::Cancel {
                    all: true,
                    yes: true,
                    ..
                }
            }
        ));
        match parse(&["queue", "cancel", "--batch", "batch-7"]).command {
            Commands::Queue {
                action: QueueAction::Cancel { batch, .. },
            } => assert_eq!(batch.as_deref(), Some("batch-7")),
            _ => panic!("expected Queue cancel"),
        }
        // Naming a job and `--all` would leave the outcome ambiguous.
        for conflicting in [
            vec!["queue", "cancel", "job-1", "--all"],
            vec!["queue", "cancel", "job-1", "--batch", "batch-7"],
            vec!["queue", "cancel", "--all", "--batch", "batch-7"],
        ] {
            assert!(
                try_parse(&conflicting).is_err(),
                "{conflicting:?} must not parse"
            );
        }
    }

    #[test]
    fn queue_retry_takes_ids_or_held_but_not_both() {
        match parse(&["queue", "retry", "job-1"]).command {
            Commands::Queue {
                action: QueueAction::Retry { job_ids, held },
            } => {
                assert_eq!(job_ids, vec!["job-1"]);
                assert!(!held);
            }
            _ => panic!("expected Queue retry"),
        }
        assert!(matches!(
            parse(&["queue", "retry", "--held"]).command,
            Commands::Queue {
                action: QueueAction::Retry { held: true, .. }
            }
        ));
        assert!(
            try_parse(&["queue", "retry", "job-1", "--held"]).is_err(),
            "--held with explicit ids must not parse"
        );
    }

    #[test]
    fn queue_move_show_and_the_lifecycle_verbs_parse() {
        match parse(&["queue", "move", "job-1", "--to", "0"]).command {
            Commands::Queue {
                action: QueueAction::Move { job_id, to },
            } => {
                assert_eq!(job_id, "job-1");
                assert_eq!(to, 0);
            }
            _ => panic!("expected Queue move"),
        }
        assert!(
            try_parse(&["queue", "move", "job-1"]).is_err(),
            "a move with no destination must not parse"
        );
        match parse(&["queue", "show", "job-1", "--json"]).command {
            Commands::Queue {
                action: QueueAction::Show { job_id, json },
            } => {
                assert_eq!(job_id, "job-1");
                assert!(json);
            }
            _ => panic!("expected Queue show"),
        }
        for (args, expected) in [
            (vec!["queue", "pause"], "pause"),
            (vec!["queue", "resume"], "resume"),
            (vec!["queue", "sweep"], "sweep"),
        ] {
            let matched = match parse(&args).command {
                Commands::Queue {
                    action: QueueAction::Pause { job_id },
                } => {
                    assert!(job_id.is_none());
                    "pause"
                }
                Commands::Queue {
                    action: QueueAction::Resume { job_id },
                } => {
                    assert!(job_id.is_none());
                    "resume"
                }
                Commands::Queue {
                    action: QueueAction::Sweep,
                } => "sweep",
                _ => panic!("expected a Queue lifecycle verb"),
            };
            assert_eq!(matched, expected);
        }
        assert!(matches!(
            parse(&["queue", "pause", "job-1"]).command,
            Commands::Queue {
                action: QueueAction::Pause { job_id: Some(job_id) }
            } if job_id == "job-1"
        ));
    }

    // ── library tests ───────────────────────────────────────────────────

    #[test]
    fn library_list_parses_filters_and_bounds_limit() {
        match parse(&[
            "library",
            "list",
            "--tag",
            "Night Owls",
            "--collection",
            "Portfolio",
            "--favorite",
            "--format",
            "png",
            "--limit",
            "1000",
            "--offset",
            "4",
            "--json",
        ])
        .command
        {
            Commands::Library {
                action:
                    LibraryAction::List {
                        tags,
                        collection,
                        favorite,
                        format,
                        limit,
                        offset,
                        json,
                        ..
                    },
            } => {
                assert_eq!(tags, vec!["Night Owls"]);
                assert_eq!(collection.as_deref(), Some("Portfolio"));
                assert!(favorite);
                assert_eq!(format, Some(OutputFormat::Png));
                assert_eq!(limit, 1000);
                assert_eq!(offset, 4);
                assert!(json);
            }
            _ => panic!("expected Library list"),
        }
        assert!(try_parse(&["library", "list", "--limit", "0"]).is_err());
        assert!(try_parse(&["library", "list", "--limit", "1001"]).is_err());
    }

    #[test]
    fn library_show_keeps_json_and_preview_exclusive() {
        assert!(try_parse(&["library", "show", "print.png", "--json", "--preview",]).is_err());
        match parse(&["library", "show", "print.png", "--preview"]).command {
            Commands::Library {
                action:
                    LibraryAction::Show {
                        filename,
                        preview,
                        json,
                    },
            } => {
                assert_eq!(filename, "print.png");
                assert!(preview);
                assert!(!json);
            }
            _ => panic!("expected Library show"),
        }
    }

    /// `--format` is parsed by the WIRE type, so the CLI can never accept a
    /// container the export endpoint would refuse.
    #[test]
    fn library_export_parses_every_mesh_container_and_rejects_the_rest() {
        for (flag, expected) in [
            ("glb", mold_core::MeshExportFormat::Glb),
            ("obj", mold_core::MeshExportFormat::Obj),
            ("stl", mold_core::MeshExportFormat::Stl),
            ("ply", mold_core::MeshExportFormat::Ply),
            ("gif", mold_core::MeshExportFormat::Gif),
            ("apng", mold_core::MeshExportFormat::Apng),
            ("webp", mold_core::MeshExportFormat::Webp),
        ] {
            match parse(&["library", "export", "chair.glb", "--format", flag]).command {
                Commands::Library {
                    action:
                        LibraryAction::Export {
                            filename,
                            format,
                            output,
                            turntable,
                            geometry,
                        },
                } => {
                    assert_eq!(filename, "chair.glb");
                    assert_eq!(format, expected);
                    assert_eq!(output, None);
                    assert_eq!(
                        mold_core::MeshTurntableOptions::from(turntable),
                        mold_core::MeshTurntableOptions::default(),
                        "no turntable flag means the server's defaults"
                    );
                    assert_eq!(
                        mold_core::MeshGeometryOptions::from(geometry),
                        mold_core::MeshGeometryOptions::default(),
                        "no geometry flag means the format's own defaults"
                    );
                }
                _ => panic!("expected Library export"),
            }
        }
        assert!(try_parse(&["library", "export", "chair.glb", "--format", "fbx"]).is_err());
        assert!(try_parse(&["library", "export", "chair.glb"]).is_err());

        match parse(&[
            "library",
            "export",
            "chair.glb",
            "--format",
            "stl",
            "-o",
            "-",
        ])
        .command
        {
            Commands::Library {
                action: LibraryAction::Export { output, .. },
            } => assert_eq!(output.as_deref(), Some("-")),
            _ => panic!("expected Library export"),
        }
    }

    /// The turntable flags are the video export's own names (`playback`,
    /// `repeat`, `max_dimension`) plus `frames` and `fps`, parsed by the wire
    /// enums so a spelling the server would refuse never leaves the shell.
    #[test]
    fn library_export_turntable_flags_parse_by_the_wire_types() {
        match parse(&[
            "library",
            "export",
            "chair.glb",
            "--format",
            "gif",
            "--playback",
            "bounce",
            "--repeat",
            "once",
            "--max-dimension",
            "480",
            "--frames",
            "24",
            "--fps",
            "12",
        ])
        .command
        {
            Commands::Library {
                action: LibraryAction::Export { turntable, .. },
            } => assert_eq!(
                mold_core::MeshTurntableOptions::from(turntable),
                mold_core::MeshTurntableOptions {
                    playback: Some(mold_core::MeshTurntablePlayback::Bounce),
                    repeat: Some(mold_core::MeshTurntableRepeat::Once),
                    max_dimension: Some(480),
                    frames: Some(24),
                    fps: Some(12),
                }
            ),
            _ => panic!("expected Library export"),
        }
        for bad in [
            ["--playback", "pingpong"],
            ["--repeat", "twice"],
            ["--frames", "many"],
            ["--fps", "-1"],
        ] {
            assert!(
                try_parse(&[
                    "library",
                    "export",
                    "chair.glb",
                    "--format",
                    "gif",
                    bad[0],
                    bad[1]
                ])
                .is_err(),
                "{bad:?} must not parse"
            );
        }
    }

    /// The geometry flags are parsed by the wire enums, so a spelling the
    /// server would refuse never leaves the shell, and an absent flag stays
    /// absent rather than becoming a value the CLI invented.
    #[test]
    fn library_export_geometry_flags_parse_by_the_wire_types() {
        match parse(&[
            "library",
            "export",
            "chair.glb",
            "--format",
            "stl",
            "--size-mm",
            "120",
            "--up-axis",
            "y",
            "--origin",
            "center",
        ])
        .command
        {
            Commands::Library {
                action: LibraryAction::Export { geometry, .. },
            } => assert_eq!(
                mold_core::MeshGeometryOptions::from(geometry),
                mold_core::MeshGeometryOptions {
                    size_mm: Some(120.0),
                    up_axis: Some(mold_core::MeshUpAxis::Y),
                    origin: Some(mold_core::MeshExportOrigin::Center),
                }
            ),
            _ => panic!("expected Library export"),
        }

        match parse(&[
            "library",
            "export",
            "chair.glb",
            "--format",
            "ply",
            "--up-axis",
            "z",
        ])
        .command
        {
            Commands::Library {
                action: LibraryAction::Export { geometry, .. },
            } => assert_eq!(
                mold_core::MeshGeometryOptions::from(geometry),
                mold_core::MeshGeometryOptions {
                    up_axis: Some(mold_core::MeshUpAxis::Z),
                    ..Default::default()
                },
                "the flags a request does not name stay absent"
            ),
            _ => panic!("expected Library export"),
        }

        for bad in [
            ["--up-axis", "w"],
            ["--origin", "bed"],
            ["--size-mm", "large"],
        ] {
            assert!(
                try_parse(&[
                    "library",
                    "export",
                    "chair.glb",
                    "--format",
                    "stl",
                    bad[0],
                    bad[1]
                ])
                .is_err(),
                "{bad:?} must not parse"
            );
        }
    }

    #[test]
    fn library_tag_and_collection_membership_parse() {
        match parse(&[
            "library", "tag", "add", "a.png", "b.png", "--tag", "owl", "--tag", "night",
        ])
        .command
        {
            Commands::Library {
                action:
                    LibraryAction::Tag {
                        action: LibraryTagAction::Add { filenames, tags },
                    },
            } => {
                assert_eq!(filenames, vec!["a.png", "b.png"]);
                assert_eq!(tags, vec!["owl", "night"]);
            }
            _ => panic!("expected Library tag add"),
        }
        match parse(&["library", "collection", "remove", "Portfolio", "a.png"]).command {
            Commands::Library {
                action:
                    LibraryAction::Collection {
                        action:
                            LibraryCollectionAction::Remove {
                                collection,
                                filenames,
                            },
                    },
            } => {
                assert_eq!(collection, "Portfolio");
                assert_eq!(filenames, vec!["a.png"]);
            }
            _ => panic!("expected Library collection remove"),
        }
    }

    #[test]
    fn library_title_requires_text_or_clear() {
        assert!(try_parse(&["library", "title", "a.png"]).is_err());
        assert!(try_parse(&["library", "title", "a.png", "new title", "--clear"]).is_err());
        assert!(matches!(
            parse(&["library", "title", "a.png", "--clear"]).command,
            Commands::Library {
                action: LibraryAction::Title { clear: true, .. }
            }
        ));
    }

    // ── trash tests ─────────────────────────────────────────────────────

    #[test]
    fn trash_list_parses_with_and_without_json() {
        let cli = parse(&["trash", "list"]);
        match cli.command {
            Commands::Trash {
                action: TrashAction::List { json },
            } => assert!(!json),
            _ => panic!("expected Trash list"),
        }
        let cli = parse(&["trash", "list", "--json"]);
        match cli.command {
            Commands::Trash {
                action: TrashAction::List { json },
            } => assert!(json),
            _ => panic!("expected Trash list --json"),
        }
    }

    #[test]
    fn trash_restore_takes_one_or_more_filenames() {
        let cli = parse(&["trash", "restore", "a.png", "b.mp4"]);
        match cli.command {
            Commands::Trash {
                action: TrashAction::Restore { filenames },
            } => assert_eq!(filenames, vec!["a.png", "b.mp4"]),
            _ => panic!("expected Trash restore"),
        }
        assert!(
            try_parse(&["trash", "restore"]).is_err(),
            "restore without filenames must be a usage error"
        );
    }

    #[test]
    fn trash_empty_confirms_unless_yes() {
        let cli = parse(&["trash", "empty"]);
        match cli.command {
            Commands::Trash {
                action: TrashAction::Empty { yes },
            } => assert!(!yes),
            _ => panic!("expected Trash empty"),
        }
        for args in [["trash", "empty", "--yes"], ["trash", "empty", "-y"]] {
            let cli = parse(&args);
            match cli.command {
                Commands::Trash {
                    action: TrashAction::Empty { yes },
                } => assert!(yes),
                _ => panic!("expected Trash empty --yes"),
            }
        }
    }

    #[test]
    fn trash_sweep_parses() {
        let cli = parse(&["trash", "sweep"]);
        assert!(matches!(
            cli.command,
            Commands::Trash {
                action: TrashAction::Sweep
            }
        ));
        assert!(try_parse(&["trash"]).is_err(), "trash needs a subcommand");
    }

    // ── --title tests ───────────────────────────────────────────────────

    #[test]
    fn run_title_flag_is_trimmed_and_defaults_to_none() {
        let cli = parse(&[
            "run",
            "flux-schnell",
            "a cat",
            "--title",
            "  Smurf village  ",
        ]);
        match cli.command {
            Commands::Run { title, .. } => assert_eq!(title.as_deref(), Some("Smurf village")),
            _ => panic!("expected Run"),
        }
        let cli = parse(&["run", "a cat"]);
        match cli.command {
            Commands::Run { title, .. } => assert!(title.is_none()),
            _ => panic!("expected Run"),
        }
    }

    #[test]
    fn run_title_flag_rejects_empty_control_and_overlong_titles() {
        assert!(try_parse(&["run", "a cat", "--title", "   "]).is_err());
        assert!(try_parse(&["run", "a cat", "--title", "bad\ntitle"]).is_err());
        let long = "x".repeat(mold_core::PRINT_TITLE_MAX_CHARS + 1);
        assert!(try_parse(&["run", "a cat", "--title", &long]).is_err());
        let max = "x".repeat(mold_core::PRINT_TITLE_MAX_CHARS);
        assert!(try_parse(&["run", "a cat", "--title", &max]).is_ok());
    }

    // ── clean tests ─────────────────────────────────────────────────────

    #[test]
    fn clean_parses() {
        let cli = parse(&["clean"]);
        match cli.command {
            Commands::Clean { force, older_than } => {
                assert!(!force);
                assert!(older_than.is_none());
            }
            _ => panic!("expected Clean"),
        }
    }

    #[test]
    fn clean_force_flag() {
        let cli = parse(&["clean", "--force"]);
        match cli.command {
            Commands::Clean { force, older_than } => {
                assert!(force);
                assert!(older_than.is_none());
            }
            _ => panic!("expected Clean"),
        }
    }

    /// `--local` reads identically on either side of the subcommand, and
    /// clap stores each in its own field. Honouring only one would record
    /// consent on MOLD_HOST after the user asked for this machine.
    #[test]
    fn licenses_accept_honours_local_on_either_side() {
        for args in [
            vec!["licenses", "--local", "accept", "some-license"],
            vec!["licenses", "accept", "some-license", "--local"],
        ] {
            let cli = parse(&args);
            match cli.command {
                Commands::Licenses { action, local } => {
                    let Some(LicensesAction::Accept {
                        ids,
                        local: accept_local,
                    }) = action
                    else {
                        panic!("expected the accept subcommand for {args:?}");
                    };
                    assert_eq!(ids, vec!["some-license".to_string()]);
                    assert!(
                        local || accept_local,
                        "--local must survive on either side: {args:?}"
                    );
                }
                _ => panic!("expected Licenses for {args:?}"),
            }
        }
    }

    /// A bundle covered by two agreements is one command, not two runs.
    #[test]
    fn accept_license_is_repeatable_on_pull() {
        let cli = parse(&[
            "pull",
            "some-model",
            "--accept-license",
            "one",
            "--accept-license",
            "two",
        ]);
        match cli.command {
            Commands::Pull { accept_license, .. } => {
                assert_eq!(accept_license, vec!["one".to_string(), "two".to_string()]);
            }
            _ => panic!("expected Pull"),
        }
    }

    /// The bare listing must keep working now that it has a subcommand.
    #[test]
    fn licenses_without_a_subcommand_still_lists() {
        let cli = parse(&["licenses"]);
        match cli.command {
            Commands::Licenses { action, local } => {
                assert!(action.is_none());
                assert!(!local);
            }
            _ => panic!("expected Licenses"),
        }
    }

    #[test]
    fn clean_older_than_flag() {
        let cli = parse(&["clean", "--older-than", "30d"]);
        match cli.command {
            Commands::Clean { force, older_than } => {
                assert!(!force);
                assert_eq!(older_than.as_deref(), Some("30d"));
            }
            _ => panic!("expected Clean"),
        }
    }

    #[test]
    fn clean_older_than_and_force() {
        let cli = parse(&["clean", "--older-than", "7d", "--force"]);
        match cli.command {
            Commands::Clean { force, older_than } => {
                assert!(force);
                assert_eq!(older_than.as_deref(), Some("7d"));
            }
            _ => panic!("expected Clean"),
        }
    }

    #[test]
    fn run_frames_flag() {
        let cli = parse(&["run", "model", "test", "--frames", "25"]);
        match cli.command {
            Commands::Run { frames, fps, .. } => {
                assert_eq!(frames, Some(25));
                assert_eq!(fps, None);
            }
            _ => panic!("expected Run"),
        }
    }

    #[test]
    fn run_predict_duration_flag_is_explicit_and_conflicts_with_frames() {
        let cli = parse(&[
            "run",
            "ltx-2.5-22b-distilled:int8-conv",
            "test",
            "--predict-duration",
        ]);
        match cli.command {
            Commands::Run {
                frames,
                predict_duration,
                ..
            } => {
                assert_eq!(frames, None);
                assert!(predict_duration);
            }
            _ => panic!("expected Run"),
        }
        assert!(try_parse(&[
            "run",
            "ltx-2.5-22b-distilled:int8-conv",
            "test",
            "--predict-duration",
            "--frames",
            "97",
        ])
        .is_err());
    }

    #[test]
    fn run_fps_flag() {
        let cli = parse(&["run", "model", "test", "--frames", "17", "--fps", "30"]);
        match cli.command {
            Commands::Run { frames, fps, .. } => {
                assert_eq!(frames, Some(17));
                assert_eq!(fps, Some(30));
            }
            _ => panic!("expected Run"),
        }
    }

    #[test]
    fn run_fps_without_frames() {
        let cli = parse(&["run", "model", "test", "--fps", "15"]);
        match cli.command {
            Commands::Run { frames, fps, .. } => {
                assert_eq!(frames, None);
                assert_eq!(fps, Some(15));
            }
            _ => panic!("expected Run"),
        }
    }

    #[test]
    fn run_h3_authoring_flags_preserve_reference_order() {
        let cli = parse(&[
            "run",
            "minimax-h3-ref2va",
            "match the cast",
            "--duration",
            "8.5",
            "--reference",
            "image=/tmp/cast.png",
            "--reference",
            "video=/tmp/motion.mp4",
            "--reference",
            "audio=/tmp/voice.wav",
        ]);
        match cli.command {
            Commands::Run {
                duration,
                reference,
                format,
                strength,
                ..
            } => {
                assert_eq!(duration, Some(8.5));
                assert_eq!(reference.len(), 3);
                assert_eq!(reference[0].kind, commands::h3::ReferenceKind::Image);
                assert_eq!(reference[1].kind, commands::h3::ReferenceKind::Video);
                assert_eq!(reference[2].kind, commands::h3::ReferenceKind::Audio);
                assert_eq!(format, None);
                assert_eq!(strength, None);
            }
            _ => panic!("expected Run command"),
        }
    }

    #[test]
    fn run_h3_duration_conflicts_with_explicit_frames() {
        let error = try_parse(&[
            "run",
            "minimax-h3",
            "a synchronized shot",
            "--duration",
            "5",
            "--frames",
            "124",
        ])
        .err()
        .expect("duration and frames must conflict");
        assert_eq!(error.kind(), clap::error::ErrorKind::ArgumentConflict);
    }

    #[test]
    fn run_h3_first_frame_conflicts_with_legacy_image_flag() {
        let error = try_parse(&[
            "run",
            "minimax-h3",
            "animate",
            "--first-frame",
            "/tmp/first.png",
            "--image",
            "/tmp/other.png",
        ])
        .err()
        .expect("first-frame and image must conflict");
        assert_eq!(error.kind(), clap::error::ErrorKind::ArgumentConflict);
    }

    /// `--last-image` is only ever the closing half of a pair, and it builds
    /// the keyframe list itself — a hand-written `--keyframe` alongside it
    /// would be two conflicting answers to the same question.
    #[test]
    fn run_last_image_requires_an_opening_image_and_owns_the_keyframes() {
        let missing_opening = try_parse(&[
            "run",
            "wan22-i2v-a14b:q5",
            "a cat leaping a fence",
            "--last-image",
            "/tmp/last.png",
        ])
        .err()
        .expect("last-image must require image");
        assert_eq!(
            missing_opening.kind(),
            clap::error::ErrorKind::MissingRequiredArgument
        );

        let conflicting = try_parse(&[
            "run",
            "wan22-i2v-a14b:q5",
            "a cat leaping a fence",
            "--image",
            "/tmp/first.png",
            "--last-image",
            "/tmp/last.png",
            "--keyframe",
            "0:/tmp/other.png",
        ])
        .err()
        .expect("last-image and keyframe must conflict");
        assert_eq!(conflicting.kind(), clap::error::ErrorKind::ArgumentConflict);

        let cli = parse(&[
            "run",
            "wan22-i2v-a14b:q5",
            "a cat leaping a fence",
            "--image",
            "/tmp/first.png",
            "--last-image",
            "/tmp/last.png",
        ]);
        match cli.command {
            Commands::Run {
                image, last_image, ..
            } => {
                assert_eq!(image, vec!["/tmp/first.png".to_string()]);
                assert_eq!(last_image.as_deref(), Some("/tmp/last.png"));
            }
            _ => panic!("expected a Run command"),
        }
    }

    #[test]
    fn run_chain_flags_parse() {
        let cli = parse(&[
            "run",
            "ltx-2-19b-distilled:fp8",
            "a cat",
            "--frames",
            "200",
            "--clip-frames",
            "97",
            "--motion-tail",
            "4",
        ]);
        match cli.command {
            Commands::Run {
                frames,
                clip_frames,
                motion_tail,
                ..
            } => {
                assert_eq!(frames, Some(200));
                assert_eq!(clip_frames, Some(97));
                assert_eq!(motion_tail, 4);
            }
            _ => panic!("expected Run"),
        }
    }

    #[test]
    fn run_motion_tail_defaults_to_seventeen() {
        let cli = parse(&["run", "ltx-2-19b-distilled:fp8", "a cat", "--frames", "200"]);
        match cli.command {
            Commands::Run {
                motion_tail,
                clip_frames,
                ..
            } => {
                assert_eq!(
                    motion_tail, 17,
                    "default motion tail must be 17 frames (three LTX-2 latent frames: causal + two continuation)"
                );
                assert_eq!(clip_frames, None);
            }
            _ => panic!("expected Run"),
        }
    }

    // --- Regression test for issue #190: --version includes git SHA ---

    #[test]
    fn version_flag_includes_git_sha() {
        // Regression: `mold --version` used bare `#[command(version)]` which only
        // printed CARGO_PKG_VERSION without the git SHA. Now it uses
        // `mold_core::build_info::FULL_VERSION` to match `mold version`.
        // Building the full clap tree needs more than the default test stack.
        let version = std::thread::Builder::new()
            .stack_size(64 << 20)
            .spawn(|| {
                Cli::command()
                    .get_version()
                    .expect("version should be set")
                    .to_string()
            })
            .expect("spawn")
            .join()
            .expect("join");
        let version = version.as_str();
        assert_eq!(
            version,
            mold_core::build_info::FULL_VERSION,
            "--version should match FULL_VERSION"
        );
        // Verify it matches the runtime version_string() too
        assert_eq!(
            version,
            mold_core::build_info::version_string(),
            "--version should match `mold version` output"
        );
    }

    #[test]
    fn run_parses_every_identity_flag() {
        let cli = parse(&[
            "run",
            "flux-dev:q4",
            "a portrait",
            "--id-image",
            "/photos/face.png",
            "--id-weight",
            "0.85",
            "--id-start-step",
            "2",
            "--true-cfg",
            "2.5",
            "--cfg-start-step",
            "3",
        ]);
        match cli.command {
            Commands::Run {
                id_image,
                id_weight,
                id_start_step,
                true_cfg,
                cfg_start_step,
                ..
            } => {
                assert_eq!(id_image, vec![std::path::PathBuf::from("/photos/face.png")]);
                assert_eq!(id_weight, Some(0.85));
                assert_eq!(id_start_step, Some(2));
                assert_eq!(true_cfg, Some(2.5));
                assert_eq!(cfg_start_step, Some(3));
            }
            _ => panic!("expected Run command"),
        }
    }

    /// Several references of the same person average into one identity, so the
    /// flag repeats and the order it was given in is the order that ships.
    #[test]
    fn run_accepts_repeated_id_image_flags_in_order() {
        let cli = parse(&[
            "run",
            "flux-dev:q4",
            "a portrait",
            "--id-image",
            "/photos/one.png",
            "--id-image",
            "/photos/two.png",
            "--id-image",
            "/photos/three.jpg",
        ]);
        match cli.command {
            Commands::Run { id_image, .. } => assert_eq!(
                id_image,
                vec![
                    std::path::PathBuf::from("/photos/one.png"),
                    std::path::PathBuf::from("/photos/two.png"),
                    std::path::PathBuf::from("/photos/three.jpg"),
                ]
            ),
            _ => panic!("expected Run command"),
        }
    }

    #[test]
    fn run_leaves_every_identity_field_absent_by_default() {
        let cli = parse(&["run", "flux-dev:q4", "a portrait"]);
        match cli.command {
            Commands::Run {
                id_image,
                id_weight,
                id_start_step,
                true_cfg,
                cfg_start_step,
                ..
            } => {
                assert!(id_image.is_empty());
                assert!(id_weight.is_none());
                assert!(id_start_step.is_none());
                assert!(true_cfg.is_none());
                assert!(cfg_start_step.is_none());
            }
            _ => panic!("expected Run command"),
        }
    }

    /// A knob with no reference photograph is caught at parse time, before a
    /// server round trip.
    #[test]
    fn run_rejects_an_identity_knob_without_an_image() {
        for flag in ["--id-weight", "--id-start-step", "--true-cfg"] {
            let Err(error) = try_parse(&["run", "flux-dev:q4", "a portrait", flag, "1"]) else {
                panic!("{flag} without an image must be refused");
            };
            let error = error.to_string();
            assert!(error.contains("--id-image"), "{flag}: {error}");
        }
        // A start step with no scale to start is refused on its own flag.
        let Err(error) = try_parse(&[
            "run",
            "flux-dev:q4",
            "a portrait",
            "--id-image",
            "/photos/face.png",
            "--cfg-start-step",
            "2",
        ]) else {
            panic!("--cfg-start-step without --true-cfg must be refused");
        };
        assert!(error.to_string().contains("--true-cfg"), "{error}");
    }

    /// The two combinations `mold_core::identity` refuses at admission are
    /// refused at parse time too, so the user is not told after the upload.
    #[test]
    fn run_rejects_identity_combined_with_img2img_or_a_lora() {
        for extra in [
            vec!["--image", "/photos/source.png"],
            vec!["--lora", "/loras/style.safetensors"],
        ] {
            let mut args = vec![
                "run",
                "flux-dev:q4",
                "a portrait",
                "--id-image",
                "/photos/face.png",
            ];
            args.extend(extra.iter().copied());
            let Err(error) = try_parse(&args) else {
                panic!("{extra:?}: identity with img2img or a LoRA is not yet qualified");
            };
            let error = error.to_string();
            assert!(error.contains("cannot be used with"), "{extra:?}: {error}");
        }
    }

    /// `--id-weight 0` is a legitimate, meaningful value — the falsification
    /// case from `tmp/sdcpp/docs/pulid.md` — and must survive parsing.
    #[test]
    fn run_accepts_an_explicit_zero_identity_weight() {
        let cli = parse(&[
            "run",
            "flux-dev:q4",
            "a portrait",
            "--id-image",
            "/photos/face.png",
            "--id-weight",
            "0",
        ]);
        match cli.command {
            Commands::Run { id_weight, .. } => assert_eq!(id_weight, Some(0.0)),
            _ => panic!("expected Run command"),
        }
    }
}
