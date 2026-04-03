mod commands;
mod control;
mod output;
mod procinfo;
#[cfg(test)]
mod test_support;
mod theme;
mod ui;

use clap::{builder::ValueHint, CommandFactory, Parser, Subcommand};
use clap_complete::engine::ArgValueCandidates;
use mold_core::{OutputFormat, Scheduler};

#[derive(Clone, clap::ValueEnum)]
enum LogFormat {
    Text,
    Json,
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
#[command(version, propagate_version = true)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
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
    },
    /// Show status of the managed server
    Status,
    /// Stop the managed server
    Stop,
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

        /// Output format
        #[arg(long, default_value_t = OutputFormat::Png, help_heading = "Output")]
        format: OutputFormat,

        /// Disable embedded generation metadata in PNG output for this run
        #[arg(long, help_heading = "Output")]
        no_metadata: bool,

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
        #[arg(long, default_value = "1", help_heading = "Image", value_parser = clap::value_parser!(u32).range(1..=16))]
        batch: u32,

        /// Server URL to connect to
        #[arg(long, env = "MOLD_HOST", help_heading = "Server")]
        host: Option<String>,

        /// Skip server and run inference locally (requires GPU features)
        #[arg(long, help_heading = "Server")]
        local: bool,

        /// T5 encoder variant: auto (default), fp16, q8, q6, q5, q4, q3
        #[arg(long, help_heading = "Advanced")]
        t5_variant: Option<String>,

        /// Qwen3 text encoder variant (Z-Image): auto (default), bf16, q8, q6, iq4, q3
        #[arg(long, help_heading = "Advanced")]
        qwen3_variant: Option<String>,

        /// Scheduler algorithm for UNet models: ddim, euler-ancestral, uni-pc
        /// Ignored by flow-matching models (FLUX, SD3, Z-Image, Flux.2, Qwen-Image).
        #[arg(long, env = "MOLD_SCHEDULER", help_heading = "Advanced")]
        scheduler: Option<Scheduler>,

        /// Keep all model components loaded simultaneously (faster but uses more memory).
        /// By default, components are loaded and unloaded sequentially to reduce peak memory.
        #[arg(long, help_heading = "Advanced")]
        eager: bool,

        /// Stream transformer blocks between CPU and GPU one at a time.
        /// Reduces VRAM from ~24GB to ~2-4GB for large models (3-5x slower).
        /// Auto-enabled when VRAM is insufficient. Force with MOLD_OFFLOAD=1.
        #[arg(long, help_heading = "Advanced")]
        offload: bool,

        /// LoRA adapter safetensors file path
        #[arg(long, help_heading = "LoRA", value_hint = ValueHint::FilePath)]
        lora: Option<String>,

        /// LoRA effect strength (0.0 = none, 1.0 = full, up to 2.0)
        #[arg(long, default_value = "1.0", help_heading = "LoRA")]
        lora_scale: f64,

        /// Source image for img2img (file path or - for stdin)
        #[arg(short = 'i', long, help_heading = "img2img", value_hint = ValueHint::FilePath)]
        image: Option<String>,

        /// Denoising strength for img2img (0.0 = no change, 1.0 = full noise)
        #[arg(long, default_value = "0.75", help_heading = "img2img")]
        strength: f64,

        /// Mask image for inpainting (file path; white = repaint, black = preserve)
        #[arg(long, requires = "image", help_heading = "img2img", value_hint = ValueHint::FilePath)]
        mask: Option<String>,

        /// Control image for ControlNet conditioning (file path, e.g. edges.png)
        #[arg(long, help_heading = "ControlNet", value_hint = ValueHint::FilePath)]
        control: Option<String>,

        /// ControlNet model name (e.g. controlnet-canny-sd15)
        #[arg(long, requires = "control", help_heading = "ControlNet")]
        control_model: Option<String>,

        /// ControlNet conditioning scale (0.0 = no effect, 1.0 = full, up to 2.0)
        #[arg(long, default_value = "1.0", help_heading = "ControlNet")]
        control_scale: f64,

        /// Negative prompt — what to avoid generating (CFG-based models: SD1.5, SDXL, SD3, Wuerstchen)
        #[arg(short = 'n', long, help_heading = "Image")]
        negative_prompt: Option<String>,

        /// Suppress config-file default negative prompt (use empty unconditional)
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
    },

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

        /// Also start the Discord bot in this process
        #[cfg(feature = "discord")]
        #[arg(long)]
        discord: bool,
    },

    /// Manage a background mold server daemon (start, stop, status)
    #[command(after_long_help = "\
Examples:
  mold server start              Start background server on port 7680
  mold server start --port 8080  Custom port
  mold server status             Check if server is running
  mold server stop               Stop the server")]
    Server {
        #[command(subcommand)]
        action: ServerAction,
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
        #[command(subcommand)]
        action: ConfigAction,
    },

    /// Preview LLM prompt expansion without generating images
    ///
    /// Expand a short prompt into detailed image generation prompts using an LLM.
    /// Useful for previewing what --expand will produce.
    #[command(after_long_help = "\
Examples:
  mold expand \"a cat\"
  mold expand \"a cat\" --model flux-schnell
  mold expand \"cyberpunk city\" --variations 5
  mold expand \"a cat\" --variations 3 --json")]
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
    },

    /// Unload the current model from the server to free GPU memory
    #[command(
        after_long_help = "Requires a running server (mold serve). Use 'mold ps' to check status."
    )]
    Unload,

    /// Show server status and loaded models
    #[command(after_long_help = "Use 'mold unload' to free GPU memory when idle.")]
    Ps,

    /// Show version information
    Version,

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
    Completions {
        /// Shell to generate completions for (bash, zsh, fish, elvish, powershell)
        shell: String,
    },
}

#[tokio::main]
async fn main() {
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
        let is_oom = msg.contains("CUDA_ERROR_OUT_OF_MEMORY")
            || msg.contains("out of memory")
            || msg.contains("exceeds available VRAM")
            || msg.contains("Failed to create metal resource");
        if is_oom {
            // Don't show misleading CUDA errors on Metal — replace with a clean message
            let is_metal_oom = cfg!(target_os = "macos")
                && (msg.contains("CUDA_ERROR_OUT_OF_MEMORY")
                    || msg.contains("Failed to create metal resource"));
            if is_metal_oom {
                eprintln!("{} Metal out of memory", theme::prefix_error());
            } else {
                eprintln!("{} {display}", theme::prefix_error());
            }
            eprintln!();
            eprintln!("  GPU ran out of memory during generation.");
            eprintln!("  Try these fixes:");
            eprintln!();
            eprintln!("    Reduce resolution:  --width 512 --height 512");
            eprintln!("    Use a smaller model: mold run <model>:q4 \"...\"");
            eprintln!();
            eprintln!("  For img2img, the source image resolution is used by default.");
            eprintln!("  Override with --width/--height to reduce VRAM usage.");
            eprintln!("  Run 'mold list' to see available models and sizes.");
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
        eprintln!("{} {display}", theme::prefix_error());
        for cause in e.chain().skip(1) {
            eprintln!("  {} {cause}", theme::prefix_cause());
        }
        std::process::exit(1);
    }
}

async fn run() -> anyhow::Result<()> {
    // Parse CLI first so we can set the log level based on the subcommand.
    clap_complete::CompleteEnv::with_factory(Cli::command).complete();
    let cli = Cli::parse();

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
            host,
            format,
            no_metadata,
            preview,
            local,
            t5_variant,
            qwen3_variant,
            scheduler,
            eager,
            offload,
            lora,
            lora_scale,
            image,
            strength,
            mask,
            control,
            control_model,
            control_scale,
            negative_prompt,
            no_negative,
            expand,
            no_expand,
            expand_backend,
            expand_model,
        } => {
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
                host,
                format,
                no_metadata,
                preview,
                local,
                t5_variant,
                qwen3_variant,
                scheduler,
                eager,
                offload,
                lora,
                lora_scale,
                image,
                strength,
                mask,
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
        } => {
            commands::expand::run(
                &prompt,
                model.as_deref(),
                variations,
                json,
                backend.as_deref(),
                expand_model.as_deref(),
            )
            .await?;
        }
        Commands::Serve {
            port,
            bind,
            models_dir,
            #[cfg(feature = "discord")]
            discord,
            ..
        } => {
            #[cfg(feature = "discord")]
            let discord_enabled = discord;
            #[cfg(not(feature = "discord"))]
            let discord_enabled = false;

            commands::serve::run(port, &bind, models_dir, discord_enabled).await?;
        }
        Commands::Server { action } => match action {
            ServerAction::Start {
                port,
                bind,
                models_dir,
                log_file,
            } => {
                commands::server::run_start(port, &bind, models_dir, log_file).await?;
            }
            ServerAction::Status => {
                commands::server::run_status().await?;
            }
            ServerAction::Stop => {
                commands::server::run_stop().await?;
            }
        },
        Commands::Pull { model, skip_verify } => {
            let opts = mold_core::download::PullOptions { skip_verify };
            commands::pull::run(&model, &opts).await?;
        }
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
        Commands::Config { action } => match action {
            ConfigAction::List { json } => commands::config::run_list(json)?,
            ConfigAction::Get { key, raw } => commands::config::run_get(&key, raw)?,
            ConfigAction::Set { key, value } => commands::config::run_set(&key, &value)?,
            ConfigAction::Path => commands::config::run_path()?,
            ConfigAction::Edit => commands::config::run_edit()?,
        },
        Commands::Unload => {
            commands::unload::run().await?;
        }
        Commands::Ps => {
            commands::ps::run().await?;
        }
        Commands::Version => {
            println!("mold {}", mold_core::build_info::version_string());
        }
        #[cfg(feature = "discord")]
        Commands::Discord => {
            commands::discord::run().await?;
        }
        #[cfg(feature = "tui")]
        Commands::Tui { host, local } => {
            mold_tui::run_tui(host, local).await?;
        }
        Commands::Completions { shell } => {
            generate_completions(&shell)?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    /// Parse CLI args from a vector (simulates command-line invocation).
    fn parse(args: &[&str]) -> Cli {
        Cli::parse_from(std::iter::once("mold").chain(args.iter().copied()))
    }

    /// Try to parse CLI args, returning the clap error on failure.
    fn try_parse(args: &[&str]) -> Result<Cli, clap::Error> {
        Cli::try_parse_from(std::iter::once("mold").chain(args.iter().copied()))
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
    fn run_seed_before_prompt() {
        let cli = parse(&["run", "model", "--seed", "42", "a", "cat"]);
        match cli.command {
            Commands::Run { seed, .. } => assert_eq!(seed, Some(42)),
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
    fn run_format_jpeg() {
        let cli = parse(&["run", "model", "test", "--format", "jpeg"]);
        match cli.command {
            Commands::Run { format, .. } => assert_eq!(format, OutputFormat::Jpeg),
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
                assert_eq!(format, OutputFormat::Jpeg);
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
                assert_eq!(format, OutputFormat::Png);
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
    fn run_batch_17_rejected() {
        let result = try_parse(&["run", "model", "test", "--batch", "17"]);
        assert!(result.is_err());
    }

    #[test]
    fn run_format_invalid_rejected() {
        let result = try_parse(&["run", "model", "test", "--format", "webp"]);
        assert!(result.is_err());
    }

    #[test]
    fn run_format_jpg_alias() {
        let cli = parse(&["run", "model", "test", "--format", "jpg"]);
        match cli.command {
            Commands::Run { format, .. } => assert_eq!(format, OutputFormat::Jpeg),
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
            Commands::Run { image, .. } => assert_eq!(image.as_deref(), Some("photo.png")),
            _ => panic!("expected Run"),
        }
    }

    #[test]
    fn run_image_stdin() {
        let cli = parse(&["run", "model", "test", "--image", "-"]);
        match cli.command {
            Commands::Run { image, .. } => assert_eq!(image.as_deref(), Some("-")),
            _ => panic!("expected Run"),
        }
    }

    #[test]
    fn run_image_short_flag() {
        let cli = parse(&["run", "model", "test", "-i", "input.jpg"]);
        match cli.command {
            Commands::Run { image, .. } => assert_eq!(image.as_deref(), Some("input.jpg")),
            _ => panic!("expected Run"),
        }
    }

    #[test]
    fn run_strength_flag() {
        let cli = parse(&["run", "model", "test", "--strength", "0.5"]);
        match cli.command {
            Commands::Run { strength, .. } => assert_eq!(strength, 0.5),
            _ => panic!("expected Run"),
        }
    }

    #[test]
    fn run_strength_default() {
        let cli = parse(&["run", "model", "test"]);
        match cli.command {
            Commands::Run { strength, .. } => assert_eq!(strength, 0.75),
            _ => panic!("expected Run"),
        }
    }

    #[test]
    fn run_image_defaults_none() {
        let cli = parse(&["run", "model", "test"]);
        match cli.command {
            Commands::Run { image, .. } => assert!(image.is_none()),
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
}
