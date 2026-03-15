mod commands;
mod tui;

use clap::{CommandFactory, Parser, Subcommand};
use clap_complete::engine::ArgValueCandidates;

#[derive(Parser)]
#[command(
    name = "mold",
    about = "AI image generation — like ollama, but for diffusion models"
)]
#[command(version, propagate_version = true)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run image generation or interactive TUI
    ///
    /// If PROMPT is provided, generates images (one-shot).
    /// If PROMPT is omitted, opens the interactive TUI.
    /// First positional arg is treated as MODEL if it matches a known model name.
    Run {
        /// Model name (e.g. flux-dev:q4, flux-schnell)
        #[arg(add = ArgValueCandidates::new(commands::run::complete_model_name))]
        model_or_prompt: Option<String>,

        /// Prompt text (remaining words after model)
        #[arg(trailing_var_arg = true)]
        prompt_rest: Vec<String>,

        /// Output file path
        #[arg(short, long)]
        output: Option<String>,

        /// Image width — defaults to model config value
        #[arg(long)]
        width: Option<u32>,

        /// Image height — defaults to model config value
        #[arg(long)]
        height: Option<u32>,

        /// Number of inference steps — defaults to model config value
        #[arg(long)]
        steps: Option<u32>,

        /// Guidance scale — defaults to model config value
        #[arg(long)]
        guidance: Option<f64>,

        /// Random seed
        #[arg(long)]
        seed: Option<u64>,

        /// Number of images to generate
        #[arg(long, default_value = "1")]
        batch: u32,

        /// Override MOLD_HOST
        #[arg(long)]
        host: Option<String>,

        /// Output format
        #[arg(long, default_value = "png")]
        format: String,

        /// Skip server and run inference locally (requires GPU features)
        #[arg(long)]
        local: bool,
    },

    /// Start the inference server
    Serve {
        /// Server port
        #[arg(long, default_value = "7680")]
        port: u16,

        /// Bind address
        #[arg(long, default_value = "0.0.0.0")]
        bind: String,

        /// Models directory
        #[arg(long)]
        models_dir: Option<String>,
    },

    /// Download model weights from HuggingFace
    Pull {
        /// Model name to download
        #[arg(add = ArgValueCandidates::new(commands::run::complete_model_name))]
        model: String,
    },

    /// List locally available models
    List,

    /// Show server status and loaded models
    Ps,

    /// Show version information
    Version,

    /// Generate shell completions
    Completions {
        /// Shell to generate completions for
        #[arg(value_enum)]
        shell: clap_complete::Shell,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    clap_complete::CompleteEnv::with_factory(Cli::command).complete();

    let cli = Cli::parse();

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
            local,
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
                local,
            )
            .await?;
        }
        Commands::Serve {
            port,
            bind,
            models_dir,
        } => {
            commands::serve::run(port, &bind, models_dir).await?;
        }
        Commands::Pull { model } => {
            commands::pull::run(&model).await?;
        }
        Commands::List => {
            commands::list::run().await?;
        }
        Commands::Ps => {
            commands::ps::run().await?;
        }
        Commands::Version => {
            println!("mold {}", env!("CARGO_PKG_VERSION"));
        }
        Commands::Completions { shell } => {
            clap_complete::generate(shell, &mut Cli::command(), "mold", &mut std::io::stdout());
        }
    }

    Ok(())
}
