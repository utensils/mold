mod commands;
mod tui;

use clap::{Parser, Subcommand};

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
    /// Generate images from a text prompt
    Generate {
        /// The text prompt to generate from
        prompt: String,

        /// Model to use
        #[arg(short, long, default_value = "flux-schnell")]
        model: String,

        /// Output file path
        #[arg(short, long)]
        output: Option<String>,

        /// Image width (max 768 — 1024+ causes VAE OOM on RTX 4090 with current GGUF models)
        #[arg(long, default_value = "768")]
        width: u32,

        /// Image height (max 768 — 1024+ causes VAE OOM on RTX 4090 with current GGUF models)
        #[arg(long, default_value = "768")]
        height: u32,

        /// Number of inference steps
        #[arg(long)]
        steps: Option<u32>,

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
        model: String,
    },

    /// List locally available models
    List,

    /// Show server status and loaded models
    Ps,

    /// Open interactive TUI session
    Run {
        /// Model to use
        #[arg(default_value = "flux-schnell")]
        model: String,
    },

    /// Show version information
    Version,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Generate {
            prompt,
            model,
            output,
            width,
            height,
            steps,
            seed,
            batch,
            host,
            format,
        } => {
            commands::generate::run(
                &prompt, &model, output, width, height, steps, seed, batch, host, &format,
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
        Commands::Run { model } => {
            tui::run(&model).await?;
        }
        Commands::Version => {
            println!("mold {}", env!("CARGO_PKG_VERSION"));
        }
    }

    Ok(())
}
