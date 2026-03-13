use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "mold-server", about = "mold inference server")]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(clap::Subcommand)]
enum Command {
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
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_env("MOLD_LOG")
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();

    match args.command {
        Command::Serve {
            port,
            bind,
            models_dir,
        } => {
            let models_path = models_dir
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("."));

            mold_server::run_server(&bind, port, models_path).await?;
        }
    }

    Ok(())
}
