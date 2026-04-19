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

        /// Log output format
        #[arg(long, default_value = "json")]
        log_format: LogFormat,

        /// Write logs to file (~/.mold/logs/)
        #[arg(long)]
        log_file: bool,
    },
}

#[derive(Clone, clap::ValueEnum)]
enum LogFormat {
    Text,
    Json,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    match args.command {
        Command::Serve {
            port,
            bind,
            models_dir,
            log_format,
            log_file,
        } => {
            let config = mold_core::Config::load_or_default();
            let log_dir = config.resolved_log_dir();
            let _log_guard = mold_server::logging::init_tracing(
                log_file,
                matches!(log_format, LogFormat::Json),
                &config.logging,
                "info",
                log_dir,
            );

            let models_path = models_dir
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("."));

            let gpu_selection = config.gpu_selection();
            let queue_size = config.queue_size();
            mold_server::run_server(&bind, port, models_path, gpu_selection, queue_size).await?;
        }
    }

    Ok(())
}
