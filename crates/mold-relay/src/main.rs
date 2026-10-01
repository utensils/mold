use clap::Parser;
#[derive(Parser)]
#[command(name = "mold-relay", about = "Optional outbound HTTPS access relay")]
struct Cli {
    #[command(subcommand)]
    command: mold_relay::RelayAction,
}
#[tokio::main]
async fn main() {
    if let Err(error) = Cli::parse().command.run().await {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
