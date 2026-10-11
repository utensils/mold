use clap::Parser;
#[derive(Parser)]
#[command(
    name = "mold-relay",
    about = "End-to-end encrypted outbound Mold host connector"
)]
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
