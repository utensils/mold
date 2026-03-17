use anyhow::Result;
use colored::Colorize;
use mold_core::MoldClient;

pub async fn run() -> Result<()> {
    let client = MoldClient::from_env();

    match client.unload_model().await {
        Ok(body) => {
            println!("{} {}", "●".green(), body);
            Ok(())
        }
        Err(e) => {
            if MoldClient::is_connection_error(&e) {
                eprintln!(
                    "{} cannot connect to mold server at {}",
                    "error:".red().bold(),
                    client.host()
                );
                eprintln!(
                    "  {} start the server with {}",
                    "hint:".dimmed(),
                    "mold serve".bold()
                );
                Err(crate::AlreadyReported.into())
            } else {
                Err(e)
            }
        }
    }
}
