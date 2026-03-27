use anyhow::Result;
use mold_core::Config;
use std::path::PathBuf;

use crate::theme;

pub async fn run(port: u16, bind: &str, models_dir: Option<String>, discord: bool) -> Result<()> {
    let config = Config::load_or_default();

    let models_path = match models_dir {
        Some(dir) => PathBuf::from(dir),
        None => config.resolved_models_dir(),
    };

    // Ensure models directory exists
    std::fs::create_dir_all(&models_path)?;

    println!(
        "{} Starting mold server on {}:{}",
        theme::icon_ok(),
        bind,
        port,
    );
    println!(
        "{} Models directory: {}",
        theme::icon_ok(),
        models_path.display(),
    );

    // Optionally spawn the Discord bot alongside the server.
    #[cfg(feature = "discord")]
    if discord {
        // Point the bot at this server if MOLD_HOST isn't already set.
        if std::env::var("MOLD_HOST").is_err() {
            std::env::set_var("MOLD_HOST", format!("http://{}:{}", bind, port));
        }
        println!("{} Discord bot enabled", theme::icon_ok());
        tokio::spawn(async {
            // Brief delay so the HTTP listener is ready before the bot connects.
            tokio::time::sleep(std::time::Duration::from_secs(1)).await;
            if let Err(e) = mold_discord::run().await {
                eprintln!("{} Discord bot error: {e:#}", crate::theme::prefix_error());
            }
        });
    }
    #[cfg(not(feature = "discord"))]
    if discord {
        anyhow::bail!(
            "Discord support is not compiled in. Rebuild with --features discord to enable."
        );
    }

    mold_server::run_server(bind, port, models_path).await
}
