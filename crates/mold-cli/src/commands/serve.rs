use anyhow::Result;
use colored::Colorize;
use mold_core::Config;
use std::path::PathBuf;

pub async fn run(port: u16, bind: &str, models_dir: Option<String>) -> Result<()> {
    let config = Config::load_or_default();

    let models_path = match models_dir {
        Some(dir) => PathBuf::from(dir),
        None => config.resolved_models_dir(),
    };

    // Ensure models directory exists
    std::fs::create_dir_all(&models_path)?;

    println!("{} Starting mold server on {}:{}", "●".green(), bind, port,);
    println!(
        "{} Models directory: {}",
        "●".green(),
        models_path.display(),
    );

    mold_server::run_server(bind, port, models_path).await
}
