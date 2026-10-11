//! Owner-private local bootstrap fixture for testing the real headless binary.
fn main() -> anyhow::Result<()> {
    let directory = std::env::args_os()
        .nth(1)
        .map(std::path::PathBuf::from)
        .ok_or_else(|| anyhow::anyhow!("provide a private fixture directory"))?;
    let prepared = mold_secure_client::prepare_bootstrap("Headless bootstrap UAT", "cli")?;
    let mut profile = prepared.profile("https://bootstrap-uat.invalid")?;
    profile.websocket = "ws://127.0.0.1:19681/_mold/secure".into();
    profile.validate()?;
    for (name, bytes) in [
        (
            "environment.json",
            zeroize::Zeroizing::new(serde_json::to_vec(&prepared.environment())?),
        ),
        (
            "config.json",
            zeroize::Zeroizing::new(serde_json::to_vec(&profile)?),
        ),
    ] {
        mold_secure_client::identity_file::write_private_file(&directory.join(name), &bytes, true)?;
    }
    mold_secure_client::save_pairing(&profile)?;
    println!(
        "Private local bootstrap fixture written to {}",
        directory.display()
    );
    Ok(())
}
