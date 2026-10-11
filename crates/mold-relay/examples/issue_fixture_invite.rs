//! Refreshes only the deterministic development fixture invitation. Never use
//! this known test identity with a production registration.
use mold_secure::*;
fn main() -> anyhow::Result<()> {
    let path = std::env::args_os()
        .nth(1)
        .map(std::path::PathBuf::from)
        .ok_or_else(|| anyhow::anyhow!("provide the existing private fixture invitation path"))?;
    let bytes = identity_file::read_private_file(&path, 16384)?;
    let mut value: serde_json::Value = serde_json::from_slice(&bytes)?;
    let host = Identity::from_bytes(&[1; 64])?;
    let namespace = namespace(&host.signing_public());
    anyhow::ensure!(
        value["namespace"] == namespace,
        "not the deterministic development fixture"
    );
    let mut invite: Invitation = serde_json::from_value(value["invite"].clone())?;
    invite.id = uuid::Uuid::new_v4().simple().to_string();
    invite.expires_at = mold_relay::secure::unix_now() + 600;
    invite.signature = hex::encode(host.sign(&invite.canonical(&namespace)?));
    value["invite"] = serde_json::to_value(invite)?;
    let encoded = zeroize::Zeroizing::new(serde_json::to_vec_pretty(&value)?);
    identity_file::write_private_file(&path, &encoded, true)?;
    println!("Refreshed private development fixture invitation");
    Ok(())
}
