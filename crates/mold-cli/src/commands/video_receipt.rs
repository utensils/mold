//! Private scalar authority for reconciling a video admission after a lost reply.
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct VideoOperationReceipt {
    pub operation_id: String,
    pub origin_sha256: String,
    pub instance_id: String,
    #[serde(default)]
    pub chain: bool,
}
fn path(root: &Path, id: &str) -> Result<PathBuf> {
    ensure!(
        uuid::Uuid::parse_str(id)?.to_string() == id,
        "Operation must be a canonical UUID"
    );
    Ok(root.join(format!("{id}.json")))
}
fn origin_hash(origin: &str) -> String {
    format!("{:x}", Sha256::digest(origin.as_bytes()))
}
fn directory() -> Result<PathBuf> {
    Ok(mold_core::Config::mold_dir()
        .context("No Mold home for the video operation receipt")?
        .join("video-operations"))
}
pub(crate) fn read_optional(id: &str, origin: &str) -> Result<Option<VideoOperationReceipt>> {
    let root = directory()?;
    if !path(&root, id)?.try_exists()? {
        return Ok(None);
    }
    read_at(&root, id, origin).map(Some)
}
pub(crate) fn store_chain(id: &str, origin: &str, instance: &str) -> Result<()> {
    store_kind_at(&directory()?, id, origin, instance, true)
}
pub(crate) fn store(id: &str, origin: &str, instance: &str) -> Result<()> {
    store_at(&directory()?, id, origin, instance)
}
pub(crate) fn read(id: &str, origin: &str) -> Result<VideoOperationReceipt> {
    read_at(&directory()?, id, origin)
}
fn read_at(root: &Path, id: &str, origin: &str) -> Result<VideoOperationReceipt> {
    use std::io::Read;
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW);
    }
    let file=options.open(path(root,id)?).context("This operation has no readable local receipt. Use jobs show on its original machine; do not start a replacement admission")?;
    ensure!(
        file.metadata()?.len() <= 4096,
        "Operation receipt is oversized"
    );
    let mut raw = Vec::new();
    file.take(4097).read_to_end(&mut raw)?;
    ensure!(raw.len() <= 4096, "Operation receipt is oversized");
    let value: VideoOperationReceipt = serde_json::from_slice(&raw)?;
    ensure!(
        value.operation_id == id
            && value.origin_sha256 == origin_hash(origin)
            && !value.instance_id.is_empty(),
        "Operation belongs to another destination; reconnect its original machine"
    );
    Ok(value)
}
fn store_at(root: &Path, id: &str, origin: &str, instance: &str) -> Result<()> {
    store_kind_at(root, id, origin, instance, false)
}
fn store_kind_at(root: &Path, id: &str, origin: &str, instance: &str, chain: bool) -> Result<()> {
    use std::io::Write;
    ensure!(
        !instance.is_empty(),
        "Operation requires a captured installation"
    );
    let target = path(root, id)?;
    if target.exists() {
        let old = read_at(root, id, origin)?;
        ensure!(
            old.instance_id == instance && old.chain == chain,
            "Operation belongs to a replaced installation"
        );
        return Ok(());
    }
    std::fs::create_dir_all(root)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(root, std::fs::Permissions::from_mode(0o700))?;
    }
    let receipt = VideoOperationReceipt {
        operation_id: id.into(),
        origin_sha256: origin_hash(origin),
        instance_id: instance.into(),
        chain,
    };
    let mut file = tempfile::NamedTempFile::new_in(root)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        file.as_file()
            .set_permissions(std::fs::Permissions::from_mode(0o600))?;
    }
    file.write_all(&serde_json::to_vec(&receipt)?)?;
    file.as_file().sync_all()?;
    file.persist_noclobber(target)?;
    #[cfg(unix)]
    std::fs::File::open(root)?.sync_all()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn duration_receipt_is_private_scalar_and_rejects_missing_replaced_or_redirected_owners() {
        let dir = tempfile::tempdir().unwrap();
        let id = uuid::Uuid::new_v4().to_string();
        assert!(read_at(dir.path(), &id, "https://secret-token@example.invalid").is_err());
        store_at(
            dir.path(),
            &id,
            "https://secret-token@example.invalid",
            "original",
        )
        .unwrap();
        let receipt = read_at(dir.path(), &id, "https://secret-token@example.invalid").unwrap();
        assert_eq!(receipt.instance_id, "original");
        assert!(read_at(dir.path(), &id, "https://other.invalid").is_err());
        assert!(store_at(
            dir.path(),
            &id,
            "https://secret-token@example.invalid",
            "replacement"
        )
        .is_err());
        let raw = std::fs::read_to_string(dir.path().join(format!("{id}.json"))).unwrap();
        assert!(!raw.contains("secret-token"));
        assert!(!raw.contains("https:"));
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                std::fs::metadata(dir.path().join(format!("{id}.json")))
                    .unwrap()
                    .permissions()
                    .mode()
                    & 0o777,
                0o600
            );
        }
    }
}
