//! Server identity creation; connector uses the shared read-only helper.
use anyhow::Result;
use mold_secure::Identity;
use std::path::Path;
pub fn load_host_identity(mold_home: &Path) -> Result<Identity> {
    mold_secure::identity_file::load_or_create_host_identity(mold_home).map_err(Into::into)
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    #[test]
    fn persistent_host_identity_never_changes_on_reload() {
        let temp = tempfile::tempdir().unwrap();
        let first = load_host_identity(temp.path()).unwrap();
        let second = load_host_identity(temp.path()).unwrap();
        assert_eq!(first.binding(), second.binding());
    }
    #[cfg(unix)]
    #[test]
    fn insecure_permissions_and_symlinks_fail_without_key_replacement() {
        use std::os::unix::fs::{symlink, PermissionsExt};
        let temp = tempfile::tempdir().unwrap();
        load_host_identity(temp.path()).unwrap();
        let key = temp.path().join("secure/host.identity");
        let original = fs::read(&key).unwrap();
        fs::set_permissions(&key, fs::Permissions::from_mode(0o644)).unwrap();
        assert!(load_host_identity(temp.path()).is_err());
        assert_eq!(fs::read(&key).unwrap(), original);
        fs::remove_file(&key).unwrap();
        let target = temp.path().join("target");
        fs::write(&target, &original).unwrap();
        symlink(&target, &key).unwrap();
        assert!(load_host_identity(temp.path()).is_err());
        assert_eq!(fs::read(target).unwrap(), original);
    }
}
