//! A fixed, root-owned boot policy. Tests use only private temporary directories.

use anyhow::{bail, Context, Result};
use std::fs::{File, OpenOptions, Permissions};
use std::io::{Read, Write};
use std::os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};

pub const LABEL: &str = "io.utensils.mold.metal-memory";
pub const FILE_NAME: &str = "io.utensils.mold.metal-memory.plist";
pub const DIRECTORY: &str = "/Library/LaunchDaemons";

pub fn render_plist(mib: u32) -> String {
    format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<!-- Managed by mold: metal-memory-v1 -->
<plist version="1.0"><dict>
<key>Label</key><string>{LABEL}</string>
<key>ProgramArguments</key><array>
<string>/usr/sbin/sysctl</string><string>-w</string>
<string>iogpu.wired_limit_mb={mib}</string>
</array>
<key>RunAtLoad</key><true/>
</dict></plist>
"#
    )
}

pub struct Store {
    directory: PathBuf,
    owner: u32,
    _lock: File,
}

impl Store {
    pub fn open(directory: &Path, owner: u32) -> Result<Self> {
        trusted_directory(directory, owner)?;
        let path = directory.join(".io.utensils.mold.metal-memory.lock");
        if let Ok(meta) = std::fs::symlink_metadata(&path) {
            trusted_file(&meta, owner)?;
        }
        let lock = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW)
            .open(&path)?;
        trusted_file(&lock.metadata()?, owner)?;
        if lock.metadata()?.permissions().mode() & 0o077 != 0 {
            bail!("administration lock must have mode 0600")
        }
        lock.try_lock().map_err(|error| {
            anyhow::anyhow!("another Metal-memory administrator holds the lock: {error}")
        })?;
        Ok(Self {
            directory: directory.into(),
            owner,
            _lock: lock,
        })
    }
    pub fn read(&self) -> Result<Option<u32>> {
        read_policy(&self.directory, self.owner)
    }
    pub fn replace(&self, value: Option<u32>, expected: Option<u32>) -> Result<()> {
        trusted_directory(&self.directory, self.owner)?;
        if self.read()? != expected {
            bail!("boot policy changed since inspection; left untouched")
        }
        let path = self.directory.join(FILE_NAME);
        if let Some(value) = value {
            if value == 0 {
                bail!("automatic mode removes the boot policy; it is never persisted as zero")
            }
            let mut staged = tempfile::NamedTempFile::new_in(&self.directory)?;
            staged
                .as_file()
                .set_permissions(Permissions::from_mode(0o644))?;
            staged.write_all(render_plist(value).as_bytes())?;
            staged.as_file().sync_all()?;
            #[cfg(target_os = "macos")]
            {
                let validation = std::process::Command::new("/usr/bin/plutil")
                    .arg("-lint")
                    .arg(staged.path())
                    .output()?;
                if !validation.status.success() {
                    bail!("generated boot policy failed plutil validation")
                }
            }
            if self.read()? != expected {
                bail!("boot policy changed while staging; left untouched")
            }
            staged
                .persist(&path)
                .context("cannot atomically install boot policy")?;
        } else if expected.is_some() {
            std::fs::remove_file(&path).context("cannot remove owned boot policy")?;
        }
        File::open(&self.directory)?
            .sync_all()
            .context("boot policy changed but directory sync failed")?;
        if self.read()? != value {
            bail!("boot policy verification failed after update")
        }
        Ok(())
    }
}

pub fn read_policy(directory: &Path, owner: u32) -> Result<Option<u32>> {
    trusted_directory(directory, owner)?;
    let path = directory.join(FILE_NAME);
    let meta = match std::fs::symlink_metadata(&path) {
        Ok(meta) => meta,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    trusted_file(&meta, owner)?;
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW)
        .open(&path)?;
    trusted_file(&file.metadata()?, owner)?;
    let mut content = String::new();
    file.take(8193).read_to_string(&mut content)?;
    let value = content
        .split("<string>iogpu.wired_limit_mb=")
        .nth(1)
        .and_then(|rest| rest.split("</string>").next())
        .and_then(|value| value.parse::<u32>().ok())
        .filter(|value| *value > 0);
    match value {
        Some(value) if content.len() <= 8192 && content == render_plist(value) => Ok(Some(value)),
        _ => bail!(
            "{} is not Mold's exact owned boot policy; refusing to modify it",
            path.display()
        ),
    }
}

fn trusted_directory(directory: &Path, owner: u32) -> Result<()> {
    let meta = std::fs::symlink_metadata(directory)?;
    if !meta.is_dir() || meta.uid() != owner || meta.permissions().mode() & 0o022 != 0 {
        bail!("boot-policy directory must be an owned directory without group/other write access or symlinks")
    }
    Ok(())
}

fn trusted_file(meta: &std::fs::Metadata, owner: u32) -> Result<()> {
    if !meta.is_file()
        || meta.uid() != owner
        || meta.permissions().mode() & 0o022 != 0
        || (owner == 0 && meta.gid() != 0)
    {
        bail!("boot-policy file must be an owned regular file without group/other write access or symlinks")
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::{symlink, MetadataExt, PermissionsExt};

    fn fixture() -> (tempfile::TempDir, u32) {
        let dir = tempfile::tempdir().unwrap();
        let owner = std::fs::metadata(dir.path()).unwrap().uid();
        (dir, owner)
    }

    #[test]
    fn metal_memory_persistence_roundtrip_reset_and_permissions() {
        let (dir, uid) = fixture();
        let store = Store::open(dir.path(), uid).unwrap();
        assert_eq!(store.read().unwrap(), None);
        store.replace(Some(16384), None).unwrap();
        assert_eq!(store.read().unwrap(), Some(16384));
        assert_eq!(
            std::fs::metadata(dir.path().join(FILE_NAME))
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o644
        );
        store.replace(None, Some(16384)).unwrap();
        assert!(!dir.path().join(FILE_NAME).exists());
    }

    #[test]
    fn metal_memory_persistence_refuses_foreign_files_and_symlinks() {
        let (dir, uid) = fixture();
        let path = dir.path().join(FILE_NAME);
        std::fs::write(&path, "another administrator owns this").unwrap();
        let store = Store::open(dir.path(), uid).unwrap();
        assert!(store.replace(Some(16384), None).is_err());
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "another administrator owns this"
        );
        std::fs::remove_file(&path).unwrap();
        let target = dir.path().join("untouched");
        std::fs::write(&target, "untouched").unwrap();
        symlink(&target, &path).unwrap();
        assert!(store.replace(Some(16384), None).is_err());
        assert_eq!(std::fs::read_to_string(target).unwrap(), "untouched");
    }

    #[test]
    fn metal_memory_persistence_serializes_writers_and_detects_drift() {
        let (dir, uid) = fixture();
        let store = Store::open(dir.path(), uid).unwrap();
        assert!(Store::open(dir.path(), uid).is_err());
        store.replace(Some(16384), None).unwrap();
        assert!(store.replace(Some(12288), None).is_err());
        assert_eq!(store.read().unwrap(), Some(16384));
    }

    #[test]
    fn metal_memory_persistence_refuses_untrusted_directory() {
        let (dir, uid) = fixture();
        std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(0o777)).unwrap();
        assert!(Store::open(dir.path(), uid).is_err());
    }

    #[test]
    fn metal_memory_persistence_only_runs_fixed_sysctl_at_boot() {
        let plist = render_plist(16384);
        assert!(plist.contains("<string>/usr/sbin/sysctl</string>"));
        assert!(plist.contains("<string>iogpu.wired_limit_mb=16384</string>"));
        assert!(plist.contains("<key>RunAtLoad</key><true/>"));
        assert!(!plist.contains("KeepAlive"));
        #[cfg(target_os = "macos")]
        {
            let (dir, _) = fixture();
            let path = dir.path().join("test.plist");
            std::fs::write(&path, plist).unwrap();
            assert!(std::process::Command::new("/usr/bin/plutil")
                .arg("-lint")
                .arg(path)
                .status()
                .unwrap()
                .success());
        }
    }
}
