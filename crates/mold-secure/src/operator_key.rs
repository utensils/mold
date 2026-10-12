//! One owner credential per Mold home, independent of the hosting application.
use crate::identity_file::{read_private_file, write_private_file};
use std::{io, path::Path};
use zeroize::Zeroizing;

/// Resolve one client header from the explicit server configuration. The caller
/// must retain the original configuration for the server's full accepted set.
pub fn client_override(raw: &str) -> io::Result<Zeroizing<String>> {
    use std::io::Read;
    let contents = if let Some(path) = raw.strip_prefix('@') {
        let mut options = std::fs::OpenOptions::new();
        options.read(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.custom_flags(libc::O_NONBLOCK | libc::O_CLOEXEC);
        }
        let file = options
            .open(path)
            .map_err(|_| io::Error::other("cannot read explicit operator key file"))?;
        if !file.metadata()?.is_file() {
            return Err(io::Error::other(
                "explicit operator key file is not a regular file",
            ));
        }
        let mut contents = Zeroizing::new(String::new());
        file.take(65537).read_to_string(&mut contents)?;
        if contents.len() > 65536 {
            return Err(io::Error::other(
                "explicit operator key file exceeds its size bound",
            ));
        }
        Some(contents)
    } else {
        None
    };
    let key = if let Some(contents) = &contents {
        contents
            .lines()
            .map(str::trim)
            .find(|line| !line.is_empty() && !line.starts_with('#'))
    } else {
        raw.split(',').map(str::trim).find(|key| !key.is_empty())
    }
    .ok_or_else(|| io::Error::other("explicit operator configuration contains no keys"))?;
    if key.len() > MAX_KEY_BYTES || !key.bytes().all(|b| (0x20..=0x7e).contains(&b)) {
        return Err(io::Error::other(
            "explicit operator key is not a valid client header",
        ));
    }
    Ok(Zeroizing::new(key.to_owned()))
}

/// Keep the complete explicit server key set while the local client uses one
/// normalized member. An empty environment value is equivalent to absent.
pub fn server_configuration<'a>(explicit: Option<&'a str>, client_key: &'a str) -> &'a str {
    explicit
        .filter(|value| !value.is_empty())
        .unwrap_or(client_key)
}

pub const MAX_KEY_BYTES: usize = 4096;
fn validate(value: &str) -> io::Result<()> {
    if value.is_empty()
        || value.len() > MAX_KEY_BYTES
        || !value
            .bytes()
            .all(|b| (0x21..=0x7e).contains(&b) && b != b',')
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid per-home operator credential",
        ));
    }
    Ok(())
}
/// Read only: an untouched headless home remains without configured auth.
/// Invalid or permissive existing state is an error, never absence.
pub fn read(home: &Path) -> io::Result<Option<Zeroizing<String>>> {
    let path = home.join("secure/operator.key");
    let bytes = match read_private_file(&path, MAX_KEY_BYTES) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error),
    };
    let value = std::str::from_utf8(&bytes).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid per-home operator credential",
        )
    })?;
    validate(value)?;
    Ok(Some(Zeroizing::new(value.to_owned())))
}
fn read_after_install(home: &Path) -> io::Result<Option<Zeroizing<String>>> {
    // A competing POSIX create-only install briefly has two hard links before
    // removing its temporary name. Retain strict validation and bound the wait.
    for _ in 0..5 {
        match read(home) {
            Err(error) if error.kind() == io::ErrorKind::PermissionDenied => {
                std::thread::sleep(std::time::Duration::from_millis(5))
            }
            result => return result,
        }
    }
    read(home)
}
/// Adopt an existing shell key once, or generate an owner credential. The first
/// durable writer wins; later shells ignore their obsolete local candidates.
/// Explicit operator environment overrides are handled by the caller instead.
pub fn resolve(home: &Path, legacy_candidate: Option<&str>) -> io::Result<Zeroizing<String>> {
    if let Some(value) = read_after_install(home)? {
        return Ok(value);
    }
    let value = if let Some(candidate) = legacy_candidate {
        validate(candidate)?;
        Zeroizing::new(candidate.to_owned())
    } else {
        let mut random = Zeroizing::new([0u8; 32]);
        getrandom::fill(&mut *random)
            .map_err(|_| io::Error::other("secure random source unavailable"))?;
        Zeroizing::new(format!("mold_operator_{}", hex::encode(*random)))
    };
    std::fs::create_dir_all(home)?;
    let home = home.canonicalize()?;
    match write_private_file(&home.join("secure/operator.key"), value.as_bytes(), false) {
        Ok(()) => {}
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
        Err(error) => return Err(error),
    }
    read_after_install(&home)?
        .ok_or_else(|| io::Error::other("per-home operator credential disappeared"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_file_and_csv_choose_an_actual_client_key() {
        let home = tempfile::tempdir().unwrap();
        let path = home.path().join("operator-keys");
        std::fs::write(&path, b"# owner keys\n\n first-key \nsecond-key\n").unwrap();
        assert_eq!(
            client_override(&format!("@{}", path.display()))
                .unwrap()
                .as_str(),
            "first-key"
        );
        assert_eq!(
            client_override(" , first-key,second-key ")
                .unwrap()
                .as_str(),
            "first-key"
        );
        assert!(client_override(", ,").is_err());
        assert!(!home.path().join("secure").exists());
        assert_eq!(server_configuration(Some("one,two"), "one"), "one,two");
        assert_eq!(
            server_configuration(Some("@/operator-keys"), "one"),
            "@/operator-keys"
        );
        assert_eq!(server_configuration(None, "shared-key"), "shared-key");
    }

    #[test]
    fn same_home_adopts_one_shell_key_without_changing_identity_or_database() {
        let home = tempfile::tempdir().unwrap();
        let identity = crate::identity_file::load_or_create_host_identity(home.path()).unwrap();
        std::fs::write(home.path().join("mold.db"), b"existing paired grants").unwrap();
        let tauri = resolve(home.path(), Some("existing-tauri-key")).unwrap();
        let native = resolve(home.path(), Some("different-native-key")).unwrap();
        assert_eq!(tauri.as_str(), native.as_str());
        assert_eq!(read(home.path()).unwrap().unwrap().as_str(), tauri.as_str());
        assert_eq!(
            crate::identity_file::read_host_identity(&home.path().join("secure/host.identity"))
                .unwrap()
                .to_bytes(),
            identity.to_bytes()
        );
        assert_eq!(
            std::fs::read(home.path().join("mold.db")).unwrap(),
            b"existing paired grants"
        );
    }

    #[test]
    fn absent_read_is_side_effect_free_and_distinct_homes_are_isolated() {
        let one = tempfile::tempdir().unwrap();
        let two = tempfile::tempdir().unwrap();
        assert!(read(one.path()).unwrap().is_none());
        assert!(!one.path().join("secure").exists());
        assert_ne!(
            resolve(one.path(), None).unwrap().as_str(),
            resolve(two.path(), None).unwrap().as_str()
        );
    }

    #[test]
    fn malformed_existing_key_is_not_replaced() {
        let home = tempfile::tempdir().unwrap();
        let path = home.path().join("secure/operator.key");
        crate::identity_file::write_private_file(&path, b"bad\nkey", false).unwrap();
        assert!(resolve(home.path(), Some("legacy-key")).is_err());
        assert_eq!(
            &*crate::identity_file::read_private_file(&path, 4096).unwrap(),
            b"bad\nkey"
        );
    }

    #[test]
    fn native_first_and_tauri_first_preserve_the_first_home_authority() {
        for (first, second) in [("native-key", "tauri-key"), ("tauri-key", "native-key")] {
            let home = tempfile::tempdir().unwrap();
            assert_eq!(resolve(home.path(), Some(first)).unwrap().as_str(), first);
            assert_eq!(resolve(home.path(), Some(second)).unwrap().as_str(), first);
        }
    }

    #[cfg(unix)]
    #[test]
    fn permissive_key_is_refused_without_repair_or_rotation() {
        use std::os::unix::fs::PermissionsExt;
        let home = tempfile::tempdir().unwrap();
        resolve(home.path(), Some("existing-key")).unwrap();
        let path = home.path().join("secure/operator.key");
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
        assert!(read(home.path()).is_err());
        assert!(resolve(home.path(), Some("replacement-key")).is_err());
        assert_eq!(std::fs::read_to_string(path).unwrap(), "existing-key");
    }

    #[test]
    fn concurrent_shells_converge_without_rotating_a_winner() {
        let home = tempfile::tempdir().unwrap();
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(8));
        let values = std::thread::scope(|scope| {
            let workers = (0..8)
                .map(|i| {
                    let barrier = barrier.clone();
                    let home = home.path();
                    scope.spawn(move || {
                        barrier.wait();
                        resolve(home, Some(&format!("shell-{i}-key"))).unwrap()
                    })
                })
                .collect::<Vec<_>>();
            workers
                .into_iter()
                .map(|w| w.join().unwrap())
                .collect::<Vec<_>>()
        });
        assert!(values.iter().all(|v| **v == *values[0]));
    }
}
