//! Owner-only native host identity storage, shared by server and connector.
//! A corrupt or permissive existing identity is never silently replaced.
use crate::Identity;
use std::{
    fs::{self, File},
    io::{self, Read, Write},
    path::Path,
};
use zeroize::Zeroizing;
#[cfg(windows)]
mod windows;
fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::PermissionDenied, message)
}
fn crypto(error: crate::Error) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, error)
}
#[cfg(unix)]
fn validate(file: &File, directory: bool) -> io::Result<()> {
    use std::os::unix::fs::{MetadataExt, PermissionsExt};
    let m = file.metadata()?;
    let mode = if directory { 0o700 } else { 0o600 };
    // SAFETY: geteuid has no pointer or lifetime requirements.
    if m.uid() != unsafe { libc::geteuid() }
        || m.permissions().mode() & 0o777 != mode
        || m.is_dir() != directory
        || (!directory && (!m.is_file() || m.nlink() != 1))
    {
        return Err(invalid("host identity requires current-user ownership, private permissions and no extra hard links"));
    }
    Ok(())
}
#[cfg(windows)]
fn validate(file: &File, directory: bool) -> io::Result<()> {
    windows::validate(file, directory)
}
#[cfg(unix)]
fn open(path: &Path, directory: bool) -> io::Result<File> {
    use std::os::unix::fs::OpenOptionsExt;
    fs::OpenOptions::new()
        .read(true)
        .custom_flags(
            libc::O_NOFOLLOW
                | libc::O_CLOEXEC
                | libc::O_NONBLOCK
                | if directory { libc::O_DIRECTORY } else { 0 },
        )
        .open(path)
}
#[cfg(windows)]
fn open(path: &Path, directory: bool) -> io::Result<File> {
    windows::open(path, directory)
}
#[cfg(unix)]
fn create_directory(path: &Path) -> io::Result<()> {
    use std::os::unix::fs::DirBuilderExt;
    fs::DirBuilder::new().mode(0o700).create(path)
}
#[cfg(windows)]
fn create_directory(path: &Path) -> io::Result<()> {
    windows::create_directory(path)
}
#[cfg(unix)]
fn create_file(path: &Path) -> io::Result<File> {
    use std::os::unix::fs::OpenOptionsExt;
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(path)
}
#[cfg(windows)]
fn create_file(path: &Path) -> io::Result<File> {
    windows::create_file(path)
}

/// Create missing private directories without creating a host identity. Existing
/// target directories must already be private; never repair permissions silently.
pub fn ensure_private_directory(path: &Path) -> io::Result<()> {
    match create_directory(path) {
        Ok(()) => {}
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            let parent = path
                .parent()
                .filter(|p| !p.as_os_str().is_empty())
                .ok_or_else(|| invalid("private directory has no existing ancestor"))?;
            ensure_private_directory(parent)?;
            match create_directory(path) {
                Ok(()) => {}
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
                Err(error) => return Err(error),
            }
        }
        Err(error) => return Err(error),
    }
    validate(&open(path, true)?, true)
}

/// Bounded read of owner-only client state. No file or directory is created.
pub fn read_private_file(path: &Path, max_bytes: usize) -> io::Result<Zeroizing<Vec<u8>>> {
    let parent = path
        .parent()
        .ok_or_else(|| invalid("private file has no parent"))?;
    let directory = open(parent, true)?;
    validate(&directory, true)?;
    let file = open(path, false)?;
    validate(&file, false)?;
    if file.metadata()?.len() > max_bytes as u64 {
        return Err(invalid("private state exceeds its size bound"));
    }
    let mut bytes = Zeroizing::new(Vec::new());
    file.take(max_bytes as u64 + 1).read_to_end(&mut bytes)?;
    if bytes.len() > max_bytes {
        return Err(invalid("private state exceeds its size bound"));
    }
    Ok(bytes)
}
/// Atomic owner-only client state writes. Replacement is explicit and validates
/// any existing destination first. Caller chooses a private subdirectory.
pub fn write_private_file(path: &Path, bytes: &[u8], replace: bool) -> io::Result<()> {
    if bytes.len() > 65536 {
        return Err(invalid("private state exceeds its size bound"));
    }
    let directory = path
        .parent()
        .ok_or_else(|| invalid("private file has no parent"))?;
    ensure_private_directory(directory)?;
    let parent = open(directory, true)?;
    validate(&parent, true)?;
    match fs::symlink_metadata(path) {
        Ok(_) => {
            let old = open(path, false)?;
            validate(&old, false)?;
            if !replace {
                return Err(io::Error::new(
                    io::ErrorKind::AlreadyExists,
                    "private state already exists",
                ));
            }
        }
        Err(e) if e.kind() == io::ErrorKind::NotFound => {}
        Err(e) => return Err(e),
    }
    let mut nonce = [0; 16];
    getrandom::fill(&mut nonce)
        .map_err(|_| io::Error::other("secure random source unavailable"))?;
    let temporary = directory.join(format!(".state-{}", hex::encode(nonce)));
    let mut file = create_file(&temporary)?;
    let installed = (|| -> io::Result<()> {
        file.write_all(bytes)?;
        file.sync_all()?;
        drop(file);
        if replace {
            fs::rename(&temporary, path)
        } else {
            fs::hard_link(&temporary, path)
        }
    })();
    let cleanup = match fs::remove_file(&temporary) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e),
    };
    installed?;
    cleanup?;
    Ok(())
}
/// Read-only connector API. Rejects symlinks/reparse points before reading keys.
pub fn read_host_identity(path: &Path) -> io::Result<Identity> {
    let bytes = read_private_file(path, 64)?;
    if bytes.len() != 64 {
        return Err(invalid(
            "host identity has invalid length; refusing to replace its pin",
        ));
    }
    Identity::from_bytes(&bytes).map_err(crypto)
}
/// Server-only creation API. New identities are installed atomically without
/// replacing a winner from another process. Existing keys always use read validation.
pub fn load_or_create_host_identity(home: &Path) -> io::Result<Identity> {
    fs::create_dir_all(home)?;
    let directory = home.canonicalize()?.join("secure");
    match create_directory(&directory) {
        Ok(()) => {}
        Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {}
        Err(e) => return Err(e),
    };
    let parent = open(&directory, true)?;
    validate(&parent, true)?;
    let path = directory.join("host.identity");
    if fs::symlink_metadata(&path).is_err_and(|e| e.kind() == io::ErrorKind::NotFound) {
        let identity = Identity::generate().map_err(crypto)?;
        let mut nonce = [0; 16];
        getrandom::fill(&mut nonce)
            .map_err(|_| io::Error::other("secure random source unavailable"))?;
        let temporary = directory.join(format!(".identity-{}", hex::encode(nonce)));
        let mut file = create_file(&temporary)?;
        let secret = Zeroizing::new(identity.to_bytes());
        let installed = (|| -> io::Result<()> {
            file.write_all(&secret)?;
            file.sync_all()?;
            drop(file);
            match fs::hard_link(&temporary, &path) {
                Ok(()) => Ok(()),
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => Ok(()),
                Err(e) => Err(e),
            }
        })();
        let cleanup = fs::remove_file(&temporary);
        installed?;
        cleanup?;
    }
    // Another process can observe the committed hard link during the tiny
    // interval before its private temporary link is removed. Never weaken link
    // validation: retry bounded reads, then surface the original refusal.
    for _ in 0..5 {
        match read_host_identity(&path) {
            Ok(identity) => return Ok(identity),
            Err(error) if error.kind() == io::ErrorKind::PermissionDenied => {
                std::thread::sleep(std::time::Duration::from_millis(5))
            }
            Err(error) => return Err(error),
        }
    }
    read_host_identity(&path)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn host_identity_round_trip_is_private_and_read_does_not_create() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("secure/host.identity");
        assert!(read_host_identity(&path).is_err());
        assert!(!path.exists());
        let first = load_or_create_host_identity(temp.path()).unwrap();
        let second = load_or_create_host_identity(temp.path()).unwrap();
        assert_eq!(first.binding(), second.binding());
        assert_eq!(
            first.binding(),
            read_host_identity(&path).unwrap().binding()
        );
    }
    #[test]
    fn concurrent_creation_has_one_pin_and_corruption_never_rekeys() {
        let temp = tempfile::tempdir().unwrap();
        let home = temp.path().to_path_buf();
        let handles = (0..8)
            .map(|_| {
                let home = home.clone();
                std::thread::spawn(move || load_or_create_host_identity(&home).unwrap().binding())
            })
            .collect::<Vec<_>>();
        let pins = handles
            .into_iter()
            .map(|h| h.join().unwrap())
            .collect::<Vec<_>>();
        assert!(pins.iter().all(|p| p == &pins[0]));
        let key = home.join("secure/host.identity");
        fs::write(&key, [1; 8]).unwrap();
        assert!(load_or_create_host_identity(&home).is_err());
        assert_eq!(fs::read(&key).unwrap(), [1; 8]);
    }
    #[test]
    fn client_state_is_bounded_and_replacement_is_explicit() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("private/client.json");
        write_private_file(&path, b"first", false).unwrap();
        assert!(write_private_file(&path, b"other", false).is_err());
        assert_eq!(&**read_private_file(&path, 10).as_ref().unwrap(), b"first");
        write_private_file(&path, b"next", true).unwrap();
        assert_eq!(&**read_private_file(&path, 10).as_ref().unwrap(), b"next");
        assert!(read_private_file(&path, 2).is_err());
    }
    #[test]
    fn client_only_state_creates_private_ancestors_without_host_identity() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("new-home/secure/clients/device.json");
        write_private_file(&path, b"client", false).unwrap();
        assert_eq!(&**read_private_file(&path, 10).as_ref().unwrap(), b"client");
        assert!(!temp.path().join("new-home/secure/host.identity").exists());
    }
    #[cfg(unix)]
    #[test]
    fn fifo_key_path_is_refused_without_waiting_for_a_writer() {
        use std::os::unix::ffi::OsStrExt;
        let temp = tempfile::tempdir().unwrap();
        load_or_create_host_identity(temp.path()).unwrap();
        let path = temp.path().join("secure/fifo");
        let name = std::ffi::CString::new(path.as_os_str().as_bytes()).unwrap();
        // SAFETY: name is a valid NUL-terminated path for this syscall.
        assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
        let (tx, rx) = std::sync::mpsc::channel();
        let read_path = path.clone();
        let reader = std::thread::spawn(move || {
            tx.send(read_private_file(&read_path, 64).is_err()).unwrap();
        });
        let refused_promptly = rx.recv_timeout(std::time::Duration::from_millis(200));
        if refused_promptly.is_err() {
            // Unblock a faulty reader so the failing test leaves no thread.
            let _writer = fs::OpenOptions::new().write(true).open(&path).unwrap();
        }
        reader.join().unwrap();
        assert_eq!(refused_promptly, Ok(true));
    }
    #[cfg(unix)]
    #[test]
    fn permissive_or_linked_identity_is_refused() {
        use std::os::unix::fs::{symlink, PermissionsExt};
        let temp = tempfile::tempdir().unwrap();
        load_or_create_host_identity(temp.path()).unwrap();
        let key = temp.path().join("secure/host.identity");
        let bytes = fs::read(&key).unwrap();
        fs::set_permissions(&key, fs::Permissions::from_mode(0o644)).unwrap();
        assert!(read_host_identity(&key).is_err());
        fs::remove_file(&key).unwrap();
        let target = temp.path().join("secret");
        fs::write(&target, &bytes).unwrap();
        symlink(target, &key).unwrap();
        assert!(read_host_identity(&key).is_err());
    }
    #[cfg(windows)]
    #[test]
    fn permissive_windows_acl_is_refused_without_changing_file() {
        let temp = tempfile::tempdir().unwrap();
        load_or_create_host_identity(temp.path()).unwrap();
        let key = temp.path().join("secure/host.identity");
        let bytes = fs::read(&key).unwrap();
        windows::weaken_for_test(&key).unwrap();
        assert!(read_host_identity(&key).is_err());
        assert_eq!(fs::read(key).unwrap(), bytes);
    }
    #[cfg(windows)]
    #[test]
    fn windows_directory_reparse_point_is_refused() {
        let temp = tempfile::tempdir().unwrap();
        let target = temp.path().join("target");
        load_or_create_host_identity(&target).unwrap();
        let attacker = temp.path().join("attacker");
        fs::create_dir(&attacker).unwrap();
        // Junctions require no developer mode or symlink privilege on Windows CI.
        let status = std::process::Command::new("cmd")
            .args(["/C", "mklink", "/J"])
            .arg(attacker.join("secure"))
            .arg(target.join("secure"))
            .status()
            .unwrap();
        assert!(status.success());
        assert!(load_or_create_host_identity(&attacker).is_err());
    }
}
