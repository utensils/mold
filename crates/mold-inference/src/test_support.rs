//! Test-only helpers shared across this crate's unit tests.

/// One process-wide lock for tests that mutate a shared `MOLD_*` env var.
///
/// `cargo test` runs this crate's tests on several threads at once, so two
/// tests touching the SAME var must serialize even when they live in
/// different files — `MOLD_DEVICE` (`sd15::pipeline`, `sdxl::pipeline`,
/// `expand`) and `MOLD_LTX2_GEMMA_DEVICE` / `MOLD_LTX2_DEBUG_FORCE_CPU_PROMPT_ENCODER`
/// (`device`, `ltx2::pipeline`) are both read/written from more than one
/// module. A lock local to one of those files only protects that file's own
/// tests against each other; it does nothing for a sibling test elsewhere
/// racing the same variable. Hold this lock for the whole mutate-assert
/// window and never across an `.await`.
pub(crate) static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Set `key` to `value`, restoring whatever `key` held before on drop.
///
/// Callers must take [`ENV_LOCK`] before constructing this guard and hold it
/// until the guard (and anything depending on the env var) is done.
pub(crate) struct EnvVarGuard {
    key: &'static str,
    previous: Option<std::ffi::OsString>,
}

impl EnvVarGuard {
    pub(crate) fn set(key: &'static str, value: &str) -> Self {
        let previous = std::env::var_os(key);
        unsafe { std::env::set_var(key, value) };
        Self { key, previous }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        match self.previous.take() {
            Some(value) => unsafe { std::env::set_var(self.key, value) },
            None => unsafe { std::env::remove_var(self.key) },
        }
    }
}

/// The inode `ctime` of `path` as `(seconds, nanoseconds)`, for pairing with
/// [`wait_until_ctime_moves`] around a tamper write.
#[cfg(unix)]
pub(crate) fn ctime_of(path: &std::path::Path) -> (i64, i64) {
    use std::os::unix::fs::MetadataExt;
    let metadata = std::fs::metadata(path).unwrap();
    (metadata.ctime(), metadata.ctime_nsec())
}

/// Spin until `path`'s `ctime` differs from `from`.
///
/// Linux stamps inodes from the coarse clock (`ktime_get_coarse_real_ts64`,
/// a 1–4 ms tick), so a same-length in-place write that lands inside the tick
/// of the previous stamp leaves `ctime` byte-identical and the identity memos
/// under test cannot tell the two files apart — which is a property of the
/// clock, not of the memo. The nudge is `set_permissions` with the file's
/// current mode: a `chmod` always marks the status-change time and touches
/// nothing else — never `mtime`, the length, or the inode — so the test's
/// tamper stays exactly the tamper it describes. Panics if `ctime` never
/// advances within two seconds: a filesystem that will not move it is a
/// broken assumption, not a flake to retry.
#[cfg(unix)]
pub(crate) fn wait_until_ctime_moves(path: &std::path::Path, from: (i64, i64)) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    loop {
        let metadata = std::fs::metadata(path).unwrap();
        if ctime_of(path) != from {
            return;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "ctime of {} stayed at {from:?} for two seconds despite chmod nudges",
            path.display()
        );
        std::fs::set_permissions(path, metadata.permissions()).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
}

/// Non-Unix platforms keep no inode change time, and the identity memos there
/// key on `(len, mtime)` instead — nothing to wait for.
#[cfg(not(unix))]
pub(crate) fn ctime_of(_path: &std::path::Path) -> (i64, i64) {
    (0, 0)
}

#[cfg(not(unix))]
pub(crate) fn wait_until_ctime_moves(_path: &std::path::Path, _from: (i64, i64)) {}
