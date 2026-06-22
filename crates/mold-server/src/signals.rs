//! SIGPIPE disposition for the long-running server process.
//!
//! Rust's runtime installs `SIG_IGN` for `SIGPIPE` at startup, so a `write()` to
//! a peer that has closed its end returns `EPIPE` (a recoverable per-request
//! error) instead of a fatal signal. The `mold` CLI deliberately resets SIGPIPE
//! to `SIG_DFL` in `main()` so short-lived commands terminate cleanly when their
//! stdout pipe closes (e.g. `mold run "a cat" | head`).
//!
//! For a long-running HTTP server that `SIG_DFL` disposition is fatal: a single
//! client disconnecting mid-write delivers SIGPIPE and kills the whole process —
//! no panic, no error log (see issue #342). Restoring `SIG_IGN` at server startup
//! makes such writes return `EPIPE`, which hyper/axum handle as a normal client
//! disconnect.

/// Restore `SIG_IGN` for `SIGPIPE` so writes to a dropped connection surface as
/// `EPIPE` rather than terminating the process. Idempotent; safe to call after
/// the CLI has reset the disposition to `SIG_DFL`. No-op on non-unix targets.
#[cfg(unix)]
pub fn ignore_sigpipe() {
    // SAFETY: installing `SIG_IGN` for a signal is async-signal-safe and has no
    // memory-safety implications. `libc::signal` returns the previous handler,
    // which we intentionally discard.
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_IGN);
    }
}

/// No-op on non-unix targets, where SIGPIPE does not exist.
#[cfg(not(unix))]
pub fn ignore_sigpipe() {}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // Signal disposition is process-global and cargo runs tests in parallel
    // threads, so serialize the tests that mutate it.
    static SIGPIPE_LOCK: Mutex<()> = Mutex::new(());

    /// Read the current SIGPIPE handler without modifying it: pass a null `act`
    /// and capture the existing disposition into `oldact`.
    fn current_sigpipe_handler() -> libc::sighandler_t {
        let mut oldact: libc::sigaction = unsafe { std::mem::zeroed() };
        let rc = unsafe { libc::sigaction(libc::SIGPIPE, std::ptr::null(), &mut oldact) };
        assert_eq!(
            rc,
            0,
            "sigaction query failed: {}",
            std::io::Error::last_os_error()
        );
        oldact.sa_sigaction
    }

    /// The actual failure mode from #342: with `SIG_DFL` a write to a pipe whose
    /// read end is closed kills the process; with our fix it must return `EPIPE`.
    /// If `ignore_sigpipe()` did nothing, this test would terminate the whole test
    /// binary via SIGPIPE instead of failing an assertion.
    #[test]
    fn write_to_broken_pipe_returns_epipe_not_signal() {
        let _guard = SIGPIPE_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        // Worst case: the CLI's global reset is in effect.
        unsafe {
            libc::signal(libc::SIGPIPE, libc::SIG_DFL);
        }
        ignore_sigpipe();

        let mut fds = [0 as libc::c_int; 2];
        assert_eq!(
            unsafe { libc::pipe(fds.as_mut_ptr()) },
            0,
            "pipe() failed: {}",
            std::io::Error::last_os_error()
        );
        let (read_fd, write_fd) = (fds[0], fds[1]);

        // Close the reader so any write to `write_fd` gets EPIPE.
        assert_eq!(unsafe { libc::close(read_fd) }, 0);

        let buf = [0u8; 16];
        let n = unsafe { libc::write(write_fd, buf.as_ptr() as *const libc::c_void, buf.len()) };
        let err = std::io::Error::last_os_error();
        unsafe {
            libc::close(write_fd);
        }

        assert_eq!(n, -1, "write to broken pipe should fail");
        assert_eq!(
            err.raw_os_error(),
            Some(libc::EPIPE),
            "broken-pipe write should report EPIPE"
        );
    }

    /// `ignore_sigpipe()` must leave SIGPIPE at `SIG_IGN` even when it was reset
    /// to `SIG_DFL` first (mirroring the CLI's `main()` global reset).
    #[test]
    fn ignore_sigpipe_installs_sig_ign_over_sig_dfl() {
        let _guard = SIGPIPE_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        unsafe {
            libc::signal(libc::SIGPIPE, libc::SIG_DFL);
        }
        assert_eq!(current_sigpipe_handler(), libc::SIG_DFL);

        ignore_sigpipe();

        assert_eq!(current_sigpipe_handler(), libc::SIG_IGN);
    }
}
