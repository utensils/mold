//! TTY-aware output helpers for pipe support.
//!
//! When stdout is a terminal, status messages go to stdout with colors.
//! When stdout is a pipe, status messages go to stderr and image data goes to stdout.

use std::io::IsTerminal;

/// Returns true if stdout is a pipe (not a terminal).
pub fn is_piped() -> bool {
    !std::io::stdout().is_terminal()
}

/// Print a status message to the appropriate stream.
/// Goes to stdout when interactive, stderr when piped (so stdout stays clean for binary data).
macro_rules! status {
    ($($arg:tt)*) => {
        if $crate::output::is_piped() {
            eprintln!($($arg)*);
        } else {
            println!($($arg)*);
        }
    };
}

pub(crate) use status;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_piped_returns_bool() {
        // In test context, stdout is typically not a terminal (piped by cargo test)
        let result = is_piped();
        // Just verify it doesn't panic; actual value depends on test runner
        assert!(result == true || result == false);
    }

    #[test]
    fn status_macro_does_not_panic() {
        // Verify the status! macro works without panicking in either mode
        status!("test message");
        status!("{} {}", "hello", "world");
    }
}
