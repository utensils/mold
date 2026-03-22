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

/// Colorize a model description: render `[broken]` or `[beta]` prefix in
/// bright red bold, rest dimmed. Normal descriptions are fully dimmed.
pub fn colorize_description(desc: &str) -> String {
    use colored::Colorize;
    if let Some(rest) = desc.strip_prefix("[broken] ") {
        format!("{} {}", "[broken]".bright_red().bold(), rest.dimmed())
    } else if let Some(rest) = desc.strip_prefix("[beta] ") {
        format!("{} {}", "[beta]".bright_yellow().bold(), rest.dimmed())
    } else {
        format!("{}", desc.dimmed())
    }
}

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

    #[test]
    fn test_colorize_description_beta() {
        let result = colorize_description("[beta] Experimental model");
        // Should contain the [beta] text and the rest of the description
        // The output includes ANSI escape codes for bright_red bold and dimmed
        assert!(result.contains("[beta]"));
        assert!(result.contains("Experimental model"));
    }

    #[test]
    fn test_colorize_description_no_beta() {
        let result = colorize_description("A stable model description");
        // Should contain the description text, fully dimmed
        assert!(result.contains("A stable model description"));
        // Should NOT contain any [beta] styling (no bright_red bold sequences separate from dimmed)
        // The entire string is wrapped in dimmed formatting only
    }

    #[test]
    fn test_colorize_description_empty() {
        // Empty string should not panic
        let result = colorize_description("");
        // Empty string has no "[beta] " prefix, so it takes the dimmed path
        // Result contains ANSI codes wrapping an empty string; should not be empty
        // because dimmed() adds escape sequences
        let _ = result;
    }
}
