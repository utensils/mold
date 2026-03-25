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

/// Colorize a model description: render `[alpha]` or `[beta]` prefix in
/// bright yellow bold, rest dimmed. Normal descriptions are fully dimmed.
pub fn colorize_description(desc: &str) -> String {
    use colored::Colorize;
    // Handle trailing 🔒 emoji separately — it can appear with any prefix
    let (desc, gated_suffix) = if let Some(rest) = desc.strip_suffix(" 🔒") {
        (rest, " 🔒")
    } else {
        (desc, "")
    };

    if let Some(rest) = desc.strip_prefix("[alpha] ") {
        format!("{}{} 🧪", rest.dimmed(), gated_suffix)
    } else if let Some(rest) = desc.strip_prefix("[beta] ") {
        format!("{}{} 🔬", rest.dimmed(), gated_suffix)
    } else {
        format!("{}{}", desc.dimmed(), gated_suffix)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_piped_returns_bool() {
        // In test context, stdout is typically not a terminal (piped by cargo test)
        let _ = is_piped();
    }

    #[test]
    fn status_macro_does_not_panic() {
        // Verify the status! macro works without panicking in either mode
        status!("test message");
        status!("{} {}", "hello", "world");
    }

    #[test]
    fn test_colorize_description_alpha() {
        let result = colorize_description("[alpha] Experimental model");
        assert!(result.contains("🧪"), "should have alpha emoji: {result}");
        assert!(result.contains("Experimental model"));
        // Emoji should be at the end
        assert!(result.ends_with("🧪"), "emoji at end: {result}");
    }

    #[test]
    fn test_colorize_description_beta() {
        let result = colorize_description("[beta] Experimental model");
        assert!(result.contains("🔬"), "should have beta emoji: {result}");
        assert!(result.contains("Experimental model"));
    }

    #[test]
    fn test_colorize_description_no_tag() {
        let result = colorize_description("A stable model description");
        assert!(result.contains("A stable model description"));
    }

    #[test]
    fn test_colorize_description_empty() {
        let result = colorize_description("");
        let _ = result;
    }
}
