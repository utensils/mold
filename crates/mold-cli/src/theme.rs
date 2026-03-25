//! Centralized color constants and semantic icon helpers.
//!
//! All user-facing colors are defined here so the CLI has a consistent
//! visual language. Uses only ANSI 16-color methods (no truecolor) for
//! broad terminal compatibility.

use colored::{ColoredString, Colorize};

// ── Status bullets (●) ──────────────────────────────────────────────

/// Green bullet — success / ready state.
pub fn icon_ok() -> ColoredString {
    "●".green()
}

/// Cyan bullet — informational / in-progress.
pub fn icon_info() -> ColoredString {
    "●".cyan()
}

/// Yellow bullet — warning / fallback.
pub fn icon_warn() -> ColoredString {
    "●".yellow()
}

/// Magenta bullet — special mode (img2img, inpainting, ControlNet).
pub fn icon_mode() -> ColoredString {
    "●".magenta()
}

/// Dimmed bullet — neutral / empty state.
pub fn icon_neutral() -> ColoredString {
    "●".dimmed()
}

// ── Result indicators ───────────────────────────────────────────────

/// Green checkmark — task/stage completed.
pub fn icon_done() -> ColoredString {
    "✓".green()
}

/// Red bold cross — hard failure.
pub fn icon_fail() -> ColoredString {
    "✗".red().bold()
}

/// Yellow bold exclamation — caution / overwrite warning.
pub fn icon_alert() -> ColoredString {
    "!".yellow().bold()
}

/// Dimmed middle-dot — minor info bullet.
pub fn icon_bullet() -> ColoredString {
    "·".dimmed()
}

// ── Prefix labels ───────────────────────────────────────────────────

pub fn prefix_error() -> ColoredString {
    "error:".red().bold()
}

pub fn prefix_warning() -> ColoredString {
    "warning:".yellow().bold()
}

pub fn prefix_note() -> ColoredString {
    "note:".dimmed()
}

pub fn prefix_cause() -> ColoredString {
    "caused by:".dimmed()
}

pub fn prefix_hint() -> ColoredString {
    "hint:".dimmed()
}

// ── Spinner / progress bar color token ──────────────────────────────

/// Color name for indicatif spinner and progress bar templates.
/// Used as `{spinner:.<SPINNER_STYLE>}` and `{bar:30.<SPINNER_STYLE>/dim}`.
pub const SPINNER_STYLE: &str = "cyan";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn icons_contain_expected_chars() {
        assert!(icon_ok().to_string().contains('●'));
        assert!(icon_info().to_string().contains('●'));
        assert!(icon_warn().to_string().contains('●'));
        assert!(icon_mode().to_string().contains('●'));
        assert!(icon_neutral().to_string().contains('●'));
        assert!(icon_done().to_string().contains('✓'));
        assert!(icon_fail().to_string().contains('✗'));
        assert!(icon_alert().to_string().contains('!'));
        assert!(icon_bullet().to_string().contains('·'));
    }

    #[test]
    fn prefixes_contain_expected_text() {
        assert!(prefix_error().to_string().contains("error:"));
        assert!(prefix_warning().to_string().contains("warning:"));
        assert!(prefix_note().to_string().contains("note:"));
        assert!(prefix_cause().to_string().contains("caused by:"));
        assert!(prefix_hint().to_string().contains("hint:"));
    }
}
