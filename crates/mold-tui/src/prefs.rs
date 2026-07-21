//! User-facing TUI preferences — the PREFERENCES section of the Settings
//! workspace, persisted in the `settings` table via `mold-db` (same
//! pattern as [`crate::session`]).
//!
//! Loaded once at boot into [`TuiPrefs`] on `App`; each toggle persists
//! immediately through [`TuiPrefs::save_key`] (like theme changes), so a
//! crash or force-quit never loses a preference flip.

use mold_core::OutputFormat;
use mold_db::{settings as keys, MetadataDb, Settings};

/// DB-backed TUI preferences with their boot-time defaults.
///
/// Defaults (fresh install, DB row absent):
/// - `default_format` — png
/// - `reduce_motion` — false (motion ON; consumed by the tachyonfx PR)
/// - `show_timeline` — true (consumed by the Create-redesign PR)
/// - `confirm_destructive` — true (destructive actions open a Confirm popup)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TuiPrefs {
    /// Seeds a fresh session's Format param (`tui.default_format`);
    /// session / per-model prefs still win once they exist.
    pub default_format: OutputFormat,
    /// `tui.reduce_motion` — disables tachyonfx motion when true.
    pub reduce_motion: bool,
    /// `tui.show_timeline` — Create-view Timeline visibility.
    pub show_timeline: bool,
    /// `tui.confirm_destructive` — when false, destructive actions
    /// dispatch immediately instead of opening the Confirm popup.
    pub confirm_destructive: bool,
}

impl Default for TuiPrefs {
    fn default() -> Self {
        Self {
            default_format: OutputFormat::Png,
            reduce_motion: false,
            show_timeline: true,
            confirm_destructive: true,
        }
    }
}

fn open_db() -> Option<MetadataDb> {
    match mold_db::open_default() {
        Ok(Some(db)) => Some(db),
        Ok(None) => None,
        Err(e) => {
            tracing::warn!(error = %e, "tui prefs: metadata DB open failed; using defaults");
            None
        }
    }
}

/// Parse a persisted `tui.default_format` value. Only png/jpeg are valid
/// session-default formats; anything else falls back to png.
fn parse_format(v: &str) -> OutputFormat {
    match v {
        "jpeg" => OutputFormat::Jpeg,
        _ => OutputFormat::Png,
    }
}

impl TuiPrefs {
    /// The persisted slug for the current `default_format` ("png"/"jpeg").
    pub fn default_format_slug(&self) -> &'static str {
        match self.default_format {
            OutputFormat::Jpeg => "jpeg",
            _ => "png",
        }
    }

    /// Load preferences from the metadata DB, falling back to defaults
    /// for missing rows or an unavailable DB (the TUI still boots; the
    /// cost is that preference flips don't persist).
    pub fn load() -> Self {
        let Some(db) = open_db() else {
            return Self::default();
        };
        Self::load_from(&db)
    }

    /// Load from an explicit DB handle — the testable core of [`load`].
    pub fn load_from(db: &MetadataDb) -> Self {
        let s = Settings::new(db);
        let defaults = Self::default();
        let get_bool = |key: &str, fallback: bool| -> bool {
            s.get_bool(key).unwrap_or(None).unwrap_or(fallback)
        };
        Self {
            default_format: s
                .get_str(keys::TUI_DEFAULT_FORMAT)
                .unwrap_or(None)
                .map(|v| parse_format(&v))
                .unwrap_or(defaults.default_format),
            reduce_motion: get_bool(keys::TUI_REDUCE_MOTION, defaults.reduce_motion),
            show_timeline: get_bool(keys::TUI_SHOW_TIMELINE, defaults.show_timeline),
            confirm_destructive: get_bool(
                keys::TUI_CONFIRM_DESTRUCTIVE,
                defaults.confirm_destructive,
            ),
        }
    }

    /// Persist the current value of a single preference key. Unknown keys
    /// and an unavailable DB are silent no-ops (write errors are logged),
    /// mirroring the session-save behavior.
    pub fn save_key(&self, key: &str) {
        let Some(db) = open_db() else {
            return;
        };
        self.save_key_to(&db, key);
    }

    /// Persist one key to an explicit DB handle — the testable core of
    /// [`save_key`].
    pub fn save_key_to(&self, db: &MetadataDb, key: &str) {
        let s = Settings::new(db);
        let result = match key {
            keys::TUI_DEFAULT_FORMAT => s.set_str(key, self.default_format_slug()),
            keys::TUI_REDUCE_MOTION => s.set_bool(key, self.reduce_motion),
            keys::TUI_SHOW_TIMELINE => s.set_bool(key, self.show_timeline),
            keys::TUI_CONFIRM_DESTRUCTIVE => s.set_bool(key, self.confirm_destructive),
            _ => return,
        };
        if let Err(e) = result {
            tracing::warn!(error = %e, key, "tui prefs: settings write failed");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_env::with_isolated_env;
    use serial_test::serial;

    #[test]
    fn defaults_match_the_spec() {
        let p = TuiPrefs::default();
        assert_eq!(p.default_format, OutputFormat::Png);
        assert!(!p.reduce_motion, "motion is ON by default");
        assert!(p.show_timeline);
        assert!(p.confirm_destructive);
    }

    #[test]
    fn format_slug_round_trips() {
        assert_eq!(parse_format("png"), OutputFormat::Png);
        assert_eq!(parse_format("jpeg"), OutputFormat::Jpeg);
        // Junk falls back to png rather than propagating an invalid
        // session default.
        assert_eq!(parse_format("mp4"), OutputFormat::Png);
        assert_eq!(parse_format(""), OutputFormat::Png);
        let p = TuiPrefs {
            default_format: OutputFormat::Jpeg,
            ..Default::default()
        };
        assert_eq!(parse_format(p.default_format_slug()), OutputFormat::Jpeg);
    }

    #[test]
    #[serial(mold_env)]
    fn save_then_load_round_trips_through_db() {
        with_isolated_env(|_home| {
            let seed = TuiPrefs {
                default_format: OutputFormat::Jpeg,
                reduce_motion: true,
                show_timeline: false,
                confirm_destructive: false,
            };
            seed.save_key(mold_db::settings::TUI_DEFAULT_FORMAT);
            seed.save_key(mold_db::settings::TUI_REDUCE_MOTION);
            seed.save_key(mold_db::settings::TUI_SHOW_TIMELINE);
            seed.save_key(mold_db::settings::TUI_CONFIRM_DESTRUCTIVE);

            let loaded = TuiPrefs::load();
            assert_eq!(loaded, seed);
        });
    }

    #[test]
    #[serial(mold_env)]
    fn save_key_writes_only_the_named_row() {
        with_isolated_env(|_home| {
            let seed = TuiPrefs {
                default_format: OutputFormat::Jpeg,
                reduce_motion: true,
                show_timeline: false,
                confirm_destructive: false,
            };
            seed.save_key(mold_db::settings::TUI_REDUCE_MOTION);

            let loaded = TuiPrefs::load();
            assert!(loaded.reduce_motion, "named key must persist");
            // Everything else stays at defaults — no shotgun writes.
            assert_eq!(loaded.default_format, OutputFormat::Png);
            assert!(loaded.show_timeline);
            assert!(loaded.confirm_destructive);
        });
    }

    #[test]
    #[serial(mold_env)]
    fn unknown_key_is_a_no_op() {
        with_isolated_env(|_home| {
            let seed = TuiPrefs {
                reduce_motion: true,
                ..Default::default()
            };
            seed.save_key("tui.not_a_pref");
            assert_eq!(TuiPrefs::load(), TuiPrefs::default());
        });
    }

    #[test]
    #[serial(mold_env)]
    fn db_disabled_returns_defaults_without_persistence() {
        with_isolated_env(|_home| {
            std::env::set_var("MOLD_DB_DISABLE", "1");
            let seed = TuiPrefs {
                reduce_motion: true,
                ..Default::default()
            };
            seed.save_key(mold_db::settings::TUI_REDUCE_MOTION); // silently no-ops
            assert_eq!(TuiPrefs::load(), TuiPrefs::default());
            std::env::remove_var("MOLD_DB_DISABLE");
        });
    }
}
