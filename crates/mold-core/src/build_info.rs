/// Crate version from Cargo.toml.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Exact git commit SHA, or `"unknown"` if built outside a git repo.
///
/// This is the full 40-character commit when git or the build environment can
/// supply one, because the private MiniMax H3 campaign identity binds an
/// unambiguous source revision. Use [`SHORT_GIT_SHA`] for anything a person
/// reads.
pub const GIT_SHA: &str = env!("MOLD_GIT_SHA");

/// Abbreviated commit SHA for display, or `"unknown"`.
///
/// Chrome that right-aligns a version string shares a row with other content —
/// the TUI tab strip is one — so the displayed width must not depend on how
/// long the exact SHA happens to be.
pub const SHORT_GIT_SHA: &str = env!("MOLD_GIT_SHA_SHORT");

/// Displayed abbreviation width. Keep in sync with `build.rs`.
pub const SHORT_SHA_LENGTH: usize = 7;

/// Date of the git commit (YYYY-MM-DD), or `"unknown"`.
pub const BUILD_DATE: &str = env!("MOLD_BUILD_DATE");

/// Distribution channel embedded by official release workflows.
///
/// Local and third-party builds use `"development"`, so commit equality alone
/// can never make them impersonate an installed official nightly.
pub const BUILD_CHANNEL: &str = env!("MOLD_BUILD_CHANNEL");

/// Compile-time version string: `"0.2.0 (abc1234 2026-03-25)"`.
///
/// Equivalent to [`version_string()`] but as a `&'static str` for use in
/// clap's `#[command(version = ...)]` attribute.
pub const FULL_VERSION: &str = env!("MOLD_FULL_VERSION");

/// Formatted version string: `"0.2.0 (abc1234 2026-03-25)"`.
///
/// Falls back to just the version if SHA is unknown.
pub fn version_string() -> String {
    if SHORT_GIT_SHA == "unknown" {
        VERSION.to_string()
    } else {
        format!("{VERSION} ({SHORT_GIT_SHA} {BUILD_DATE})")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_matches_cargo_pkg() {
        assert_eq!(VERSION, env!("CARGO_PKG_VERSION"));
    }

    #[test]
    fn git_sha_is_populated() {
        assert!(!GIT_SHA.is_empty());
        if GIT_SHA == "unknown" {
            return; // no git context (crates.io / sandboxed build)
        }
        assert!(
            (7..=12).contains(&GIT_SHA.len()) || GIT_SHA.len() == 40,
            "SHA should be a 7-12 character abbreviation or exact 40-character commit, got {}: {GIT_SHA}",
            GIT_SHA.len()
        );
        assert!(
            GIT_SHA.chars().all(|c| c.is_ascii_hexdigit()),
            "SHA should be hex: {GIT_SHA}"
        );
    }

    #[test]
    fn build_date_is_valid() {
        assert!(!BUILD_DATE.is_empty());
        if BUILD_DATE == "unknown" {
            return; // no git context (crates.io / sandboxed build)
        }
        // YYYY-MM-DD
        assert_eq!(
            BUILD_DATE.len(),
            10,
            "date should be YYYY-MM-DD: {BUILD_DATE}"
        );
        let parts: Vec<&str> = BUILD_DATE.split('-').collect();
        assert_eq!(parts.len(), 3, "date should have 3 parts: {BUILD_DATE}");
        assert!(parts[0].parse::<u32>().is_ok(), "year should be numeric");
        assert!(parts[1].parse::<u32>().is_ok(), "month should be numeric");
        assert!(parts[2].parse::<u32>().is_ok(), "day should be numeric");
    }

    #[test]
    fn build_channel_is_known() {
        assert!(matches!(
            BUILD_CHANNEL,
            "stable" | "nightly" | "development"
        ));
    }

    #[test]
    fn version_string_includes_all_components() {
        let vs = version_string();
        assert!(vs.contains(VERSION), "should contain version: {vs}");
        if SHORT_GIT_SHA != "unknown" {
            assert!(vs.contains(SHORT_GIT_SHA), "should contain SHA: {vs}");
            assert!(vs.contains(BUILD_DATE), "should contain date: {vs}");
        }
    }

    #[test]
    fn displayed_sha_width_is_independent_of_the_exact_sha() {
        // The TUI tab strip right-aligns `version_string()` onto the row that
        // holds the tab labels, so a 40-character SHA here overlaps the tabs
        // and moves what a click lands on. The display width is pinned rather
        // than inherited from however long GIT_SHA happens to be.
        if SHORT_GIT_SHA == "unknown" {
            assert_eq!(GIT_SHA, "unknown");
            return;
        }
        assert_eq!(
            SHORT_GIT_SHA.len(),
            SHORT_SHA_LENGTH,
            "displayed SHA must be exactly {SHORT_SHA_LENGTH} characters, got {SHORT_GIT_SHA}"
        );
        assert!(
            GIT_SHA.starts_with(SHORT_GIT_SHA),
            "displayed SHA must abbreviate the exact one: {SHORT_GIT_SHA} vs {GIT_SHA}"
        );
        assert!(
            !version_string().contains(GIT_SHA) || GIT_SHA.len() == SHORT_SHA_LENGTH,
            "the exact SHA must not reach a human-facing version string"
        );
    }
}
