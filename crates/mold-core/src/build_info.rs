/// Crate version from Cargo.toml.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Short git commit SHA, or `"unknown"` if built outside a git repo.
pub const GIT_SHA: &str = env!("MOLD_GIT_SHA");

/// Date of the git commit (YYYY-MM-DD), or `"unknown"`.
pub const BUILD_DATE: &str = env!("MOLD_BUILD_DATE");

/// Formatted version string: `"0.2.0 (abc1234 2026-03-25)"`.
///
/// Falls back to just the version if SHA is unknown.
pub fn version_string() -> String {
    if GIT_SHA == "unknown" {
        VERSION.to_string()
    } else {
        format!("{VERSION} ({GIT_SHA} {BUILD_DATE})")
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
            GIT_SHA.len() >= 7 && GIT_SHA.len() <= 12,
            "short SHA should be 7-12 chars, got {}: {GIT_SHA}",
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
    fn version_string_includes_all_components() {
        let vs = version_string();
        assert!(vs.contains(VERSION), "should contain version: {vs}");
        if GIT_SHA != "unknown" {
            assert!(vs.contains(GIT_SHA), "should contain SHA: {vs}");
            assert!(vs.contains(BUILD_DATE), "should contain date: {vs}");
        }
    }
}
