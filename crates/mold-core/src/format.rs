//! Human-readable byte formatting shared by every surface (CLI, TUI, MCP).
//!
//! Two flavors cover the real conventions in the tree:
//! - [`human_bytes`] — spaced, e.g. `"7.0 GB"`, for verbose prose output.
//! - [`human_bytes_compact`] — single-letter, e.g. `"7.0G"`, for dense
//!   progress lines and table columns.
//!
//! Both are 1024-based with one decimal. Presentational one-offs with a
//! fixed unit (e.g. `mold list`'s padded GB column) stay local to their
//! callers — add a flavor here only when a second caller needs it.

/// Spaced, 1-decimal, 1024-based: "7.0 GB", "1.5 KB", "42 B".
pub fn human_bytes(bytes: u64) -> String {
    if bytes >= 1_073_741_824 {
        format!("{:.1} GB", bytes as f64 / 1_073_741_824.0)
    } else if bytes >= 1_048_576 {
        format!("{:.1} MB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{bytes} B")
    }
}

/// Compact single-letter, 1-decimal, 1024-based: "7.0G", "5.0M", "1.5K", "42B".
pub fn human_bytes_compact(bytes: u64) -> String {
    const K: f64 = 1024.0;
    let f = bytes as f64;
    if f < K {
        format!("{bytes}B")
    } else if f < K * K {
        format!("{:.1}K", f / K)
    } else if f < K * K * K {
        format!("{:.1}M", f / (K * K))
    } else {
        format!("{:.1}G", f / (K * K * K))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn human_bytes_compact_picks_the_right_unit_at_each_threshold() {
        // Sub-KiB → bare bytes.
        assert_eq!(human_bytes_compact(0), "0B");
        assert_eq!(human_bytes_compact(1), "1B");
        assert_eq!(human_bytes_compact(1023), "1023B");
        // KiB.
        assert_eq!(human_bytes_compact(1024), "1.0K");
        assert_eq!(human_bytes_compact(1536), "1.5K");
        // MiB.
        assert_eq!(human_bytes_compact(1024 * 1024), "1.0M");
        assert_eq!(human_bytes_compact(5 * 1024 * 1024), "5.0M");
        // GiB.
        assert_eq!(human_bytes_compact(1024u64.pow(3)), "1.0G");
        assert_eq!(human_bytes_compact(5_368_709_120), "5.0G");
    }

    #[test]
    fn human_bytes_spaced_flavor_matches_units_and_spacing() {
        assert_eq!(human_bytes(42), "42 B");
        assert_eq!(human_bytes(1536), "1.5 KB");
        assert_eq!(human_bytes(5 * 1024 * 1024), "5.0 MB");
        assert_eq!(human_bytes(7_516_192_768), "7.0 GB");
    }
}
