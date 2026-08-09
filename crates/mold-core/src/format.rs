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

/// Human-readable display name for a model family identifier, shared by the
/// CLI (`mold list`, run banners) and the TUI (Models tables/details) so the
/// two terminal surfaces can never disagree about how a family is presented
/// (#806). `None` means the identifier is unknown here — callers keep their
/// own fallback (the CLI shows the raw identifier, the TUI uppercases it).
///
/// The browser surfaces carry their own table in `studio/lib/familyLabels.ts`;
/// the wan row is pinned across both languages by
/// `tests/fixtures/wan/surface-parity-v1.json`.
pub fn family_display_label(family: &str) -> Option<&'static str> {
    Some(match family {
        "flux" => "FLUX.1",
        "flux2" => "FLUX.2",
        "sd15" => "SD 1.5",
        "sd3" | "sd3.5" => "SD 3.5",
        "sdxl" => "SDXL",
        "z-image" => "Z-Image",
        "qwen-image" | "qwen_image" => "Qwen-Image",
        "qwen-image-edit" => "Qwen-Image-Edit",
        "wuerstchen" | "wuerstchen-v2" => "Wuerstchen",
        "ltx-video" | "ltx_video" => "LTX Video",
        "ltx2" => "LTX-2",
        "wan" => "Wan Video",
        "minimax-h3" | "minimax_h3" | "minimaxh3" => "MiniMax H3",
        "controlnet" => "ControlNet",
        "qwen3-expand" => "Expand",
        "upscaler" => "Upscaler",
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wan_family_label_matches_the_shared_surface_parity_fixture() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/fixtures/wan/surface-parity-v1.json"
        )))
        .expect("fixture parses");
        let family = fixture["family"].as_str().expect("family");
        let label = fixture["family_label"].as_str().expect("family_label");
        assert_eq!(family_display_label(family), Some(label));
    }

    #[test]
    fn unknown_families_have_no_label_so_callers_keep_their_fallback() {
        assert_eq!(family_display_label("companion"), None);
        assert_eq!(family_display_label(""), None);
    }

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
