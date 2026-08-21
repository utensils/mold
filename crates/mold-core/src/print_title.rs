//! Print titles: validation, filename slugs, and the titled output filename.
//!
//! A print title is user-authored text that rides `GenerateRequest.title` →
//! `OutputMetadata.title` → the gallery row. At save time the title is also
//! folded into the output filename as a lossy ASCII slug so prints stay
//! recognizable in a plain file browser:
//!
//! ```text
//! mold-{model}-{timestamp}[-{index}]~{slug}.{ext}
//! ```
//!
//! The `~` separator is the one character the legacy `mold-<model>-<unix>
//! [-<idx>].<ext>` shape never produced (model names sanitize `:` to `-`, and
//! the rest is alphanumerics, `-`, `_`, and `.`), so a reader can strip a
//! trailing `~<slug>` from the stem and hand the remainder to the existing
//! filename parser. The slug is deliberately lossy — the authoritative title
//! lives in the embedded metadata and the DB row, never in the filename.

/// Maximum length, in bytes, of a filename slug derived from a title.
pub const TITLE_SLUG_MAX_LEN: usize = 40;

/// Maximum length, in characters, of a print title.
pub const PRINT_TITLE_MAX_CHARS: usize = 120;

/// Separator between the legacy filename stem and the title slug.
pub const TITLE_SLUG_SEPARATOR: char = '~';

/// Lossy ASCII slug for a title: lowercase ASCII letters and digits are
/// kept, every other character (including whitespace and non-ASCII) becomes
/// `-`, runs collapse to one `-`, leading/trailing `-` are trimmed, and the
/// result is capped at [`TITLE_SLUG_MAX_LEN`] bytes (re-trimmed so the cut
/// never leaves a dangling `-`). Returns `None` when nothing survives.
pub fn title_slug(title: &str) -> Option<String> {
    let mut slug = String::with_capacity(title.len().min(TITLE_SLUG_MAX_LEN));
    let mut pending_dash = false;
    for ch in title.chars() {
        if ch.is_ascii_alphanumeric() {
            if pending_dash && !slug.is_empty() {
                if slug.len() + 1 >= TITLE_SLUG_MAX_LEN {
                    break;
                }
                slug.push('-');
            }
            pending_dash = false;
            if slug.len() >= TITLE_SLUG_MAX_LEN {
                break;
            }
            slug.push(ch.to_ascii_lowercase());
        } else {
            pending_dash = true;
        }
    }
    let trimmed = slug.trim_matches('-');
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

/// Validate a user-supplied print title: trims surrounding whitespace,
/// rejects control characters, caps the length at [`PRINT_TITLE_MAX_CHARS`]
/// characters, and maps an empty (or all-whitespace) title to `Ok(None)` so
/// callers can treat "cleared" and "never set" identically.
pub fn validate_print_title(raw: &str) -> Result<Option<String>, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    if trimmed.chars().any(char::is_control) {
        return Err("title must not contain control characters".to_string());
    }
    let chars = trimmed.chars().count();
    if chars > PRINT_TITLE_MAX_CHARS {
        return Err(format!(
            "title is {chars} characters; the maximum is {PRINT_TITLE_MAX_CHARS}"
        ));
    }
    Ok(Some(trimmed.to_string()))
}

/// Build an output filename like [`crate::default_output_filename`] and, when
/// `slug` is present, append `~{slug}` to the stem:
/// `mold-{model}-{timestamp}[-{index}]~{slug}.{ext}`. A `None` slug yields
/// exactly the legacy filename so untitled prints are byte-identical to
/// before.
pub fn default_output_filename_titled(
    model: &str,
    timestamp: u64,
    ext: &str,
    batch: u32,
    index: u32,
    slug: Option<&str>,
) -> String {
    let legacy = crate::types::default_output_filename(model, timestamp, ext, batch, index);
    let Some(slug) = slug.filter(|slug| !slug.is_empty()) else {
        return legacy;
    };
    let suffix = format!(".{ext}");
    let stem = legacy
        .strip_suffix(suffix.as_str())
        .unwrap_or(legacy.as_str());
    format!("{stem}{TITLE_SLUG_SEPARATOR}{slug}.{ext}")
}

/// Separator between the components of a *download* file name.
pub const DOWNLOAD_NAME_SEPARATOR: &str = "__";

/// Maximum length, in bytes, of the model component of a download name.
///
/// Capped for a filesystem reason, not a cosmetic one: `cv:` / `hf:` ids are
/// caller-supplied and unbounded, and `title(40) + model + s<20 digits> + ext`
/// has to stay under the 255-byte filename limit every common filesystem
/// enforces. Two very long model ids sharing a truncated name is a cosmetic
/// annoyance in a Downloads folder; `ENAMETOOLONG` is a failed save.
pub const DOWNLOAD_MODEL_SLUG_MAX_LEN: usize = 80;

/// Last-resort stem when nothing about a print slugs to anything usable.
pub const DOWNLOAD_FALLBACK_STEM: &str = "print";

/// Name to suggest when the user saves or exports a print, which is a
/// different question from what the gallery calls the file on disk.
///
/// The gallery filename is an identity — timestamped, never renamed, and
/// stable across a rename of the print ([`title_slug`] folds the
/// creation-time title in once and that is that). A download name is a
/// *label*: it is read by a human in a Downloads folder, so it leads with
/// the print's current title and drops the timestamp entirely:
///
/// ```text
/// {title-slug}__{model}__s{seed}.{ext}
/// ```
///
/// The three components are independent: each is dropped when it has nothing
/// to contribute, and the survivors are joined with `__` (chosen because a
/// single `-` is legal *inside* every one of them). An untitled print is
/// `{model}__s{seed}.{ext}`; a model that slugs to nothing is
/// `{title-slug}__s{seed}.{ext}`; and if nothing at all survives the stem is
/// [`DOWNLOAD_FALLBACK_STEM`]. The model is sanitized so `flux-dev:q4`
/// becomes `flux-dev-q4` and `cv:12345` becomes `cv-12345`, keeping the name
/// portable across filesystems.
///
/// `seed` is not optional here because Rust never has an unresolved one — a
/// saved print's `OutputMetadata.seed` is a `u64`. Browser mirrors that can
/// hold a seedless gallery entry drop that segment, which is exactly what
/// this function would do if it could.
///
/// This is the Rust authority; the browser surfaces mirror it.
pub fn download_file_name(title: Option<&str>, model: &str, seed: u64, ext: &str) -> String {
    let mut segments: Vec<String> = Vec::with_capacity(3);
    segments.extend(title.and_then(title_slug));
    let model = sanitize_download_component(model, DOWNLOAD_MODEL_SLUG_MAX_LEN);
    if !model.is_empty() {
        segments.push(model);
    }
    segments.push(format!("s{seed}"));

    let stem = if segments.is_empty() {
        DOWNLOAD_FALLBACK_STEM.to_string()
    } else {
        segments.join(DOWNLOAD_NAME_SEPARATOR)
    };
    match normalize_download_extension(ext) {
        Some(ext) => format!("{stem}.{ext}"),
        None => stem,
    }
}

/// Normalize a caller-supplied extension: a leading `.` is stripped (so
/// `".PNG"` and `"png"` agree), the rest is lowercased, and anything that
/// leaves nothing behind yields `None` — a bare trailing dot is not a
/// filename anyone wants.
fn normalize_download_extension(ext: &str) -> Option<String> {
    let trimmed = ext.trim().trim_start_matches('.').trim();
    (!trimmed.is_empty()).then(|| trimmed.to_ascii_lowercase())
}

/// Reduce one download-name component to filesystem-portable characters:
/// ASCII alphanumerics survive as lowercase, everything else collapses to a
/// single `-`, the edges are trimmed, and the result is capped at `max_len`
/// bytes (re-trimmed so the cut never leaves a dangling `-`). Deliberately
/// never emits `_`, so the `__` separator stays unambiguous.
fn sanitize_download_component(raw: &str, max_len: usize) -> String {
    let mut out = String::with_capacity(raw.len().min(max_len));
    let mut pending_dash = false;
    for ch in raw.chars() {
        if ch.is_ascii_alphanumeric() {
            if pending_dash && !out.is_empty() {
                if out.len() + 1 >= max_len {
                    break;
                }
                out.push('-');
            }
            pending_dash = false;
            if out.len() >= max_len {
                break;
            }
            out.push(ch.to_ascii_lowercase());
        } else {
            pending_dash = true;
        }
    }
    out.trim_matches('-').to_string()
}

/// Strip a trailing `~<slug>` from a filename stem (no extension), returning
/// the legacy stem. Stems without a separator are returned unchanged.
pub fn strip_title_slug(stem: &str) -> &str {
    match stem.rfind(TITLE_SLUG_SEPARATOR) {
        Some(index) => &stem[..index],
        None => stem,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slug_lowercases_and_joins_words() {
        assert_eq!(
            title_slug("Smurf Village").as_deref(),
            Some("smurf-village")
        );
        assert_eq!(title_slug("smurf-04").as_deref(), Some("smurf-04"));
    }

    #[test]
    fn slug_collapses_punctuation_runs_and_trims_edges() {
        assert_eq!(
            title_slug("  ...Hello,   World!!!  ").as_deref(),
            Some("hello-world")
        );
        assert_eq!(title_slug("a -- b").as_deref(), Some("a-b"));
        assert_eq!(title_slug("--").as_deref(), None);
    }

    #[test]
    fn slug_drops_non_ascii_but_keeps_ascii_neighbours() {
        assert_eq!(title_slug("Café au lait").as_deref(), Some("caf-au-lait"));
        assert_eq!(title_slug("日本語").as_deref(), None);
        assert_eq!(title_slug("日本語 cat 🐈").as_deref(), Some("cat"));
    }

    #[test]
    fn slug_is_none_for_empty_or_symbol_only() {
        assert_eq!(title_slug(""), None);
        assert_eq!(title_slug("   "), None);
        assert_eq!(title_slug("!!!"), None);
        assert_eq!(title_slug("~~~"), None);
    }

    #[test]
    fn slug_never_contains_the_separator() {
        let slug = title_slug("tilde~in~title").unwrap();
        assert!(!slug.contains(TITLE_SLUG_SEPARATOR));
        assert_eq!(slug, "tilde-in-title");
    }

    #[test]
    fn slug_caps_length_without_dangling_dash() {
        let long = "word ".repeat(30);
        let slug = title_slug(&long).unwrap();
        assert!(slug.len() <= TITLE_SLUG_MAX_LEN, "{slug}");
        assert!(!slug.ends_with('-'), "{slug}");
        assert!(!slug.starts_with('-'), "{slug}");
        let solid = "x".repeat(100);
        assert_eq!(title_slug(&solid).unwrap().len(), TITLE_SLUG_MAX_LEN);
        // Exactly the cap is accepted untouched.
        let exact = "y".repeat(TITLE_SLUG_MAX_LEN);
        assert_eq!(title_slug(&exact).as_deref(), Some(exact.as_str()));
    }

    #[test]
    fn slug_is_idempotent() {
        for title in [
            "Smurf Village",
            "  ...Hello,   World!!!  ",
            "Café au lait",
            &"word ".repeat(30),
            "tilde~in~title",
        ] {
            let once = title_slug(title).unwrap();
            assert_eq!(title_slug(&once).as_deref(), Some(once.as_str()));
        }
    }

    #[test]
    fn titled_filename_places_slug_before_extension() {
        assert_eq!(
            default_output_filename_titled(
                "flux-dev:q4",
                1700000000000,
                "png",
                1,
                0,
                Some("smurf")
            ),
            "mold-flux-dev-q4-1700000000000~smurf.png"
        );
        assert_eq!(
            default_output_filename_titled("sdxl", 1700000000000, "png", 4, 2, Some("river")),
            "mold-sdxl-1700000000000-2~river.png"
        );
    }

    #[test]
    fn titled_filename_without_slug_matches_legacy() {
        for slug in [None, Some("")] {
            assert_eq!(
                default_output_filename_titled("flux-dev:q8", 1700000000, "jpeg", 1, 0, slug),
                crate::types::default_output_filename("flux-dev:q8", 1700000000, "jpeg", 1, 0)
            );
        }
    }

    #[test]
    fn titled_filename_contains_exactly_one_separator() {
        let name = default_output_filename_titled(
            "flux-dev:q4",
            1700000000000,
            "png",
            1,
            0,
            title_slug("Smurf ~ Village").as_deref(),
        );
        assert_eq!(name.matches(TITLE_SLUG_SEPARATOR).count(), 1, "{name}");
        assert_eq!(name, "mold-flux-dev-q4-1700000000000~smurf-village.png");
    }

    #[test]
    fn strip_title_slug_round_trips_titled_stems() {
        assert_eq!(
            strip_title_slug("mold-flux-dev-q4-1700000000000~smurf-04"),
            "mold-flux-dev-q4-1700000000000"
        );
        assert_eq!(
            strip_title_slug("mold-sdxl-1700000000000-2~river"),
            "mold-sdxl-1700000000000-2"
        );
        assert_eq!(
            strip_title_slug("mold-sdxl-1700000000000-2"),
            "mold-sdxl-1700000000000-2"
        );
    }

    #[test]
    fn download_name_leads_with_the_title_slug() {
        assert_eq!(
            download_file_name(Some("Smurf Village at Dusk"), "flux-dev:q4", 42, "png"),
            "smurf-village-at-dusk__flux-dev-q4__s42.png"
        );
    }

    #[test]
    fn download_name_falls_back_to_model_and_seed_without_a_usable_title() {
        for title in [None, Some(""), Some("   "), Some("!!!"), Some("日本語")] {
            assert_eq!(
                download_file_name(title, "sdxl", 7, "jpeg"),
                "sdxl__s7.jpeg",
                "title {title:?}"
            );
        }
    }

    #[test]
    fn download_name_sanitizes_the_model_and_keeps_exactly_two_separators() {
        let name = download_file_name(Some("Owl"), "ltx-2-19b-distilled:fp8", 1, "mp4");
        assert_eq!(name, "owl__ltx-2-19b-distilled-fp8__s1.mp4");
        assert_eq!(name.matches(DOWNLOAD_NAME_SEPARATOR).count(), 2, "{name}");
        // Catalog ids keep their source prefix, readably.
        assert_eq!(
            download_file_name(None, "cv:12345", 9, "png"),
            "cv-12345__s9.png"
        );
    }

    /// Each component is independent: an absent one is dropped rather than
    /// substituted, so the name never carries a placeholder word that looks
    /// like a real title or model.
    #[test]
    fn download_name_drops_unusable_components_rather_than_substituting_them() {
        // No title.
        assert_eq!(
            download_file_name(None, "flux-dev:q4", 42, "png"),
            "flux-dev-q4__s42.png"
        );
        // No sluggable model.
        assert_eq!(
            download_file_name(Some("Owl"), "???", 42, "png"),
            "owl__s42.png"
        );
        // Neither — the seed alone still names the file.
        assert_eq!(download_file_name(None, "???", 42, "png"), "s42.png");
    }

    /// The model component is capped so a caller-supplied `cv:` / `hf:` id
    /// cannot push the whole name past the 255-byte filename limit.
    #[test]
    fn download_name_caps_the_model_without_a_dangling_dash() {
        let long = format!("hf:{}", "Segment/".repeat(40));
        let name = download_file_name(Some("Owl"), &long, 42, "png");
        let model = name
            .strip_prefix("owl__")
            .and_then(|rest| rest.strip_suffix("__s42.png"))
            .expect("the model sits between the title and the seed");
        assert!(model.len() <= DOWNLOAD_MODEL_SLUG_MAX_LEN, "{model}");
        assert!(!model.ends_with('-'), "{model}");
        assert!(!model.starts_with('-'), "{model}");
        assert!(name.len() < 255, "{} bytes: {name}", name.len());
        // A model exactly at the cap is kept whole.
        let exact = "z".repeat(DOWNLOAD_MODEL_SLUG_MAX_LEN);
        assert_eq!(
            download_file_name(None, &exact, 1, "png"),
            format!("{exact}__s1.png")
        );
    }

    /// Extensions are caller-supplied on the browser side too, so `.PNG`,
    /// `png`, and `""` must all agree — and an absent one must never leave a
    /// bare trailing dot.
    #[test]
    fn download_name_normalizes_the_extension() {
        for ext in ["png", ".png", "PNG", ".PNG", "  .Png "] {
            assert_eq!(
                download_file_name(Some("Owl"), "flux-dev", 1, ext),
                "owl__flux-dev__s1.png",
                "ext {ext:?}"
            );
        }
        for ext in ["", "   ", "."] {
            assert_eq!(
                download_file_name(Some("Owl"), "flux-dev", 1, ext),
                "owl__flux-dev__s1",
                "ext {ext:?}"
            );
        }
    }

    /// A defensive last resort the Rust side cannot actually reach — `seed`
    /// is a `u64`, so `s{seed}` is always a segment — but the browser mirror
    /// can hold a seedless entry, and both must land on the same word.
    #[test]
    fn download_name_falls_back_to_a_shared_stem_word() {
        assert_eq!(DOWNLOAD_FALLBACK_STEM, "print");
    }

    #[test]
    fn download_name_carries_the_full_seed_range() {
        assert_eq!(
            download_file_name(None, "flux-dev", u64::MAX, "png"),
            format!("flux-dev__s{}.png", u64::MAX)
        );
        assert_eq!(
            download_file_name(None, "flux-dev", 0, "png"),
            "flux-dev__s0.png"
        );
    }

    /// The download name is a label; the gallery filename is an identity.
    /// They must not converge — no timestamp here, no `~` separator there.
    #[test]
    fn download_name_is_not_the_gallery_filename() {
        let gallery =
            default_output_filename_titled("flux-dev:q4", 1700000000000, "png", 1, 0, Some("owl"));
        let download = download_file_name(Some("Owl"), "flux-dev:q4", 42, "png");
        assert_ne!(gallery, download);
        assert!(!download.contains("1700000000000"), "{download}");
        assert!(!download.contains(TITLE_SLUG_SEPARATOR), "{download}");
    }

    #[test]
    fn validate_title_trims_and_clears_empty() {
        assert_eq!(validate_print_title("").unwrap(), None);
        assert_eq!(validate_print_title("   \t ").unwrap(), None);
        assert_eq!(
            validate_print_title("  Smurf Village  ")
                .unwrap()
                .as_deref(),
            Some("Smurf Village")
        );
    }

    #[test]
    fn validate_title_keeps_unicode() {
        assert_eq!(
            validate_print_title("Café 日本語 🐈").unwrap().as_deref(),
            Some("Café 日本語 🐈")
        );
    }

    #[test]
    fn validate_title_rejects_control_characters() {
        assert!(validate_print_title("line\nbreak").is_err());
        assert!(validate_print_title("tab\tinside").is_err());
        assert!(validate_print_title("nul\0").is_err());
        assert!(validate_print_title("esc\u{1b}[0m").is_err());
    }

    #[test]
    fn validate_title_caps_at_120_characters_not_bytes() {
        let exact = "é".repeat(PRINT_TITLE_MAX_CHARS);
        assert_eq!(
            validate_print_title(&exact).unwrap().as_deref(),
            Some(exact.as_str())
        );
        let over = "é".repeat(PRINT_TITLE_MAX_CHARS + 1);
        let err = validate_print_title(&over).unwrap_err();
        assert!(err.contains("121"), "{err}");
        // Surrounding whitespace does not count toward the cap.
        let padded = format!("   {}   ", "x".repeat(PRINT_TITLE_MAX_CHARS));
        assert!(validate_print_title(&padded).is_ok());
    }
}
