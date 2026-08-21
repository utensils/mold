//! Creation-time library organization: the tag and collection contract a
//! generation request may carry.
//!
//! Library organization (V3) is *user-owned row state* once a print exists —
//! `mold_db::organization` is its only writer. But a print can also arrive
//! **already** tagged and filed, chosen at Create time: `GenerateRequest`
//! (and the chain wire's stitched output) carry `tags` and `collection`,
//! which are validated here at admission, embedded into
//! `OutputMetadata.tags` / `.collection`, and seeded onto the row exactly
//! once at insert.
//!
//! The normalization rules live in mold-core rather than mold-db because
//! admission has to refuse a bad tag *before* any model work is paid for,
//! and mold-core cannot depend on mold-db. `mold_db::organization`
//! delegates to these functions so the two can never drift — a contract
//! test in that module pins the delegation.

/// Longest tag name accepted, in characters (after normalization).
pub const MAX_TAG_CHARS: usize = 64;
/// Longest collection name accepted, in characters (after whitespace
/// normalization).
pub const MAX_COLLECTION_NAME_CHARS: usize = 120;
/// Longest collection slug produced by [`collection_slug`].
pub const MAX_COLLECTION_SLUG_CHARS: usize = 80;
/// Most tags one generation request may file a print under. Counted after
/// normalization (empties dropped, case-insensitive duplicates collapsed),
/// so a client that repeats a tag is not punished for it.
pub const MAX_REQUEST_TAGS: usize = 20;

/// Normalize a tag name: trim, collapse interior whitespace runs to a single
/// space. Returns `Ok(None)` for an empty result (callers drop those) and an
/// error for control characters or over-long names.
///
/// Whitespace controls (tab, newline) are collapsed rather than refused —
/// they are indistinguishable from a space once collapsed. Every other
/// control character (NUL, escape, …) has no place in a tag.
pub fn normalize_tag_name(raw: &str) -> Result<Option<String>, String> {
    if raw.chars().any(|c| c.is_control() && !c.is_whitespace()) {
        return Err("tag names must not contain control characters".to_string());
    }
    let collapsed = raw.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.is_empty() {
        return Ok(None);
    }
    if collapsed.chars().count() > MAX_TAG_CHARS {
        return Err(format!(
            "tag names must be at most {MAX_TAG_CHARS} characters"
        ));
    }
    Ok(Some(collapsed))
}

/// Normalize a request's tag list: drop empties, collapse case-insensitive
/// duplicates (first spelling wins), preserve order, and refuse more than
/// [`MAX_REQUEST_TAGS`] distinct tags.
///
/// An invalid tag is a hard error (422 at admission); an empty or duplicate
/// one is dropped silently, because neither is a mistake a user can see.
pub fn normalize_request_tags(raw: &[String]) -> Result<Vec<String>, String> {
    let mut seen: Vec<String> = Vec::new();
    let mut out: Vec<String> = Vec::new();
    for name in raw {
        let Some(normalized) = normalize_tag_name(name)? else {
            continue;
        };
        let folded = normalized.to_lowercase();
        if seen.iter().any(|existing| existing == &folded) {
            continue;
        }
        seen.push(folded);
        out.push(normalized);
    }
    if out.len() > MAX_REQUEST_TAGS {
        return Err(format!(
            "a request may carry at most {MAX_REQUEST_TAGS} tags; this one has {}",
            out.len()
        ));
    }
    Ok(out)
}

/// One client's decision about what tags to submit for a titled print.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ComposedClientTags {
    /// The tags to send, normalized and deduplicated.
    pub tags: Vec<String>,
    /// The tag added from the title, when one was. Present only so the
    /// client can disclose it — a tag the user did not type must be visible,
    /// not a surprise discovered later in the Library.
    pub auto_tagged: Option<String>,
}

/// Compose the tag list a CLI or TUI submits for a print, optionally adding
/// the title's slug as a tag.
///
/// This is a **client** policy on purpose. The server never auto-tags: it
/// cannot tell a title a person typed from one a script generated, and a
/// host silently adding tags to every print that crosses it would be
/// surprising from any other machine on the fleet. So the decision is made
/// where the intent is known, and travels as an ordinary explicit tag.
///
/// The auto tag is the title's [`crate::title_slug`] — the same lossy ASCII
/// form the filename carries — so "Smurf Village" files under
/// `smurf-village` and matches the print's name in a file browser. It is
/// skipped when the user already asked for that tag (case-insensitively),
/// when the title has no usable slug, and whenever `auto_tag_title` is off.
pub fn compose_client_tags(
    explicit: &[String],
    title: Option<&str>,
    auto_tag_title: bool,
) -> Result<ComposedClientTags, String> {
    let mut tags = normalize_request_tags(explicit)?;
    let auto_tagged = auto_tag_title
        .then(|| title.and_then(crate::title_slug))
        .flatten()
        .filter(|slug| !tags.iter().any(|tag| tag.eq_ignore_ascii_case(slug)));
    if let Some(slug) = auto_tagged.as_ref() {
        tags.push(slug.clone());
        // The explicit list was already at the cap; re-check rather than
        // submitting a request admission would refuse.
        if tags.len() > MAX_REQUEST_TAGS {
            return Err(format!(
                "adding the title tag '{slug}' would exceed the {MAX_REQUEST_TAGS}-tag limit; \
                 drop a tag or pass --no-auto-tag"
            ));
        }
    }
    Ok(ComposedClientTags { tags, auto_tagged })
}

/// Slug for a collection name: lowercase ASCII, `[a-z0-9]` kept, every
/// other character becomes `-`, runs collapsed, ends trimmed, at most
/// [`MAX_COLLECTION_SLUG_CHARS`]. Same algorithm as
/// [`crate::title_slug`] with a longer cap. `None` when nothing survives.
///
/// The slug is how clients merge collections of the same name across hosts,
/// so it is the identity a create-by-name resolves against.
pub fn collection_slug(name: &str) -> Option<String> {
    let mut slug = String::with_capacity(name.len());
    let mut pending_dash = false;
    for ch in name.chars() {
        let lowered = ch.to_ascii_lowercase();
        if lowered.is_ascii_alphanumeric() {
            if pending_dash && !slug.is_empty() {
                slug.push('-');
            }
            pending_dash = false;
            slug.push(lowered);
        } else {
            pending_dash = true;
        }
        if slug.len() >= MAX_COLLECTION_SLUG_CHARS {
            break;
        }
    }
    let slug: String = slug.chars().take(MAX_COLLECTION_SLUG_CHARS).collect();
    let slug = slug.trim_matches('-').to_string();
    if slug.is_empty() {
        None
    } else {
        Some(slug)
    }
}

/// Validate a collection name, returning `(normalized name, slug)`.
///
/// Whitespace runs collapse, control characters are refused, the name is
/// capped at [`MAX_COLLECTION_NAME_CHARS`], and it must contain at least one
/// letter or digit so a slug exists to merge on.
pub fn validate_collection_name(raw: &str) -> Result<(String, String), String> {
    if raw.chars().any(|c| c.is_control() && !c.is_whitespace()) {
        return Err("collection names must not contain control characters".to_string());
    }
    let name = raw.split_whitespace().collect::<Vec<_>>().join(" ");
    if name.is_empty() {
        return Err("collection name must not be empty".to_string());
    }
    if name.chars().count() > MAX_COLLECTION_NAME_CHARS {
        return Err(format!(
            "collection names must be at most {MAX_COLLECTION_NAME_CHARS} characters"
        ));
    }
    let slug = collection_slug(&name)
        .ok_or_else(|| "collection name must contain at least one letter or digit".to_string())?;
    Ok((name, slug))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tag_normalization_trims_and_collapses_whitespace() {
        assert_eq!(
            normalize_tag_name("  smurf   village  ")
                .unwrap()
                .as_deref(),
            Some("smurf village")
        );
        assert_eq!(
            normalize_tag_name("line\nbreak").unwrap().as_deref(),
            Some("line break")
        );
    }

    #[test]
    fn tag_normalization_drops_empty_and_refuses_controls_and_length() {
        assert_eq!(normalize_tag_name("").unwrap(), None);
        assert_eq!(normalize_tag_name("   \t ").unwrap(), None);
        assert!(normalize_tag_name("nul\0").is_err());
        assert!(normalize_tag_name("esc\u{1b}[0m").is_err());
        let exact = "é".repeat(MAX_TAG_CHARS);
        assert_eq!(
            normalize_tag_name(&exact).unwrap().as_deref(),
            Some(exact.as_str())
        );
        assert!(normalize_tag_name(&"é".repeat(MAX_TAG_CHARS + 1)).is_err());
    }

    #[test]
    fn request_tags_dedupe_case_insensitively_keeping_first_spelling() {
        let tags = normalize_request_tags(&[
            "Smurfs".into(),
            "smurfs".into(),
            "  SMURFS ".into(),
            "village".into(),
        ])
        .unwrap();
        assert_eq!(tags, vec!["Smurfs".to_string(), "village".to_string()]);
    }

    #[test]
    fn request_tags_drop_empties_silently() {
        let tags = normalize_request_tags(&["".into(), "  ".into(), "keep".into()]).unwrap();
        assert_eq!(tags, vec!["keep".to_string()]);
    }

    /// The cap counts what will actually be applied, so 30 spellings of one
    /// tag is one tag — but 21 distinct ones is a 422.
    #[test]
    fn request_tags_cap_counts_distinct_tags() {
        let repeated: Vec<String> = (0..30).map(|_| "same".to_string()).collect();
        assert_eq!(normalize_request_tags(&repeated).unwrap().len(), 1);

        let at_cap: Vec<String> = (0..MAX_REQUEST_TAGS).map(|i| format!("t{i}")).collect();
        assert_eq!(
            normalize_request_tags(&at_cap).unwrap().len(),
            MAX_REQUEST_TAGS
        );

        let over: Vec<String> = (0..MAX_REQUEST_TAGS + 1).map(|i| format!("t{i}")).collect();
        let err = normalize_request_tags(&over).unwrap_err();
        assert!(err.contains("21"), "{err}");
    }

    #[test]
    fn request_tags_propagate_an_invalid_tag_as_an_error() {
        assert!(normalize_request_tags(&["fine".into(), "bad\0".into()]).is_err());
        assert!(normalize_request_tags(&["x".repeat(MAX_TAG_CHARS + 1)]).is_err());
    }

    // ── client-side auto-tagging ────────────────────────────────────────

    #[test]
    fn auto_tag_adds_the_title_slug_and_reports_it() {
        let composed =
            compose_client_tags(&["village".into()], Some("Smurf Village"), true).unwrap();
        assert_eq!(
            composed.tags,
            vec!["village".to_string(), "smurf-village".to_string()]
        );
        assert_eq!(composed.auto_tagged.as_deref(), Some("smurf-village"));
    }

    #[test]
    fn auto_tag_is_skipped_when_disabled_untitled_or_slugless() {
        for (title, enabled) in [
            (Some("Smurf Village"), false),
            (None, true),
            (Some(""), true),
            (Some("日本語"), true),
            (Some("!!!"), true),
        ] {
            let composed = compose_client_tags(&["village".into()], title, enabled).unwrap();
            assert_eq!(
                composed.tags,
                vec!["village".to_string()],
                "{title:?}/{enabled}"
            );
            assert_eq!(composed.auto_tagged, None, "{title:?}/{enabled}");
        }
    }

    /// A tag the user already typed is not duplicated, and nothing is
    /// reported as auto-added — they chose it.
    #[test]
    fn auto_tag_does_not_duplicate_a_tag_the_user_already_asked_for() {
        let composed =
            compose_client_tags(&["Smurf-Village".into()], Some("Smurf Village"), true).unwrap();
        assert_eq!(composed.tags, vec!["Smurf-Village".to_string()]);
        assert_eq!(composed.auto_tagged, None);
    }

    /// The auto tag must never push a request past the cap into a 422 the
    /// user cannot explain; name the remedy instead.
    #[test]
    fn auto_tag_refuses_rather_than_overflowing_the_cap() {
        let full: Vec<String> = (0..MAX_REQUEST_TAGS).map(|i| format!("t{i}")).collect();
        let err = compose_client_tags(&full, Some("Smurf Village"), true).unwrap_err();
        assert!(err.contains("--no-auto-tag"), "{err}");
        // Without the auto tag the same list is fine.
        assert_eq!(
            compose_client_tags(&full, Some("Smurf Village"), false)
                .unwrap()
                .tags
                .len(),
            MAX_REQUEST_TAGS
        );
    }

    #[test]
    fn auto_tag_still_normalizes_the_explicit_list() {
        let composed = compose_client_tags(
            &["  Smurfs  ".into(), "smurfs".into(), "".into()],
            None,
            true,
        )
        .unwrap();
        assert_eq!(composed.tags, vec!["Smurfs".to_string()]);
        assert!(compose_client_tags(&["bad\0".into()], None, true).is_err());
    }

    #[test]
    fn collection_names_normalize_and_slug() {
        let (name, slug) = validate_collection_name("  Smurf   Village  ").unwrap();
        assert_eq!(name, "Smurf Village");
        assert_eq!(slug, "smurf-village");
    }

    #[test]
    fn collection_names_refuse_empty_controls_length_and_slugless() {
        assert!(validate_collection_name("").is_err());
        assert!(validate_collection_name("   ").is_err());
        assert!(validate_collection_name("nul\0").is_err());
        assert!(validate_collection_name(&"x".repeat(MAX_COLLECTION_NAME_CHARS + 1)).is_err());
        // Slugless: no ASCII alphanumeric survives.
        assert!(validate_collection_name("日本語").is_err());
        assert!(validate_collection_name("!!!").is_err());
    }

    #[test]
    fn collection_slug_matches_the_title_slug_algorithm_with_a_longer_cap() {
        assert_eq!(
            collection_slug("Hello, World!").as_deref(),
            Some("hello-world")
        );
        assert_eq!(collection_slug("--").as_deref(), None);
        let long = "word ".repeat(40);
        let slug = collection_slug(&long).unwrap();
        assert!(slug.len() <= MAX_COLLECTION_SLUG_CHARS, "{slug}");
        assert!(!slug.ends_with('-'), "{slug}");
    }
}
