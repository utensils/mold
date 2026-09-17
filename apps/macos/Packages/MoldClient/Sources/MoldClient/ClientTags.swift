import Foundation

/// The tags a client submits for a titled print, and the one it added.
///
/// A port of `mold_core::organization::compose_client_tags`
/// (`crates/mold-core/src/organization.rs:107-148`) and of `title_slug`
/// (`crates/mold-core/src/print_title.rs:33`), which are CLIENT policy on
/// purpose -- the server never auto-tags, because it cannot tell a title a
/// person typed from one a script generated. Ported rather than invented so
/// this app files a print exactly the way `mold run --title` does.
public enum ClientTags {
    /// `MAX_REQUEST_TAGS` (`organization.rs:28`).
    public static let maxTags = 20
    /// `MAX_TAG_CHARS` (`organization.rs:19`).
    public static let maxTagChars = 64
    /// `PRINT_TITLE_MAX_CHARS` (`print_title.rs:23`).
    public static let titleMaxChars = 120
    /// `TITLE_SLUG_MAX_LEN` (`print_title.rs:20`).
    public static let titleSlugMaxBytes = 40

    /// The client-side mirror of `normalize_request_tags`
    /// (`organization.rs:59-78`): interior whitespace runs collapsed to one
    /// space, empties dropped, case-insensitive duplicates collapsed with the
    /// FIRST spelling kept, order preserved, capped at `maxTags`. Where the
    /// server refuses an over-long or over-full list outright, this trims and
    /// truncates instead -- the app validates as you type rather than
    /// erroring after a render.
    public static func normalize(_ raw: [String]) -> [String] {
        var seenFolded = Set<String>()
        var out: [String] = []
        for tag in raw {
            let collapsed = tag.split(whereSeparator: \.isWhitespace).joined(separator: " ")
            guard !collapsed.isEmpty else { continue }
            let capped = collapsed.count > maxTagChars ? String(collapsed.prefix(maxTagChars)) : collapsed
            let folded = capped.lowercased()
            guard !seenFolded.contains(folded) else { continue }
            seenFolded.insert(folded)
            out.append(capped)
        }
        return out.count > maxTags ? Array(out.prefix(maxTags)) : out
    }

    /// Lowercase ASCII alphanumerics kept, everything else a `-`, runs
    /// collapsed, ends trimmed, capped at `titleSlugMaxBytes` bytes and
    /// re-trimmed so the cut never leaves a dangling dash. `nil` when nothing
    /// survives. Ports `print_title::slug_with_cap`.
    public static func titleSlug(_ title: String) -> String? {
        var slug = ""
        var pendingDash = false
        for ch in title {
            if ch.isASCII, ch.isLetter || ch.isNumber {
                if pendingDash, !slug.isEmpty {
                    if slug.utf8.count + 1 >= titleSlugMaxBytes { break }
                    slug.append("-")
                }
                pendingDash = false
                if slug.utf8.count >= titleSlugMaxBytes { break }
                slug += ch.lowercased()
            } else {
                pendingDash = true
            }
        }
        let trimmed = slug.trimmingCharacters(in: CharacterSet(charactersIn: "-"))
        return trimmed.isEmpty ? nil : trimmed
    }

    /// Compose the tag list a client submits for a print, optionally adding
    /// the title's slug as a tag. Mirrors `compose_client_tags` exactly: the
    /// auto tag is skipped when the switch is off, when the title has no
    /// usable slug, when it is already one of the explicit tags
    /// (case-insensitively), or when adding it would push the list past
    /// `maxTags` -- the slug is dropped rather than bumping something the
    /// user actually typed.
    public static func compose(
        explicit: [String], title: String?, autoTagTitle: Bool
    ) -> (tags: [String], autoTagged: String?) {
        let tags = normalize(explicit)
        guard autoTagTitle, let title, let slug = titleSlug(title) else {
            return (tags, nil)
        }
        guard !tags.contains(where: { $0.caseInsensitiveCompare(slug) == .orderedSame }) else {
            return (tags, nil)
        }
        guard tags.count < maxTags else {
            return (tags, nil)
        }
        return (tags + [slug], slug)
    }
}
