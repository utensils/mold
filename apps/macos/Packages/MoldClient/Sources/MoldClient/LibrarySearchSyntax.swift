import Foundation

/// What a person can TYPE into the Library's search field beyond words:
/// `is:video`, `tag:cat`, `on:hal9000`, `favourite`. The chips existed and
/// the sidebar and the inspector could add them, but typing `is:mesh`
/// matched nothing -- the field was plain text over prompts, models and
/// tags, and the README promised otherwise (UAT 2026-09-17 #9).
///
/// Pure, so the vocabulary is a test. The pane feeds it what the library
/// holds -- a chip that can only ever match nothing is worse than none.
public enum LibrarySearchSyntax {
    /// The three prefixes. `is:` takes a kind, `tag:` a tag, `on:` a machine.
    public enum Field: String, CaseIterable, Sendable {
        case kind = "is", tag, machine = "on"
    }

    /// `is:video` -> (`.kind`, `video`); `video` -> (nil, `video`).
    public static func split(_ text: String) -> (field: Field?, term: String) {
        let trimmed = text.trimmingCharacters(in: .whitespaces)
        guard let colon = trimmed.firstIndex(of: ":"),
              let field = Field(rawValue: trimmed[..<colon].lowercased())
        else { return (nil, trimmed) }
        return (field, trimmed[trimmed.index(after: colon)...].trimmingCharacters(in: .whitespaces))
    }

    /// The words for each kind: this app's own first, then a person's.
    static let kindWords: [PrintKind: [String]] = [
        .clip: ["clip", "video", "movie", "mp4"],
        .picture: ["picture", "image", "photo", "still", "png"],
        .mesh: ["mesh", "3d", "model", "glb"],
    ]

    /// The kind a word names, in a person's words as well as this app's.
    public static func kind(_ term: String) -> PrintKind? {
        let folded = LibraryEntry.fold(term)
        return PrintKind.allCases.first { kindWords[$0]!.contains(folded) }
    }

    /// The kinds a word STARTS to name -- `vid` is on its way to `video`.
    /// Only behind `is:`, where a kind is the only thing it can be; a bare
    /// `m` in the field is a word, and offering Clip for it (movie, mp4)
    /// would be noise.
    static func kinds(startingWith folded: String, aliases: Bool) -> [PrintKind] {
        PrintKind.allCases.filter { kind in
            aliases ? kindWords[kind]!.contains { $0.hasPrefix(folded) } : kind.rawValue.hasPrefix(folded)
        }
    }

    /// Chips to offer for what has been typed so far.
    ///
    /// A prefix narrows the list to its own field; no prefix offers every
    /// field, as before. `applied` are the chips already in the field, which
    /// would otherwise read as "add this twice".
    public static func suggestions(
        for text: String, machines: [(id: MoldHost.ID, name: String)], tags: [String],
        applied: Set<String>
    ) -> [LibraryToken] {
        let (field, term) = split(text)
        guard !term.isEmpty || field != nil else { return [] }
        let folded = LibraryEntry.fold(term)
        var found: [LibraryToken] = []
        if field == nil || field == .machine {
            found += machines
                .filter { LibraryEntry.fold($0.name).contains(folded) }
                .map { .machine(id: $0.id, name: $0.name) }
        }
        if field == nil || field == .tag {
            found += tags.filter { LibraryEntry.fold($0).contains(folded) }.prefix(5).map { .tag($0) }
        }
        if field == nil || field == .kind {
            found += kinds(startingWith: folded, aliases: field == .kind).map { .kind($0) }
        }
        if field == nil,
           LibraryEntry.fold("favourite").hasPrefix(folded) || LibraryEntry.fold("favorite").hasPrefix(folded) {
            found.append(.favorite)
        }
        return found.filter { !applied.contains($0.id) }
    }

    /// The one chip a submit turns the text into -- only when it names
    /// exactly one thing, so Return on a plain word still searches for it.
    public static func committed(
        _ text: String, machines: [(id: MoldHost.ID, name: String)], tags: [String]
    ) -> LibraryToken? {
        let (field, term) = split(text)
        guard let field, !term.isEmpty else { return nil }
        let folded = LibraryEntry.fold(term)
        switch field {
        case .kind:
            return kind(term).map { .kind($0) }
        case .tag:
            return tags.first { LibraryEntry.fold($0) == folded }.map { .tag($0) }
        case .machine:
            return machines.first { LibraryEntry.fold($0.name) == folded }
                .map { .machine(id: $0.id, name: $0.name) }
        }
    }
}
