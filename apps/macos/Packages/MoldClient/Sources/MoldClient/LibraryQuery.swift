import Foundation

/// Which prints are on screen.
///
/// Filtering happens here rather than on the host because `GET /api/gallery`
/// has no pagination: each machine answers with its whole index and the app
/// holds it. So this runs on every keystroke over thousands of rows, and the
/// folding that makes search forgiving is done ONCE per print, at
/// construction, in `LibraryEntry.searchKey`.
public struct LibraryQuery: Hashable, Sendable {
    public var text: String = ""
    public var tokens: [LibraryToken] = []
    public var sort: LibrarySort = .newest
    /// Collections each machine hides, by host. Their members stay out of the
    /// default grid -- which is what hidden means -- unless the shelf itself
    /// is what was asked for.
    public var hiddenCollectionIDs: [MoldHost.ID: Set<String>] = [:]

    public init() {}

    public var isNarrowed: Bool {
        !tokens.isEmpty || !text.trimmingCharacters(in: .whitespaces).isEmpty
    }

    public func apply(to entries: [LibraryEntry]) -> [LibraryEntry] {
        let showHidden = tokens.contains { if case .collection = $0 { true } else { false } }
        var shown = entries.filter { entry in
            matchesEveryGroup(entry) && (showHidden || !isHidden(entry))
        }
        if !text.trimmingCharacters(in: .whitespaces).isEmpty {
            shown = shown.filter { $0.matches(text) }
        }
        return sorted(shown)
    }

    /// Tokens of DIFFERENT kinds narrow; tokens of the SAME kind widen.
    ///
    /// A print cannot be on two machines at once, so ANDing two machine chips
    /// would always give nothing -- never what someone who added a second one
    /// meant. Two TAGS are the exception that proves the rule: one print can
    /// carry both, so they narrow, and that is what "more words narrow" means
    /// everywhere else in this app.
    private func matchesEveryGroup(_ entry: LibraryEntry) -> Bool {
        var machines: [LibraryToken] = []
        var kinds: [LibraryToken] = []
        var collections: [LibraryToken] = []
        for token in tokens {
            switch token {
            case .machine: machines.append(token)
            case .kind: kinds.append(token)
            case .collection: collections.append(token)
            case let .tag(name):
                guard entry.print.tagList.contains(where: {
                    $0.caseInsensitiveCompare(name) == .orderedSame
                }) else { return false }
            case .favorite:
                guard entry.print.isFavorite else { return false }
            }
        }
        for group in [machines, kinds, collections] where !group.isEmpty {
            guard group.contains(where: { matches($0, entry) }) else { return false }
        }
        return true
    }

    private func matches(_ token: LibraryToken, _ entry: LibraryEntry) -> Bool {
        switch token {
        case let .machine(id, _):
            entry.hostID == id
        case let .kind(kind):
            entry.print.kind == kind
        case let .collection(_, _, ids):
            // The id for THIS print's machine, and no other.
            ids[entry.hostID].map { entry.print.collectionList.contains($0) } ?? false
        case .tag, .favorite:
            true
        }
    }

    private func isHidden(_ entry: LibraryEntry) -> Bool {
        guard let hidden = hiddenCollectionIDs[entry.hostID], !hidden.isEmpty else { return false }
        return entry.print.collectionList.contains(where: hidden.contains)
    }

    /// Ties are broken by filename, because a batch writes several prints in
    /// the same second and an unstable order reshuffles the grid on every
    /// refresh.
    private func sorted(_ entries: [LibraryEntry]) -> [LibraryEntry] {
        entries.sorted { lhs, rhs in
            switch sort {
            case .newest where lhs.print.timestamp != rhs.print.timestamp:
                lhs.print.timestamp > rhs.print.timestamp
            case .oldest where lhs.print.timestamp != rhs.print.timestamp:
                lhs.print.timestamp < rhs.print.timestamp
            case .largest where (lhs.print.sizeBytes ?? 0) != (rhs.print.sizeBytes ?? 0):
                (lhs.print.sizeBytes ?? 0) > (rhs.print.sizeBytes ?? 0)
            default:
                lhs.print.filename.localizedStandardCompare(rhs.print.filename)
                    == .orderedAscending
            }
        }
    }
}
