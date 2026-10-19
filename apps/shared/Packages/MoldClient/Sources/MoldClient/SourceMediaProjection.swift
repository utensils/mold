import Foundation

/// The Library's organization/filter authority over exact source owners, cached
/// independently of source search, thumbnail size, scan progress and selection.
@MainActor public final class SourceMediaProjection {
    private var revision: Int?
    private var query: LibraryQuery?
    private var result: [LibraryEntry] = []
    public private(set) var derivations = 0
    public init() {}

    public func owners(in entries: [LibraryEntry], query: LibraryQuery, revision: Int) -> [LibraryEntry] {
        var owners = query
        owners.text = ""
        owners.tokens.removeAll { if case .kind = $0 { true } else { false } }
        if self.revision == revision, self.query == owners { return result }
        var seen = Set<PrintID>()
        result = owners.apply(to: entries).flatMap(\.everyCopy).filter { seen.insert($0.id).inserted }
        self.revision = revision; self.query = owners; derivations += 1
        return result
    }
}
