import Foundation

/// Counts depend on library data and filters, never on sync progress, focus or
/// selection. Privacy evidence shares the caller's row/receipt revision instead
/// of hashing every print and lineage edge on each SwiftUI body pass.
@MainActor public final class LibraryQueryCountCache {
    private struct Epoch: Equatable {
        let rows: Int
        let receipts: Int?
    }
    private struct Key: Hashable {
        let query: LibraryQuery
        let partition: LibraryScope
    }
    private var epoch: Epoch?
    private var counts: [Key: Int] = [:]
    public private(set) var derivations = 0
    public init() {}

    public func count(in entries: [LibraryEntry], query: LibraryQuery,
                      partition: LibraryScope = .all, revision: Int, privacyRevision: Int? = nil) -> Int {
        let next = Epoch(rows: revision, receipts: privacyRevision)
        if epoch != next { epoch = next; counts.removeAll(keepingCapacity: true) }
        var key = query
        key.privacyEvidence = []; key.privacyLinks = [:]
        key.sort = .newest
        let cacheKey = Key(query: key, partition: partition)
        if let count = counts[cacheKey] { return count }
        let count = query.apply(to: entries).count
        // Search text is unbounded; retain only a small set of current contexts.
        if counts.count >= 64 { counts.removeAll(keepingCapacity: true) }
        counts[cacheKey] = count; derivations += 1
        return count
    }
}
