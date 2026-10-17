import Foundation

// A machine with no change feed: the conditional full listings, diffed by row
// into the same outcome the feed produces.
extension GalleryReplica {
    func relist(_ scope: ReplicaDelta.Scope, full: Bool,
                using backend: any MoldGalleryBackend) async throws -> Outcome {
        var outcome = Outcome(scope: scope)
        if full {
            liveETag = nil
            trashETag = nil
        }
        // Both reads land before either is applied, so a print moving between
        // the two partitions is one change rather than a removal and an add.
        let liveValidator = liveETag
        let trashValidator = trashETag
        async let live = Self.read(scope.contains(.live)) { try await backend.gallery(etag: liveValidator) }
        async let trash = Self.read(scope.contains(.trash)) { try await backend.trashedPrints(etag: trashValidator) }
        let (liveAnswer, trashAnswer) = try await (live, trash)
        try Task.checkCancellation()
        if case let .fresh(prints, etag)? = liveAnswer {
            absorb(prints, as: .live, into: &outcome)
            liveETag = etag
        }
        if case let .fresh(prints, etag)? = trashAnswer {
            absorb(prints, as: .trash, into: &outcome)
            trashETag = etag
        }
        if case .fresh? = liveAnswer { scheduleSave() } else if case .fresh? = trashAnswer { scheduleSave() }
        return outcome
    }

    private static func read(
        _ wanted: Bool, _ fetch: @Sendable () async throws -> Fetched<[GalleryPrint]>
    ) async throws -> Fetched<[GalleryPrint]>? {
        wanted ? try await fetch() : nil
    }

    /// One listing is the whole truth for its partition: every row it names is
    /// that row now, and every row of that partition it does not name is gone
    /// from it (or moved to the other partition, which its own listing says).
    private func absorb(_ prints: [GalleryPrint], as partition: ReplicaDelta.Scope,
                        into outcome: inout Outcome) {
        var named = Set<String>()
        named.reserveCapacity(prints.count)
        let trash = partition == .trash
        for print in prints {
            named.insert(print.filename)
            guard rows[print.filename] != print || isTrash(print.filename) != trash else { continue }
            note(print.filename, in: &outcome)
            rows[print.filename] = print
            if trash { listedInTrash.insert(print.filename) } else { listedInTrash.remove(print.filename) }
        }
        let gone = rows.keys.filter { partition.covers(trash: isTrash($0)) && !named.contains($0) }
        for name in gone {
            note(name, in: &outcome)
            rows[name] = nil
            listedInTrash.remove(name)
        }
    }
}
