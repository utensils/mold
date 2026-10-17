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
        let liveRead = scope.contains(.live) ? startListing(.live) : 0
        let trashRead = scope.contains(.trash) ? startListing(.trash) : 0
        async let live = Self.read(scope.contains(.live)) { try await backend.gallery(etag: liveValidator) }
        async let trash = Self.read(scope.contains(.trash)) { try await backend.trashedPrints(etag: trashValidator) }
        let (liveAnswer, trashAnswer) = try await (live, trash)
        try Task.checkCancellation()
        if case let .fresh(prints, etag)? = liveAnswer, finishListing(.live, liveRead) {
            absorb(prints, as: .live, into: &outcome)
            liveETag = etag
        }
        if case let .fresh(prints, etag)? = trashAnswer, finishListing(.trash, trashRead) {
            absorb(prints, as: .trash, into: &outcome)
            trashETag = etag
        }
        if case .fresh? = liveAnswer { scheduleSave() } else if case .fresh? = trashAnswer { scheduleSave() }
        return outcome
    }

    private func startListing(_ partition: ReplicaDelta.Scope) -> Int {
        listingsStarted[partition.rawValue, default: 0] += 1
        return listingsStarted[partition.rawValue, default: 0]
    }

    /// Whether this read is still the newest word on its partition.
    private func finishListing(_ partition: ReplicaDelta.Scope, _ read: Int) -> Bool {
        guard read > listingsApplied[partition.rawValue, default: 0] else { return false }
        listingsApplied[partition.rawValue] = read
        return true
    }

    /// A feed commit is newer than every listing read already in flight.
    func supersedeListings() {
        for partition in [ReplicaDelta.Scope.live, .trash] {
            listingsApplied[partition.rawValue] = listingsStarted[partition.rawValue, default: 0]
        }
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
