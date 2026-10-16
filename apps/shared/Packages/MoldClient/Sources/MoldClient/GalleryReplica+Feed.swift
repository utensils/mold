import Foundation

// Following `GET /api/gallery/changes` page by page.
extension GalleryReplica {
    /// What a pull did to `rows`: the names it touched with their values from
    /// BEFORE the pull, so the delta can drop a change that changed nothing.
    struct Outcome {
        var changed: Set<String> = []
        var previous: [String: GalleryPrint?] = [:]
        var serverReset = false
        var scope: ReplicaDelta.Scope

        mutating func record(_ name: String, was old: GalleryPrint?) {
            if previous[name] == nil { previous[name] = .some(old) }
            changed.insert(name)
        }
    }

    /// No real gallery has this many pages; a server looping on one cursor does.
    static let maxPages = 10_000

    func followFeed(using backend: any MoldGalleryBackend) async throws -> Outcome {
        var since: String? = instanceID == nil ? nil : cursor
        var snapshot: [String: GalleryPrint]? = since == nil ? [:] : nil
        var outcome = Outcome(scope: .all)
        var instance = instanceID
        do {
            for _ in 0..<Self.maxPages {
                let page = try await backend.galleryChanges(since: since, limit: pageLimit)
                // A cancelled pull changes nothing (the catch below puts back
                // what earlier pages applied).
                try Task.checkCancellation()
                if let known = instanceID, page.instanceId != known { outcome.serverReset = true }
                if page.reset || (since != nil && snapshot == nil && page.instanceId != instance) {
                    // The cursor means nothing to this server: put back what
                    // earlier pages applied and rebuild from a snapshot.
                    rollback(&outcome)
                    outcome.serverReset = true
                    snapshot = [:]
                    if !page.reset {
                        since = nil
                        instance = page.instanceId
                        continue
                    }
                }
                instance = page.instanceId
                for change in page.changes {
                    if snapshot != nil {
                        snapshot?[change.filename] = change.image
                    } else {
                        outcome.record(change.filename, was: rows[change.filename])
                        rows[change.filename] = change.image
                    }
                }
                if page.more, page.cursor == since { throw MoldClientError.malformedResponse }
                since = page.cursor
                if !page.more {
                    if let snapshot { adopt(snapshot, into: &outcome) }
                    cursor = since
                    instanceID = instance
                    // The listing validators describe a state older than this.
                    if !outcome.changed.isEmpty {
                        liveETag = nil
                        trashETag = nil
                    }
                    scheduleSave()
                    return outcome
                }
            }
            throw MoldClientError.malformedResponse
        } catch {
            rollback(&outcome)
            throw error
        }
    }

    /// Replaces `rows` with a full snapshot, recording only real differences.
    func adopt(_ snapshot: [String: GalleryPrint], into outcome: inout Outcome) {
        for (name, old) in rows where snapshot[name] != old { outcome.record(name, was: old) }
        for name in snapshot.keys where rows[name] == nil { outcome.record(name, was: nil) }
        rows = snapshot
    }

    func rollback(_ outcome: inout Outcome) {
        for (name, old) in outcome.previous { rows[name] = old }
        outcome.previous = [:]
        outcome.changed = []
    }

    /// The delta a pull's outcome means for a consumer, or nil for none.
    func delta(for outcome: Outcome, full: Bool) -> ReplicaDelta? {
        let scope = outcome.scope
        if full || outcome.serverReset || !deliveredScopes.isSuperset(of: scope) {
            revision += 1
            deliveredScopes.formUnion(scope)
            let whole = rows.values.filter { scope.covers($0) }.sorted { $0.filename < $1.filename }
            return ReplicaDelta(upserted: whole, removed: [], reset: true, revision: revision, scope: scope)
        }
        var upserted: [GalleryPrint] = []
        var removed: [String] = []
        for name in outcome.changed.sorted() {
            let old = outcome.previous[name] ?? nil
            let new = rows[name]
            guard old != new else { continue }
            if let new { upserted.append(new) } else { removed.append(name) }
        }
        guard !upserted.isEmpty || !removed.isEmpty else { return nil }
        revision += 1
        return ReplicaDelta(upserted: upserted, removed: removed, reset: false, revision: revision, scope: scope)
    }
}
