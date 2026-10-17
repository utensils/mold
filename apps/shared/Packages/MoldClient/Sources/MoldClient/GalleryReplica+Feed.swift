import Foundation

// Following `GET /api/gallery/changes` page by page.
extension GalleryReplica {
    /// What a pull did to `rows`: the names it touched with their values from
    /// BEFORE the pull, so the delta can drop a change that changed nothing.
    struct Outcome {
        var changed: Set<String> = []
        var previous: [String: GalleryPrint?] = [:]
        /// Each touched name's partition before the pull.
        var previousTrash: [String: Bool] = [:]
        /// Names a consumer spliced on its own (`forget`): emitted as they are
        /// now whether or not this pull saw them change.
        var forced: Set<String> = []
        var serverReset = false
        var scope: ReplicaDelta.Scope

        mutating func record(_ name: String, was old: GalleryPrint?, trash: Bool) {
            if previous[name] == nil {
                previous[name] = .some(old)
                previousTrash[name] = trash
            }
            changed.insert(name)
        }
    }

    /// No real gallery has this many pages; a server looping on one cursor does.
    static let maxPages = 10_000

    /// Reads every page into a local overlay and commits it in one step at
    /// the end, so a cancelled or failed pull -- or one interleaved with a
    /// listing pull on another lane -- never leaves half a log applied and
    /// never has anything to put back.
    func followFeed(using backend: any MoldGalleryBackend) async throws -> Outcome {
        var since: String? = instanceID == nil ? nil : cursor
        var snapshot: [String: GalleryPrint]? = since == nil ? [:] : nil
        var overlay: [String: GalleryPrint?] = [:]
        var serverReset = false
        var instance = instanceID
        for _ in 0..<Self.maxPages {
            let page = try await backend.galleryChanges(since: since, limit: pageLimit)
            try Task.checkCancellation()
            if let known = instanceID, page.instanceId != known { serverReset = true }
            if page.reset || (since != nil && snapshot == nil && page.instanceId != instance) {
                // The cursor means nothing to this server: rebuild from a
                // snapshot, from sequence zero.
                overlay = [:]
                serverReset = serverReset || since != nil
                snapshot = [:]
                if !page.reset {
                    since = nil
                    instance = page.instanceId
                    continue
                }
            }
            instance = page.instanceId
            for change in page.changes {
                if snapshot != nil { snapshot?[change.filename] = change.image } else { overlay[change.filename] = .some(change.image) }
            }
            if page.more, page.cursor == since { throw MoldClientError.malformedResponse }
            since = page.cursor
            guard !page.more else { continue }
            var outcome = Outcome(scope: .all)
            outcome.serverReset = serverReset
            if let snapshot {
                adopt(snapshot, into: &outcome)
            } else {
                for (name, row) in overlay {
                    note(name, in: &outcome)
                    rows[name] = row
                    // The feed's row says which partition it is in.
                    listedInTrash.remove(name)
                }
            }
            cursor = since
            instanceID = instance
            // The listing validators describe a state older than this, and a
            // listing answer that started before it must not land after it.
            if !outcome.changed.isEmpty {
                liveETag = nil
                trashETag = nil
                supersedeListings()
            }
            scheduleSave()
            return outcome
        }
        throw MoldClientError.malformedResponse
    }

    /// Replaces `rows` with a full snapshot, recording only real differences.
    func adopt(_ snapshot: [String: GalleryPrint], into outcome: inout Outcome) {
        for (name, old) in rows where snapshot[name] != old || listedInTrash.contains(name) {
            note(name, in: &outcome)
        }
        for name in snapshot.keys where rows[name] == nil { note(name, in: &outcome) }
        rows = snapshot
        listedInTrash = []
    }

    /// Records a name's value and partition before this pull first touches it.
    func note(_ name: String, in outcome: inout Outcome) {
        outcome.record(name, was: rows[name], trash: isTrash(name))
    }

    /// The delta a pull's outcome means for a consumer, or nil for none.
    func delta(for outcome: Outcome, full: Bool, origin: String? = nil) -> ReplicaDelta? {
        let scope = outcome.scope
        if full || outcome.serverReset || !deliveredScopes.isSuperset(of: scope) {
            revision += 1
            logChange(outcome.changed.union(outcome.forced))
            deliveredScopes.formUnion(scope)
            let whole = rows.values.filter { scope.covers(trash: isTrash($0.filename)) }
                .sorted { $0.filename < $1.filename }
            return ReplicaDelta(upserted: whole, removed: [], reset: true, revision: revision, scope: scope,
                                inTrash: Set(whole.map(\.filename).filter(isTrash)), origin: origin)
        }
        var upserted: [GalleryPrint] = []
        var removed: [String] = []
        for name in outcome.changed.union(outcome.forced).sorted() {
            let old = outcome.previous[name] ?? nil
            let new = rows[name]
            guard outcome.forced.contains(name) || old != new || outcome.previousTrash[name] != isTrash(name)
            else { continue }
            if let new { upserted.append(new) } else { removed.append(name) }

        }
        guard !upserted.isEmpty || !removed.isEmpty else { return nil }
        revision += 1
        logChange(upserted.map(\.filename) + removed)
        return ReplicaDelta(upserted: upserted, removed: removed, reset: false, revision: revision, scope: scope,
                            inTrash: Set(upserted.map(\.filename).filter(isTrash)), origin: origin)
    }
}
