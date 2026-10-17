import Foundation
import MoldClient

// One `GalleryReplica` per machine, and what its deltas do to the rows.
//
// The replica does the reading and the diffing off the main actor and hands
// each delta to `absorb` in order, before the pull that produced it returns.
// `absorb` only edits `perHost` / `trashPerHost` and notes which ids moved;
// the timelines are settled once, by whoever started the pull, so a refresh
// of five machines is one settle rather than five merges.
@MainActor
extension LibraryStore {
    /// The replica for this machine. Kept across route changes -- a roaming
    /// address or a refreshed endpoint list is the same gallery -- and
    /// replaced only when the credential or the server identity changes,
    /// which may be another gallery altogether. Each pull is stamped with the
    /// exact record it read through (`origin`), and `absorb` refuses an
    /// answer read through a record that has since been replaced -- the rule
    /// every listing here has always had -- asking the replica to hand its
    /// whole state over again next time instead.
    func replica(for host: MoldHost) -> GalleryReplica {
        let identity = Self.replicaIdentity(host)
        if let existing = replicas[host.id], existing.identity == identity { return existing.replica }
        let id = host.id
        let replica = GalleryReplica(hostID: id, store: replicaStore) { [weak self] delta in
            await self?.absorb(delta, on: id)
        }
        replicas[id] = (identity, replica)
        return replica
    }

    static func replicaIdentity(_ host: MoldHost) -> String {
        "\(host.apiKey ?? "")|\(host.connectionMode)|\(host.connectionInstanceID ?? "")"
    }

    /// The record a pull read through. In-process only, like the fence.
    static func origin(_ host: MoldHost) -> String { String(host.hashValue) }

    /// A live frame spliced these rows in place; the replica must not believe
    /// it still knows them (see `GalleryReplica.forget`).
    func spliced(_ host: MoldHost.ID, _ names: [String]) {
        replicas[host]?.replica.forget(names)
    }

    /// Whether this machine is followed by its change log rather than listed.
    func followsFeed(_ host: MoldHost.ID) -> Bool {
        hosts.capabilities[host]?.hasGalleryChangesFeed == true
    }

    func pullRequest(_ host: MoldHost, _ scope: ReplicaDelta.Scope, full: Bool = false) -> GalleryReplica.Pull {
        let origin = Self.origin(host)
        return followsFeed(host.id) ? GalleryReplica.Pull(useFeed: true, full: full, origin: origin)
            : .legacy(scope, full: full, origin: origin)
    }

    /// Every machine at once; the timelines settle once all have answered.
    func pullAll(_ scope: ReplicaDelta.Scope, doing verb: String) async {
        await withTaskGroup(of: (MoldHost, (any Error)?).self) { group in
            for host in hosts.hosts {
                let replica = replica(for: host)
                let client = hosts.backend(for: host)
                let request = pullRequest(host, scope)
                group.addTask {
                    do {
                        try await replica.pull(request, using: client)
                        return (host, nil)
                    } catch { return (host, error) }
                }
            }
            for await (host, error) in group {
                guard !Task.isCancelled, hosts.host(host.id) == host else { continue }
                // A host going down must not erase what it already showed us.
                if let error { hosts.report(error, on: host.id, doing: verb) }
                // Scoped: this passive listing runs right after other actions
                // and must not clear what one of THOSE just reported.
                else { hosts.succeeded(on: host.id, doing: verb) }
            }
        }
        settleTimelines()
    }

    /// One machine's frames say something changed: read it, coalesced by the
    /// replica's single flight -- a burst of frames is one read and at most
    /// one behind it, never one rebuild per frame.
    func tickle(_ id: MoldHost.ID) {
        guard let machine = hosts.host(id) else { return }
        let replica = replica(for: machine)
        let client = hosts.backend(for: machine)
        let request = pullRequest(machine, .all)
        Task {
            do {
                try await replica.pull(request, using: client)
                if hosts.host(id) == machine { hosts.succeeded(on: id, doing: "list its prints") }
            } catch {
                if hosts.host(id) == machine { hosts.report(error, on: id, doing: "list its prints") }
            }
            settleTimelines()
        }
    }

    /// The next legacy listing of these machines is a full one. Replaces
    /// clearing the store's own ETags: the replica owns its validators.
    func invalidateListings(_ hostIDs: [MoldHost.ID]? = nil, _ scope: ReplicaDelta.Scope = .all) {
        for id in hostIDs ?? Array(replicas.keys) { replicas[id]?.replica.invalidateValidators(scope) }
    }

    /// Applies one delta to a machine's rows. Never touches the timelines --
    /// see `settleTimelines`.
    func absorb(_ delta: ReplicaDelta, on id: MoldHost.ID) {
        guard let machine = hosts.host(id) else { return }
        guard delta.origin == nil || delta.origin == Self.origin(machine) else {
            replicas[id]?.replica.redeliver()
            // Through the current record, so the redelivery is not left
            // waiting for the next unrelated event.
            tickle(id)
            return
        }
        writingRows = true
        defer { writingRows = false }
        let entries = delta.upserted.map { LibraryEntry(host: machine, print: $0) }
        // The replica says which partition each row is in (`inTrash`): an
        // older machine's trash listing need not stamp `trashed_at`.
        let live = entries.filter { !delta.inTrash.contains($0.print.filename) }
        let trash = entries.filter { delta.inTrash.contains($0.print.filename) }
        let removed = Set(delta.removed)
        if delta.scope.contains(.live) || !entries.isEmpty {
            let current = perHost[id] ?? []
            if delta.reset && delta.scope.contains(.live) {
                let asked = relistAsked.removeValue(forKey: id)
                perHost[id] = asked.map { RelistMerge.merged(answer: live, onScreen: current, asked: $0) } ?? live
                pendingLive.full = true
            } else {
                let leaving = (delta.scope.contains(.live) ? removed : []).union(trash.map(\.print.filename))
                if !live.isEmpty || !leaving.isEmpty {
                    perHost[id] = Self.upserting(live, removing: leaving, into: current)
                    pendingLive.note(live.map(\.id), removed: leaving.map { PrintID(host: id, filename: $0) })
                }
            }
        }
        if delta.scope.contains(.trash) || !entries.isEmpty {
            let current = trashPerHost[id] ?? []
            if delta.reset && delta.scope.contains(.trash) {
                trashPerHost[id] = trash
                pendingTrash.full = true
            } else {
                let leaving = (delta.scope.contains(.trash) ? removed : []).union(live.map(\.print.filename))
                if !trash.isEmpty || !leaving.isEmpty {
                    trashPerHost[id] = Self.upserting(trash, removing: leaving, into: current)
                    pendingTrash.note(trash.map(\.id), removed: leaving.map { PrintID(host: id, filename: $0) })
                }
            }
        }
        // Rows read from a machine must not undo an edit still on its way to
        // it. Replayed locally only; the timelines settle after.
        deferredSettles += 1
        mutations.replayPending(on: id, in: self)
        deferredSettles -= 1
    }

    /// One pass over a machine's rows, however many changed.
    static func upserting(_ rows: [LibraryEntry], removing names: Set<String>,
                          into current: [LibraryEntry]) -> [LibraryEntry] {
        guard !rows.isEmpty || !names.isEmpty else { return current }
        var replacements = Dictionary(rows.map { ($0.print.filename, $0) }, uniquingKeysWith: { $1 })
        var result: [LibraryEntry] = []
        result.reserveCapacity(current.count + replacements.count)
        for entry in current where !names.contains(entry.print.filename) {
            result.append(replacements.removeValue(forKey: entry.print.filename) ?? entry)
        }
        result.append(contentsOf: rows.filter { replacements[$0.print.filename] != nil })
        return result
    }

    /// A machine left the list: its replica, and its saved file, go with it.
    func discardReplicas(keeping live: Set<MoldHost.ID>) {
        var removed = false
        for (id, entry) in replicas where !live.contains(id) {
            replicas[id] = nil
            removed = true
            Task { await entry.replica.discard() }
        }
        // And any file a machine removed in an earlier launch left behind.
        guard let replicaStore, removed || !prunedReplicaFiles else { return }
        prunedReplicaFiles = true
        Task.detached(priority: .utility) { replicaStore.prune(keeping: live) }
    }

    /// Writes every replica's state now, for quitting.
    func flushReplicas() async {
        for (_, entry) in replicas { await entry.replica.flushIfDirty() }
    }
}
