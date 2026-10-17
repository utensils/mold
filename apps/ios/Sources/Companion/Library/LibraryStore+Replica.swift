import Foundation
import MoldClient

/// Ids whose rows changed since the timelines last settled.
struct PendingRows {
    var full = false
    var upserted: Set<PrintID> = []
    var removed: Set<PrintID> = []
    var isEmpty: Bool { !full && upserted.isEmpty && removed.isEmpty }

    mutating func note(_ changed: [PrintID], removed gone: [PrintID] = []) {
        upserted.formUnion(changed)
        upserted.subtract(gone)
        removed.formUnion(gone)
        removed.subtract(changed)
    }
}

// One `GalleryReplica` per machine, and what its deltas do to the rows -- the
// Mac's contract (`macos-native.md`, "Library replicas").
//
// The replica reads and diffs off the main actor and hands each delta to
// `absorb` in order, before the pull that produced it returns. `absorb` only
// edits `live` / `trashed` and notes which ids moved; whoever started the pull
// settles the timelines once, so a refresh of five machines is one settle,
// and a pull that changed nothing settles nothing.
extension LibraryStore {
    /// The replica for this machine: kept across route changes (a roaming
    /// address is the same gallery), replaced when the key, connection mode
    /// or server instance changes, which may be another gallery altogether.
    func replica(for host: MoldHost) -> GalleryReplica {
        let identity = Self.replicaIdentity(host)
        if let existing = replicas[host.id], existing.identity == identity { return existing.replica }
        let id = host.id
        let epoch = cacheEpoch
        let replica = GalleryReplica(hostID: id, store: snapshots.replicas) { [weak self] delta in
            await self?.absorb(delta, on: id, epoch: epoch)
        }
        replicas[id] = (identity, replica)
        return replica
    }

    static func replicaIdentity(_ host: MoldHost) -> String {
        "\(host.apiKey ?? "")|\(host.connectionMode)|\(host.connectionInstanceID ?? "")"
    }

    /// The exact record a pull read through. In-process only.
    static func origin(_ host: MoldHost) -> String { String(host.hashValue) }

    func pullRequest(_ host: MoldHost, full: Bool = false) -> GalleryReplica.Pull {
        let origin = Self.origin(host)
        if hosts.capabilities[host.id]?.hasGalleryChangesFeed == true {
            return GalleryReplica.Pull(useFeed: true, full: full, origin: origin)
        }
        // A machine without a trash has no trash listing to ask for.
        let scope: ReplicaDelta.Scope = hosts.capabilities[host.id]?.trashEnabled == true ? .all : .live
        return .legacy(scope, full: full, origin: origin)
    }

    /// One machine's frames say something changed. A short pause gathers a
    /// burst into one pull; the replica's single flight folds anything that
    /// arrives mid-read into at most one more.
    func tickle(_ id: MoldHost.ID) {
        guard tickles[id] == nil else { return }
        tickles[id] = Task { [weak self] in
            try? await Task.sleep(for: .milliseconds(300))
            guard !Task.isCancelled, let self else { return }
            tickles[id] = nil
            await reload(id)
        }
    }

    /// Reads one machine through its replica. Never settles: callers do.
    func pull(_ host: MoldHost, full: Bool = false) async {
        // A pull starting now answers every frame heard before it.
        tickles.removeValue(forKey: host.id)?.cancel()
        let epoch = cacheEpoch
        let client = hosts.backend(for: host)
        let doing = String(localized: "list its prints")
        do {
            try await replica(for: host).pull(pullRequest(host, full: full), using: client)
            guard epoch == cacheEpoch, hosts.host(host.id) == host, !Task.isCancelled else { return }
            if collectionsStale.contains(host.id) || !collectionInventoryAvailable.contains(host.id) {
                try await readCollections(host, using: client, epoch: epoch)
            }
            hosts.clearFailures(for: host.id, doing: doing)
        } catch is CancellationError {
            // Foreground navigation can end a refresh after prints loaded.
            // Cancellation is not a host failure.
        } catch {
            guard epoch == cacheEpoch, hosts.host(host.id) == host, !Task.isCancelled else { return }
            collectionInventoryAvailable.remove(host.id)
            hosts.report(host, doing: doing, error)
        }
    }

    private func readCollections(_ host: MoldHost, using client: any MoldBackend, epoch: Int) async throws {
        guard canOrganize(on: host.id) else { collectionsStale.remove(host.id); return }
        let visibilityGeneration = collectionVisibility.generation
        let fresh = try await client.collections()
        guard epoch == cacheEpoch, hosts.host(host.id) == host,
              collectionVisibility.generation == visibilityGeneration else { return }
        collectionsStale.remove(host.id)
        collectionInventoryAvailable.insert(host.id)
        guard fresh != collections[host.id] else { return }
        collections[host.id] = fresh
        saveCollections(host.id)
        refreshShelves()
    }

    /// Applies one delta to a machine's rows; the timelines settle later.
    func absorb(_ delta: ReplicaDelta, on id: MoldHost.ID, epoch: Int) {
        guard epoch == cacheEpoch, let machine = hosts.host(id) else { return }
        guard delta.origin == nil || delta.origin == Self.origin(machine) else {
            // Read through a record since replaced: hand everything over again.
            replicas[id]?.replica.redeliver()
            return
        }
        let liveRows = delta.live, trashRows = delta.trashed
        let removed = Set(delta.removed)
        if organizes(delta, on: id) { collectionsStale.insert(id) }
        if delta.scope.contains(.live) || !delta.upserted.isEmpty {
            if delta.reset && delta.scope.contains(.live) {
                live[id] = liveRows
                pendingLive.full = true
            } else {
                let leaving = (delta.scope.contains(.live) ? removed : []).union(trashRows.map(\.filename))
                if !liveRows.isEmpty || !leaving.isEmpty {
                    live[id] = Self.upserting(liveRows, removing: leaving, into: live[id] ?? [])
                    pendingLive.note(liveRows.map { PrintID(host: id, filename: $0.filename) },
                                     removed: leaving.map { PrintID(host: id, filename: $0) })
                }
            }
        }
        if delta.scope.contains(.trash) || !delta.upserted.isEmpty {
            if delta.reset && delta.scope.contains(.trash) {
                trashed[id] = trashRows
                pendingTrash.full = true
            } else {
                let leaving = (delta.scope.contains(.trash) ? removed : []).union(liveRows.map(\.filename))
                if !trashRows.isEmpty || !leaving.isEmpty {
                    trashed[id] = Self.upserting(trashRows, removing: leaving, into: trashed[id] ?? [])
                    pendingTrash.note(trashRows.map { PrintID(host: id, filename: $0.filename) },
                                      removed: leaving.map { PrintID(host: id, filename: $0) })
                }
            }
        }
        // Rows read from a machine must not undo an edit still on its way.
        replayPending(on: id)
    }

    /// Whether a delta can have changed this machine's collections (a shelf's
    /// count or cover): a reset, or a row that is or was filed somewhere.
    private func organizes(_ delta: ReplicaDelta, on id: MoldHost.ID) -> Bool {
        if delta.reset || delta.upserted.contains(where: { !($0.collections ?? []).isEmpty }) { return true }
        guard !delta.removed.isEmpty else { return false }
        let gone = Set(delta.removed)
        return ((live[id] ?? []) + (trashed[id] ?? [])).contains {
            gone.contains($0.filename) && !($0.collections ?? []).isEmpty
        }
    }

    /// One pass over a machine's rows, however many changed. New rows go last.
    static func upserting(_ rows: [GalleryPrint], removing names: Set<String>,
                          into current: [GalleryPrint]) -> [GalleryPrint] {
        guard !rows.isEmpty || !names.isEmpty else { return current }
        var replacements = Dictionary(rows.map { ($0.filename, $0) }, uniquingKeysWith: { $1 })
        var result: [GalleryPrint] = []
        result.reserveCapacity(current.count + replacements.count)
        for print in current where !names.contains(print.filename) {
            result.append(replacements.removeValue(forKey: print.filename) ?? print)
        }
        result.append(contentsOf: rows.filter { replacements[$0.filename] != nil })
        return result
    }

    /// Brings both timelines up to the rows: in place where a merge could not
    /// differ, by a full merge otherwise, and not at all -- no new pool, no
    /// `revision` bump -- when nothing changed.
    func settleTimelines() {
        let order = hosts.hosts
        let livePool = Self.settle(&liveTimeline, &pendingLive, rows: live, order: order)
        let trashTiles = Self.settle(&trashTimeline, &pendingTrash, rows: trashed, order: order)
        guard livePool != nil || trashTiles != nil else { return }
        if let livePool {
            pool = livePool
            knownTags = Array(Set(livePool.flatMap(\.tags)))
                .sorted { $0.localizedStandardCompare($1) == .orderedAscending }
        }
        if let trashTiles { trashPool = trashTiles }
        refreshUnreadMedia()
        revision += 1
    }

    private static func settle(_ timeline: inout LibraryTimeline, _ pending: inout PendingRows,
                               rows: [MoldHost.ID: [GalleryPrint]], order: [MoldHost]) -> [LibraryEntry]? {
        guard !pending.isEmpty else { return nil }
        defer { pending = PendingRows() }
        if !pending.full {
            let (changed, gone) = current(pending, in: rows, order: order)
            if timeline.patch(upserted: changed, removed: gone) { return timeline.tiles }
        }
        // In machine-list order, so which copy leads never depends on
        // dictionary order.
        timeline.rebuild(order.flatMap { host in (rows[host.id] ?? []).map { LibraryEntry(host: host, print: $0) } },
                         localHost: nil)
        return timeline.tiles
    }

    /// The rows as they are NOW for a pending set: an id noted as changed
    /// whose row has since gone is a removal.
    private static func current(_ pending: PendingRows, in rows: [MoldHost.ID: [GalleryPrint]],
                                order: [MoldHost]) -> ([LibraryEntry], [PrintID]) {
        var lookup: [MoldHost.ID: [String: GalleryPrint]] = [:]
        for id in Set(pending.upserted.map(\.host)) {
            lookup[id] = Dictionary((rows[id] ?? []).map { ($0.filename, $0) }, uniquingKeysWith: { $1 })
        }
        let machines = Dictionary(order.map { ($0.id, $0) }, uniquingKeysWith: { $1 })
        var changed: [LibraryEntry] = []
        var gone = Array(pending.removed)
        for id in pending.upserted {
            if let host = machines[id.host], let print = lookup[id.host]?[id.filename] {
                changed.append(LibraryEntry(host: host, print: print))
            } else {
                gone.append(id)
            }
        }
        return (changed, gone)
    }

    /// Machines that left the list leave the grid; their replicas, saved
    /// files and pictures go with them -- and, once per launch, any file a
    /// machine removed in an earlier launch left behind.
    func prune(to hostList: [MoldHost]) {
        let present = Set(hostList.map(\.id))
        let gone = Set(live.keys).union(trashed.keys).union(collections.keys).union(replicas.keys)
            .subtracting(present)
        for id in gone { forget(id) }
        guard !prunedReplicaFiles || !gone.isEmpty else { return }
        prunedReplicaFiles = true
        let snapshots = snapshots
        Task.detached(priority: .utility) { snapshots.prune(keeping: present) }
    }

    private func forget(_ id: MoldHost.ID) {
        collectionInventoryAvailable.remove(id)
        collectionsStale.remove(id)
        tickles.removeValue(forKey: id)?.cancel()
        if live.removeValue(forKey: id) != nil { pendingLive.full = true }
        if trashed.removeValue(forKey: id) != nil { pendingTrash.full = true }
        collections[id] = nil
        if let replica = replicas.removeValue(forKey: id)?.replica {
            Task { await replica.discard() }
        }
        snapshots.remove(id)
        forgot?(id)
    }
}
