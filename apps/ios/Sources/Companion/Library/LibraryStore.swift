import Foundation
import MoldClient

/// Every machine's prints, merged into one timeline (DESIGN.md §5.2).
///
/// Holds DATA only: which shelf a window shows, what it searched for and what
/// it selected are that window's own state. A print held on several machines
/// is one entry (`LibraryMerge`, the Mac's rule). Listings are fetched with
/// the machine's ETag, so an unchanged library costs a 304 and no decoding.
@Observable
final class LibraryStore {
    var newMedia = LibraryNewMedia()
    var machineID: MoldHost.ID?
    var collectionInventoryAvailable: Set<MoldHost.ID> = []
    let collectionVisibility = CollectionVisibilityLedger.load(from: .standard, key: "library.collectionVisibility")
    var reconcilingCollectionVisibility = false
    var collectionInventory: [MoldHost.ID: [Collection]] { collections }

    func markLibrarySeen() {
        guard !live.isEmpty else { return }
        newMedia.markSeen(pool.map { $0.print.filename })
    }

    private(set) var pool: [LibraryEntry] = []
    private(set) var trashPool: [LibraryEntry] = []
    private(set) var shelves: [CollectionShelf] = []
    private(set) var knownTags: [String] = []
    /// Bumped whenever the pool changes; `LibraryShowingCache` keys on it.
    private(set) var revision = 0
    private(set) var isLoading = false
    /// Machines whose prints are showing from this device's saved copy
    /// because they are not answering right now.
    var offlineHosts: [MoldHost] {
        hosts.hosts.filter { host in
            // Only a definite answer: before a machine has been asked
            // (launch, return to the app) it is not "down".
            switch hosts.reachability(of: host) {
            case .down, .needsKey: !(live[host.id] ?? []).isEmpty
            case .unknown, .checking, .up: false
            }
        }
    }
    /// Told when a machine's library is dropped (the machine was removed).
    @ObservationIgnored var forgot: ((MoldHost.ID) -> Void)?
    /// The last snapshot write per machine: the next waits for it, so two
    /// quick listings can never land on disk in the wrong order.
    @ObservationIgnored private var writes: [MoldHost.ID: Task<Void, Never>] = [:]
    @ObservationIgnored private var cacheEpoch = 0

    @ObservationIgnored let hosts: HostStore
    @ObservationIgnored private var live: [MoldHost.ID: [GalleryPrint]] = [:]
    @ObservationIgnored private var trashed: [MoldHost.ID: [GalleryPrint]] = [:]
    @ObservationIgnored private var collections: [MoldHost.ID: [Collection]] = [:]
    @ObservationIgnored private var etags: [MoldHost.ID: String] = [:]
    @ObservationIgnored private var trashEtags: [MoldHost.ID: String] = [:]
    @ObservationIgnored private var pending: [MoldHost.ID: Task<Void, Never>] = [:]
    @ObservationIgnored let snapshots: LibrarySnapshots
    // Edits (`LibraryStore+Edits`).
    @ObservationIgnored var outbox = MutationOutbox()
    @ObservationIgnored var draining: Set<MoldHost.ID> = []
    @ObservationIgnored var lastFailure: [MoldHost.ID: Error] = [:]
    /// The change Undo would put back.
    var lastEdit: PrintEdit?
    /// The focused window's, so ⌘Z and shake-to-undo reach `undo()`.
    @ObservationIgnored weak var undoManager: UndoManager?

    init(hosts: HostStore, snapshots: LibrarySnapshots = .standard) {
        self.hosts = hosts
        self.snapshots = snapshots
        hosts.listen { [weak self] id, event in self?.heard(event, from: id) }
    }

    /// The saved copy of every machine's library, shown before any machine is
    /// asked (and when none answers). Decoded off the main thread.
    func restoreSaved() async {
        // Only machines not already listed: after the first launch this is
        // nothing, and nothing is decoded.
        let ids = hosts.hosts.map(\.id).filter { live[$0] == nil }
        guard !ids.isEmpty else { return }
        let epoch = cacheEpoch
        let snapshots = snapshots
        let saved = await Task.detached(priority: .userInitiated) {
            ids.compactMap { id in snapshots.load(id).map { (id, $0) } }
        }.value
        acceptRestored(saved, from: epoch)
    }

    /// The detached decoder may finish after Clear Cache. Only its original
    /// cache generation is allowed to put those rows and ETags back in memory.
    func acceptRestored(_ saved: [(MoldHost.ID, LibrarySnapshot)], from epoch: Int) {
        guard epoch == cacheEpoch else { return }
        for (id, snapshot) in saved where live[id] == nil {
            live[id] = snapshot.prints
            trashed[id] = snapshot.trashed
            collections[id] = snapshot.collections
            etags[id] = snapshot.etag
            trashEtags[id] = snapshot.trashEtag
        }
        if !saved.isEmpty { rebuild() }
    }

    /// Every machine that answers, at once; one merge at the end.
    func reload() async {
        isLoading = true
        defer { isLoading = false }
        let targets = hosts.upHosts
        await withTaskGroup(of: Void.self) { group in
            for host in targets { group.addTask { await self.fetch(host) } }
        }
        // A machine that left the list leaves the grid. One that stopped
        // answering keeps its saved prints (`offlineHosts`).
        let present = Set(hosts.hosts.map(\.id))
        for id in Set(live.keys).subtracting(present) { forget(id) }
        rebuild()
        await reconcileCollectionVisibility()
    }

    func reload(_ id: MoldHost.ID) async {
        guard let host = hosts.host(id), hosts.isUp(host) else { return }
        await fetch(host)
        rebuild()
        await reconcileCollectionVisibility()
    }

    /// Clear the device's saved listings, including their ETags. Prints from
    /// machines still answering remain on screen until the next refresh;
    /// offline-only prints leave immediately because their saved copy is gone.
    func clearOfflineCache() async {
        cacheEpoch += 1
        for task in pending.values { task.cancel() }
        pending.removeAll()
        await waitForSnapshotWrites()
        writes.removeAll()
        snapshots.purge()
        etags.removeAll()
        trashEtags.removeAll()
        for host in hosts.hosts where !hosts.isUp(host) {
            live[host.id] = nil
            trashed[host.id] = nil
            collections[host.id] = nil
        }
        rebuild()
    }

    /// Await the writes already queued when this barrier starts. Capturing
    /// their tasks keeps later listings from changing what completion means.
    func waitForSnapshotWrites() async {
        let pendingWrites = Array(writes.values)
        for task in pendingWrites { await task.value }
    }

    private func fetch(_ host: MoldHost) async {
        let client = hosts.backend(for: host)
        let epoch = cacheEpoch
        var changed = false
        defer { if changed && epoch == cacheEpoch { saveSnapshot(host.id) } }
        do {
            let gallery = try await client.gallery(etag: etags[host.id])
            guard epoch == cacheEpoch else { return }
            switch gallery {
            case let .fresh(prints, etag):
                live[host.id] = prints
                etags[host.id] = etag
                changed = true
            case .notModified:
                break
            }
            if hosts.capabilities[host.id]?.trashEnabled == true {
                let trash = try await client.trashedPrints(etag: trashEtags[host.id])
                guard epoch == cacheEpoch else { return }
                switch trash {
                case let .fresh(prints, etag):
                    trashed[host.id] = prints
                    trashEtags[host.id] = etag
                    changed = true
                case .notModified:
                    break
                }
            }
            if hosts.capabilities[host.id]?.canOrganize == true {
                let visibilityGeneration = collectionVisibility.generation
                let fresh = try await client.collections()
                guard epoch == cacheEpoch, hosts.host(host.id) == host,
                      collectionVisibility.generation == visibilityGeneration else { return }
                collectionInventoryAvailable.insert(host.id)
                if fresh != collections[host.id] { collections[host.id] = fresh; changed = true }
            }
        } catch is CancellationError {
            // Foreground navigation and coalesced gallery events can end a
            // refresh after prints loaded. Cancellation is not a host failure.
        } catch {
            collectionInventoryAvailable.remove(host.id)
            hosts.report(host, doing: String(localized: "list its prints"), error)
        }
    }

    private func forget(_ id: MoldHost.ID) {
        collectionInventoryAvailable.remove(id)
        live[id] = nil; trashed[id] = nil; collections[id] = nil
        etags[id] = nil; trashEtags[id] = nil
        snapshots.remove(id)
        forgot?(id)
    }

    /// Written off the main thread; a listing is small, but not free.
    private func saveSnapshot(_ id: MoldHost.ID) {
        guard let prints = live[id] else { return }
        let snapshot = LibrarySnapshot(prints: prints, trashed: trashed[id], collections: collections[id],
                                       etag: etags[id], trashEtag: trashEtags[id])
        let snapshots = snapshots
        let previous = writes[id]
        writes[id] = Task.detached(priority: .utility) {
            await previous?.value
            snapshots.save(snapshot, for: id)
        }
    }

    /// For an edit shown before the machine confirms it.
    func updateLive(_ id: MoldHost.ID, _ change: ([GalleryPrint]) -> [GalleryPrint]) {
        if let prints = live[id] { live[id] = change(prints) }
        if let prints = trashed[id] { trashed[id] = change(prints) }
    }

    func acceptCollection(_ collection: Collection, on hostID: MoldHost.ID) {
        collections[hostID] = collections[hostID]?.map { $0.id == collection.id ? collection : $0 }
        rebuild()
    }

    func rebuildNow() { rebuild() }

    private func rebuild() {
        pool = merged(live)
        trashPool = merged(trashed)
        shelves = CollectionShelf.merge(collections).map {
            $0.overriding(hidden: collectionVisibility.desiredHidden(slug: $0.slug, fallback: $0.hidden))
        }
        knownTags = Array(Set(pool.flatMap(\.tags)))
            .sorted { $0.localizedStandardCompare($1) == .orderedAscending }
        revision += 1
    }

    /// In machine-list order, so the merge is stable across reloads.
    private func merged(_ perHost: [MoldHost.ID: [GalleryPrint]]) -> [LibraryEntry] {
        let entries = hosts.hosts.flatMap { host in
            (perHost[host.id] ?? []).map { LibraryEntry(host: host, print: $0) }
        }
        return LibraryMerge.merge(entries, localHost: nil)
    }

    /// A gallery event on one machine re-reads that machine, coalesced: a
    /// batch of four prints is one reload, not four.
    private func heard(_ event: MoldEvent, from id: MoldHost.ID) {
        switch event {
        case .gallery, .resyncRequired:
            pending[id]?.cancel()
            pending[id] = Task { [weak self] in
                try? await Task.sleep(for: .milliseconds(300))
                guard !Task.isCancelled else { return }
                await self?.reload(id)
            }
        default:
            break
        }
    }
}
