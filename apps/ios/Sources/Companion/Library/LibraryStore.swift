import Foundation
import MoldClient

/// Every machine's prints, merged into one timeline (DESIGN.md §5.2).
///
/// Holds DATA only: which shelf a window shows, what it searched for and what
/// it selected are that window's own state. A print held on several machines
/// is one entry (`LibraryMerge`, the Mac's rule). Each machine is read through
/// one MoldClient `GalleryReplica` (`LibraryStore+Replica`): its change feed
/// where it has one, else its conditional listings, so an unchanged library
/// costs a feed page or a 304 and a change costs the rows that changed.
@Observable
final class LibraryStore {
    let sourceMediaCatalog = SourceMediaCatalog()
    @ObservationIgnored let sourceAccessIndex = SourceMediaAccessIndex()
    var sourceAccessRevision = 0
    var openSourceDetails: [UUID: SourceMediaAuthority] = [:]
    private(set) var unreadMedia: LibraryUnreadLedger
    var unreadCount: Int { unreadMedia.count }
    @ObservationIgnored private let readDefaults: UserDefaults
    @ObservationIgnored var unreadCountChanged: ((Int) -> Void)?

    func finishVisit(_ ids: Set<PrintID>) {
        guard !ids.isEmpty else { return }
        unreadMedia.markSeen(ids)
        unreadMedia.save(to: readDefaults)
        unreadCountChanged?(unreadCount)
    }

    func markViewed(_ id: PrintID) {
        if let entry = liveTimeline.tile(containing: id) { unreadMedia.view(entry) }
        else { unreadMedia.view(id) }
        unreadMedia.save(to: readDefaults)
        unreadCountChanged?(unreadCount)
    }
    var machineID: MoldHost.ID?
    var collectionInventoryAvailable: Set<MoldHost.ID> = []
    let collectionVisibility = CollectionVisibilityLedger.load(from: .standard, key: "library.collectionVisibility")
    var reconcilingCollectionVisibility = false
    var collectionInventory: [MoldHost.ID: [Collection]] { collections }

    var pool: [LibraryEntry] = [] { didSet { sourceAccessRevision &+= 1 } }
    var trashPool: [LibraryEntry] = [] { didSet { sourceAccessRevision &+= 1 } }
    private(set) var shelves: [CollectionShelf] = [] { didSet { sourceAccessRevision &+= 1 } }
    var knownTags: [String] = []
    /// Bumped whenever the pool, the trash or the shelves change -- and only
    /// then: a pull that changed nothing leaves it alone. `LibraryShowingCache`
    /// keys on it.
    var revision = 0
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
    /// The last collections write per machine: the next waits for it, so two
    /// quick listings can never land on disk in the wrong order.
    @ObservationIgnored var writes: [MoldHost.ID: Task<Void, Never>] = [:]
    /// Bumped by Clear Cache: a replica, restore or listing begun before it
    /// may not put anything back.
    @ObservationIgnored var cacheEpoch = 0

    @ObservationIgnored let hosts: HostStore
    /// Each machine's rows as shown, optimistic edits included.
    @ObservationIgnored var live: [MoldHost.ID: [GalleryPrint]] = [:]
    @ObservationIgnored var trashed: [MoldHost.ID: [GalleryPrint]] = [:]
    @ObservationIgnored var collections: [MoldHost.ID: [Collection]] = [:]
    /// Each machine's replica and the record identity it was made for.
    @ObservationIgnored var replicas: [MoldHost.ID: (identity: String, replica: GalleryReplica)] = [:]
    @ObservationIgnored var prunedReplicaFiles = false
    /// The merged timelines and their copy -> tile indexes, and what changed
    /// in the rows since they were last settled.
    @ObservationIgnored var liveTimeline = LibraryTimeline(order: .newestFirst)
    @ObservationIgnored var trashTimeline = LibraryTimeline(order: .recentlyTrashedFirst)
    @ObservationIgnored var pendingLive = PendingRows()
    @ObservationIgnored var pendingTrash = PendingRows()
    /// Machines whose collections must be read again (an organization delta,
    /// a `collectionsChanged` frame, missed events), not on every pull.
    @ObservationIgnored var collectionsStale: Set<MoldHost.ID> = []
    /// A coalescing pause in front of an event's pull, per machine.
    @ObservationIgnored var tickles: [MoldHost.ID: Task<Void, Never>] = [:]
    @ObservationIgnored let snapshots: LibrarySnapshots
    // Edits (`LibraryStore+Edits`).
    @ObservationIgnored var outbox = MutationOutbox()
    @ObservationIgnored var draining: Set<MoldHost.ID> = []
    @ObservationIgnored var lastFailure: [MoldHost.ID: Error] = [:]
    /// The change Undo would put back.
    var lastEdit: PrintEdit?
    /// The focused window's, so ⌘Z and shake-to-undo reach `undo()`.
    @ObservationIgnored weak var undoManager: UndoManager?

    init(hosts: HostStore, snapshots: LibrarySnapshots = .standard, readDefaults: UserDefaults = .standard) {
        self.hosts = hosts
        self.snapshots = snapshots
        self.readDefaults = readDefaults
        unreadMedia = LibraryUnreadLedger.load(from: readDefaults)
        unreadMedia.retainHosts(Set(hosts.hosts.map(\.id)))
        unreadMedia.save(to: readDefaults)
        hosts.listen { [weak self] id, event in self?.heard(event, from: id) }
    }

    /// The saved copy of every machine's library, shown before any machine is
    /// asked (and when none answers). Each replica decodes its own file off
    /// the main thread, once; after the first launch this is nothing.
    func restoreSaved() async {
        let epoch = cacheEpoch
        let ids = hosts.hosts.map(\.id).filter { collections[$0] == nil }
        let snapshots = snapshots
        let saved = await Task.detached(priority: .userInitiated) {
            ids.compactMap { id in
                snapshots.migrate(id)
                return snapshots.load(id).map { (id, $0) }
            }
        }.value
        guard epoch == cacheEpoch else { return }
        for (id, snapshot) in saved where collections[id] == nil { collections[id] = snapshot.collections }
        if !saved.isEmpty { refreshShelves() }
        let restoring = hosts.hosts.map { replica(for: $0) }
        for replica in restoring { await replica.restoreSaved() }
        settleTimelines()
    }

    /// Every machine that answers, at once; one settle at the end. This is
    /// the reconcile -- pull to refresh, a return to the foreground, a
    /// background wake -- after which frames may have been missed, so it
    /// reads collections too.
    func reload() async {
        isLoading = true
        defer { isLoading = false }
        // A machine that left the list leaves the grid. One that stopped
        // answering keeps its saved prints (`offlineHosts`).
        prune(to: hosts.hosts)
        let targets = hosts.upHosts
        collectionsStale.formUnion(targets.map(\.id))
        await withTaskGroup(of: Void.self) { group in
            for host in targets { group.addTask { await self.pull(host) } }
        }
        settleTimelines()
        await reconcileCollectionVisibility()
    }

    /// One machine, after something this app did there (or an event).
    /// `full` delivers its whole state even when unchanged: the repair after
    /// an optimistic edit the machine refused.
    func reload(_ id: MoldHost.ID, full: Bool = false) async {
        guard let host = hosts.host(id), hosts.isUp(host) else { return }
        await pull(host, full: full)
        settleTimelines()
        await reconcileCollectionVisibility()
    }

    /// Clear the device's saved listings, including their validators. Prints
    /// from machines still answering remain on screen until the next
    /// refresh; offline-only prints leave immediately because their saved
    /// copy is gone.
    func clearOfflineCache() async {
        cacheEpoch += 1
        for task in tickles.values { task.cancel() }
        tickles.removeAll()
        // Only the collections writes: flushing a replica about to be
        // discarded would write its whole file just to delete it.
        let pendingWrites = Array(writes.values)
        for task in pendingWrites { await task.value }
        writes.removeAll()
        let dropped = replicas.values.map(\.replica)
        replicas.removeAll()
        for replica in dropped { await replica.discard() }
        snapshots.purge()
        for host in hosts.hosts where !hosts.isUp(host) {
            live[host.id] = nil
            trashed[host.id] = nil
            collections[host.id] = nil
        }
        rebuildNow()
    }

    /// Await the writes already queued when this barrier starts, and write
    /// every replica's state now. Capturing the tasks keeps later listings
    /// from changing what completion means.
    func waitForSnapshotWrites() async {
        let pendingWrites = Array(writes.values)
        for task in pendingWrites { await task.value }
        for entry in replicas.values { await entry.replica.flush() }
    }

    /// Collections written off the main thread, when they change.
    func saveCollections(_ id: MoldHost.ID) {
        let snapshot = LibrarySnapshot(collections: collections[id])
        let snapshots = snapshots
        let previous = writes[id]
        writes[id] = Task.detached(priority: .utility) {
            await previous?.value
            snapshots.save(snapshot, for: id)
        }
    }

    /// For an edit shown before the machine confirms it. A change that keeps
    /// every row in place is settled as a patch; anything else re-merges.
    func updateLive(_ id: MoldHost.ID, _ change: ([GalleryPrint]) -> [GalleryPrint]) {
        if let prints = live[id] { live[id] = change(prints); note(prints, live[id] ?? [], on: id, trash: false) }
        if let prints = trashed[id] { trashed[id] = change(prints); note(prints, trashed[id] ?? [], on: id, trash: true) }
    }

    private func note(_ old: [GalleryPrint], _ new: [GalleryPrint], on id: MoldHost.ID, trash: Bool) {
        guard old.count == new.count, zip(old, new).allSatisfy({ $0.filename == $1.filename }) else {
            if trash { pendingTrash.full = true } else { pendingLive.full = true }
            return
        }
        let changed = zip(old, new).filter { $0 != $1 }.map { PrintID(host: id, filename: $1.filename) }
        if trash { pendingTrash.note(changed) } else { pendingLive.note(changed) }
    }

    func acceptCollection(_ collection: Collection, on hostID: MoldHost.ID) {
        collections[hostID] = collections[hostID]?.map { $0.id == collection.id ? collection : $0 }
        refreshShelves()
    }

    /// Everything from scratch. For what no delta describes (Clear Cache).
    func rebuildNow() {
        pendingLive.full = true
        pendingTrash.full = true
        settleTimelines()
        refreshShelves()
    }

    /// The shelves, from the collections and the visibility ledger. Only a
    /// real difference re-renders.
    func refreshShelves() {
        let fresh = CollectionShelf.merge(collections).map {
            $0.overriding(hidden: collectionVisibility.desiredHidden(slug: $0.slug, fallback: $0.hidden))
        }
        guard fresh != shelves else { return }
        shelves = fresh
        refreshUnreadMedia()
        revision += 1
    }

    func refreshUnreadMedia() {
        let previous = unreadMedia
        unreadMedia.retainHosts(Set(hosts.hosts.map(\.id)))
        let query = LibraryScope.all.resolve(LibraryQuery(), shelves: shelves, hiddenCollectionIDs: hiddenCollectionIDs)
        unreadMedia.observe(entries: pool, visible: query.apply(to: pool),
                            loadedHosts: Set(live.keys), presentHosts: Set(hosts.hosts.map(\.id)))
        if unreadMedia != previous { unreadMedia.save(to: readDefaults) }
        if unreadMedia.count != previous.count { unreadCountChanged?(unreadCount) }
    }

    /// A gallery event on one machine tickles its replica, coalesced: frames
    /// arriving within the pause share one pull, and a pull that starts
    /// later answers for them too (`pull` drops the pause).
    private func heard(_ event: MoldEvent, from id: MoldHost.ID) {
        switch event {
        case .gallery(.collectionsChanged):
            collectionsStale.insert(id)
            tickle(id)
        case .resyncRequired:
            sourceMediaChanged(event, host: id)
            collectionsStale.insert(id)
            tickle(id)
        case .gallery:
            sourceMediaChanged(event, host: id)
            tickle(id)
        default:
            break
        }
    }
}
