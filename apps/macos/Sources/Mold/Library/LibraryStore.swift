import Foundation
import MoldClient

/// Every machine's prints, in one timeline.
///
/// Each machine is followed by its own `GalleryReplica` (off the main actor):
/// by cursor through `GET /api/gallery/changes` where the machine advertises
/// `gallery.changes_feed`, otherwise through the conditional full listing.
/// Either way a refresh hands back a `ReplicaDelta` -- or nothing, in which
/// case the merged timeline is not touched at all -- and the delta is applied
/// to `perHost` / `trashPerHost` and patched into the timelines
/// (`LibraryTimeline`) without a re-merge wherever a merge could not differ.
@MainActor
@Observable
final class LibraryStore {
    let sourceMediaCatalog = SourceMediaCatalog()
    @ObservationIgnored let sourceAccessIndex = SourceMediaAccessIndex()
    var sourceAccessRevision = 0
    var openSourceDetails: [UUID: SourceMediaAuthority] = [:]
    var unreadMedia: LibraryUnreadLedger
    let readDefaults: UserDefaults
    var unreadCount: Int { unreadMedia.count }
    @ObservationIgnored var unreadCountChanged: ((Int) -> Void)?

    func finishVisit(_ ids: Set<PrintID>) {
        guard !ids.isEmpty else { return }
        unreadMedia.markSeen(ids)
        unreadMedia.save(to: readDefaults)
        unreadCountChanged?(unreadCount)
    }

    func markViewed(_ id: PrintID) {
        if let tile = tile(containing: id) { unreadMedia.view(tile) }
        else { unreadMedia.view(id) }
        unreadMedia.save(to: readDefaults)
        unreadCountChanged?(unreadCount)
    }



    /// The one object that knows which machines exist and how to reach them.
    let hosts: HostStore

    /// `internal(set)`, like `trashed`: `rebuild()` lives in `+Rows`.
    internal(set) var items: [LibraryEntry] = [] { didSet { sourceAccessRevision &+= 1 } }
    /// A count, not a flag: two overlapping refreshes (a manual ⌘R while an
    /// automatic one is still in flight, say) used to have the first one's
    /// `defer` turn this off while the second was still running.
    private var loads = 0
    var isLoading: Bool { loads > 0 }

    /// Set from here and from `+Mutations`'s `refreshTrash()`; `private(set)`
    /// does not cross that file boundary.
    internal(set) var trashed: [LibraryEntry] = [] { didSet { sourceAccessRevision &+= 1 } }

    /// One machine's rows as shown, optimistic edits included. A write from
    /// outside the store's own delta paths (a test, a bulk reconcile) marks
    /// the timeline for a full rebuild at its next settle.
    var perHost: [MoldHost.ID: [LibraryEntry]] = [:] {
        didSet { if !writingRows { pendingLive.full = true } }
    }
    var trashPerHost: [MoldHost.ID: [LibraryEntry]] = [:] {
        didSet { if !writingRows { pendingTrash.full = true } }
    }

    /// Each machine's replica, with the exact machine record (route, key) it
    /// reads through. See `LibraryStore+Replica`.
    @ObservationIgnored var replicas: [MoldHost.ID: (identity: String, replica: GalleryReplica)] = [:]
    @ObservationIgnored var prunedReplicaFiles = false
    @ObservationIgnored let replicaStore: GalleryReplicaStore?
    /// What a machine had when a repairing re-list ASKED, so its reset can
    /// keep a print that landed while the answer was in flight (`RelistMerge`).
    @ObservationIgnored var relistAsked: [MoldHost.ID: Set<String>] = [:]
    /// The merged timelines and their copy -> tile indexes.
    @ObservationIgnored var liveTimeline = LibraryTimeline(order: .newestFirst)
    @ObservationIgnored var trashTimeline = LibraryTimeline(order: .recentlyTrashedFirst)
    /// Rows changed since the timelines were last settled.
    @ObservationIgnored var pendingLive = PendingRows()
    @ObservationIgnored var pendingTrash = PendingRows()
    @ObservationIgnored var writingRows = false
    @ObservationIgnored var deferredSettles = 0

    /// Collections as each MACHINE holds them. A collection's id is that
    /// machine's, so these are never merged in storage -- only when they are
    /// read, by `CollectionShelf.merge`.
    var collectionsPerHost: [MoldHost.ID: [Collection]] = [:] { didSet { sourceAccessRevision &+= 1 } }
    var collectionInventoryAvailable: Set<MoldHost.ID> = []
    let collectionVisibility = CollectionVisibilityLedger.load(from: AppStorageSuite.defaults, key: "library.collectionVisibility")
    /// One collection-visibility writer at a time. See `VisibilityWriterLock`.
    @ObservationIgnored let visibilityWriters = VisibilityWriterLock()

    /// The tag index, and renaming or deleting a tag everywhere. See
    /// `LibraryTags`.
    let tags = LibraryTags()

    /// The Edit menu's Undo, for the edits this store makes. Its manager is
    /// the window's, handed over by the pane -- see `MoldUndo`.
    let undo = MoldUndo()

    /// Organization edits on their way to the machines, and the one task per
    /// machine walking them there. See `LibraryMutations`.
    let mutations = LibraryMutations()
    /// What the machines' live frames do to the rows on screen, and the two
    /// pieces of memory that decision needs. See `GalleryLive`.
    let live = GalleryLive()

    /// A Library import to the embedded host is a batch, not a Finder export.
    var bulkActivities: [UUID: String] = [:]
    var bulkProgress: String?
    var bulkResult: String?
    var bulkResultHasIssues = false
    var activityDisclosure = ActivityDisclosure()
    var syncActivityID = UUID()
    var bulkActivityID = UUID()
    var bulkRunning = false { didSet { if bulkRunning && !oldValue { bulkActivityID = UUID() } } }
    var bulkEmptying = false
    var isBulkBusy: Bool { bulkRunning }
    var bulkStopRequested = false
    var bulkTargets: Set<PrintID> = []

    var localSaveRunning = false { didSet { if localSaveRunning && !oldValue { syncActivityID = UUID() } } }
    var localSaveSuppressImports = false
    var syncDestinationHost: MoldHost.ID?
    var localSaveProgress: String?
    /// When `localSaveProgress` last changed, for its ~10 Hz throttle.
    @ObservationIgnored var localSaveProgressPublished: ContinuousClock.Instant?
    /// Library Sync's copy records, cursors and retry set.
    @ObservationIgnored let syncRecords: LibrarySyncRecords
    var localSaveTask: Task<Void, Never>?
    var localSaveStopRequested = false
    var localSaveReport = ""
    var localSaveFailures: [String] = []
    var localSaveNotices: [String] = []
    var localSaveAlertPresented = false
    let syncSession: LibrarySyncSession
    var localSaveIssueKeys: [String: String] = [:]

    /// How many times the rows have changed. See `LibraryRevision`.
    let rows = LibraryRevision()

    /// `replicaStore` nil keeps replicas in memory only -- every test's
    /// default, so no test writes the real Application Support.
    init(hosts: HostStore, syncSession: LibrarySyncSession = LibrarySyncSession(),
         readDefaults: UserDefaults = AppStorageSuite.defaults, replicaStore: GalleryReplicaStore? = nil,
         syncRecords: LibrarySyncRecords = LibrarySyncRecords()) {
        self.hosts = hosts
        self.syncRecords = syncRecords
        self.syncSession = syncSession
        self.readDefaults = readDefaults
        self.replicaStore = replicaStore
        unreadMedia = LibraryUnreadLedger.load(from: readDefaults)
        unreadMedia.retainHosts(Set(hosts.hosts.map(\.id)))
        unreadMedia.save(to: readDefaults)
        // Listening starts with the store, not with a pane. The Library used
        // to register on appearing, so a print made while Generate was showing
        // reached nobody and the timeline only caught up on the next ⌘R.
        // RootView starts the first listing independently of the destination,
        // so sidebar shelves and counts do not wait for Library to open.
        hosts.onEvent { [weak self] host, event in
            guard let self else { return }
            sourceMediaChanged(event, host: host)
            live.apply(event, from: host, in: self)
        }
    }

    /// Prints from every host, newest first. A machine whose replica reports
    /// no change leaves the timeline exactly as it was.
    func refresh() async {
        loads += 1
        defer { loads -= 1 }
        // A machine that was removed must not keep contributing prints to a
        // merged timeline nobody can attribute them from.
        prune(to: hosts.hosts)
        await pullAll(.live, doing: "list its prints")
    }

    /// Independent sidebar data starts together: a slow image index or trash
    /// listing must not hold collections hostage on another destination.
    func reload() async {
        guard !Task.isCancelled else { return }
        async let prints: Void = refresh()
        async let trash: Void = refreshTrash()
        async let organization: Void = refreshOrganization()
        _ = await (prints, trash, organization)
    }
}
