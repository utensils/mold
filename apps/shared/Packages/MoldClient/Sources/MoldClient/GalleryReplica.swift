import Foundation
import Synchronization

/// One machine's gallery as this device last read it, kept current off the
/// main actor.
///
/// A machine with `gallery.changes_feed` is followed by cursor
/// (`GET /api/gallery/changes`), so a refresh costs the rows that changed, not
/// the whole index. An older machine -- or one whose feed is not ready -- is
/// read through the conditional full listings and diffed by row, into the SAME
/// `ReplicaDelta`: a caller never branches on which kind of machine it has.
///
/// `pull()` is single-flight: callers that arrive while a pull runs share ONE
/// follow-up pull (a tickle mid-read means "read once more after this one",
/// never N more), which every one of them awaits. Deltas reach the optional
/// `sink` strictly in revision order, before the pull that produced them
/// returns, so a consumer applying them never sees an older delta after a
/// newer one.
public actor GalleryReplica {
    public typealias Sink = @Sendable (ReplicaDelta) async -> Void

    public nonisolated let hostID: UUID
    let store: GalleryReplicaStore?
    let saveDelay: Duration
    let pageLimit: Int

    /// Every row this machine holds, live and trashed, by filename.
    public internal(set) var rows: [String: GalleryPrint] = [:]
    /// The feed position the next pull resumes from.
    public internal(set) var cursor: String?
    /// The server instance `cursor` belongs to. A different one is a reset.
    public internal(set) var instanceID: String?
    /// How many pulls actually ran (joined callers do not count).
    public internal(set) var pullCount = 0
    /// Callers waiting on the queued follow-up. What a test settles on.
    var followers = 0

    var liveETag: String?
    var trashETag: String?
    /// Names the legacy trash listing named, whatever their `trashedAt`.
    var listedInTrash: Set<String> = []
    var revision = 0
    /// What this process has already been handed whole. The first delta for
    /// a partition is always a reset, so a consumer starts from the replica's
    /// state rather than from whatever it happened to hold before.
    var deliveredScopes: ReplicaDelta.Scope = []
    var loaded = false
    var sink: Sink?
    var saveTask: Task<Void, Never>?
    private nonisolated let invalidations = Mutex<ReplicaDelta.Scope>([])
    private nonisolated let forgotten = Mutex<Set<String>>([])
    private nonisolated let redeliveries = Mutex(false)
    /// Per listing partition: how many reads have started, and the newest one
    /// whose answer was applied. An answer older than the applied one is
    /// dropped, so two lanes (or a feed commit) can never land out of order.
    var listingsStarted: [Int: Int] = [:]
    var listingsApplied: [Int: Int] = [:]
    /// Removed for good: nothing may write the file again.
    var discarded = false
    /// Which names each delivered revision touched, oldest first, bounded by
    /// `changeLogLimit` names. What a mirror reads to plan only what changed
    /// since its own cursor (`changedNames(since:)`).
    var changeLog: [(revision: Int, names: [String])] = []
    var changeLogNames = 0
    /// The oldest revision the log can still answer from.
    var changeLogFloor = 0
    let changeLogLimit: Int
    /// This replica object, as opposed to another one for the same machine
    /// (a relaunch, a credential or instance change). Revisions restart with
    /// each, so a cursor naming another epoch can only mean "start over".
    public nonisolated let epoch = UUID()

    /// One flight per lane: the feed, or one legacy listing scope. Two
    /// listings are separate lanes so a slow trash listing never holds the
    /// timeline's read behind it.
    private struct Flight {
        var inFlight: Task<ReplicaDelta?, Error>?
        var queued: Task<ReplicaDelta?, Error>?
        var queuedPull: Pull?
        var queuedBackend: (any MoldGalleryBackend)?
    }
    private var flights: [Int: Flight] = [:]

    public init(hostID: UUID, store: GalleryReplicaStore? = nil,
                saveDelay: Duration = .seconds(5), pageLimit: Int = 2_000,
                changeLogLimit: Int = 100_000, sink: Sink? = nil) {
        self.hostID = hostID
        self.changeLogLimit = changeLogLimit
        self.store = store
        self.saveDelay = saveDelay
        self.pageLimit = pageLimit
        self.sink = sink
    }

    public func setSink(_ sink: Sink?) { self.sink = sink }

    /// Forgets the listing validators, so the next legacy read is a full one.
    /// Synchronous on purpose: a main-actor caller that has just made its
    /// rows diverge must not have to hop to say so.
    public nonisolated func invalidateValidators(_ scope: ReplicaDelta.Scope = .all) {
        invalidations.withLock { $0.formUnion(scope) }
    }

    /// The consumer changed these names on its own (a live frame spliced into
    /// what it shows). The next pull re-reads its listings in full and emits
    /// each of them as it is then -- present or gone -- so a splice the
    /// machine does not agree with can never outlive the next refresh.
    public nonisolated func forget(_ names: some Sequence<String>) {
        forgotten.withLock { $0.formUnion(names) }
        invalidateValidators()
    }

    /// The consumer dropped a delta (read through a route it has since
    /// replaced): hand it the whole state again on the next pull.
    public nonisolated func redeliver() {
        redeliveries.withLock { $0 = true }
    }

    /// Reads what changed since the last pull. `nil` when nothing did.
    ///
    /// Cancelling the caller that STARTED a pull cancels it before it changes
    /// anything; a follow-up shared by several callers always runs.
    @discardableResult
    public func pull(_ request: Pull = .feed, using backend: any MoldGalleryBackend) async throws -> ReplicaDelta? {
        let lane = request.useFeed ? 0 : request.scope.rawValue
        var flight = flights[lane, default: Flight()]
        if let queued = flight.queued {
            flight.queuedPull = flight.queuedPull?.merged(with: request) ?? request
            flight.queuedBackend = backend
            flights[lane] = flight
            followers += 1
            defer { followers -= 1 }
            return try await queued.value
        }
        if let inFlight = flight.inFlight {
            flight.queuedPull = request
            flight.queuedBackend = backend
            let follow = Task {
                _ = try? await inFlight.value
                return try await self.runQueued(lane)
            }
            flight.queued = follow
            flights[lane] = flight
            followers += 1
            defer { followers -= 1 }
            return try await follow.value
        }
        return try await run(request, backend, lane)
    }

    private func runQueued(_ lane: Int) async throws -> ReplicaDelta? {
        guard var flight = flights[lane], let backend = flight.queuedBackend else { return nil }
        let request = flight.queuedPull ?? .feed
        flight.queuedPull = nil
        flight.queuedBackend = nil
        flight.queued = nil
        flights[lane] = flight
        return try await run(request, backend, lane)
    }

    private func run(_ request: Pull, _ backend: any MoldGalleryBackend, _ lane: Int) async throws -> ReplicaDelta? {
        let task = Task { try await self.execute(request, using: backend) }
        flights[lane, default: Flight()].inFlight = task
        defer { if flights[lane]?.inFlight == task { flights[lane]?.inFlight = nil } }
        return try await withTaskCancellationHandler {
            try await task.value
        } onCancel: {
            task.cancel()
        }
    }

    private func execute(_ request: Pull, using backend: any MoldGalleryBackend) async throws -> ReplicaDelta? {
        ensureLoaded()
        pullCount += 1
        let forced = invalidations.withLock { scope in
            defer { scope = [] }
            return scope
        }
        if forced.contains(.live) { liveETag = nil }
        if forced.contains(.trash) { trashETag = nil }
        let spliced = forgotten.withLock { names in
            defer { names = [] }
            return names
        }
        if redeliveries.withLock({ flag in defer { flag = false }; return flag }) { deliveredScopes = [] }
        var outcome: Outcome
        do {
            if request.useFeed {
                do {
                    outcome = try await followFeed(using: backend)
                } catch let error where Self.fallsBackToListing(error) {
                    outcome = try await relist(.all, full: request.full, using: backend)
                }
            } else {
                outcome = try await relist(request.scope, full: request.full, using: backend)
            }
        } catch {
            // Still owed: the next pull re-reads and emits them.
            if !spliced.isEmpty { forget(spliced) }
            throw error
        }
        outcome.forced = spliced
        guard let delta = delta(for: outcome, full: request.full, origin: request.origin) else { return nil }
        if let sink { await sink(delta) }
        return delta
    }

    /// Which partition a row is in: the trash listing's word, else its fields.
    func isTrash(_ name: String) -> Bool {
        listedInTrash.contains(name) || rows[name]?.trashedAt != nil
    }

    /// A machine without the route, or whose log is still being built.
    static func fallsBackToListing(_ error: any Error) -> Bool {
        guard let error = error as? MoldClientError, case let .http(status, code, _) = error else { return false }
        return status == 404 || status == 405 || (status == 503 && code == "GALLERY_FEED_NOT_READY")
    }
}
