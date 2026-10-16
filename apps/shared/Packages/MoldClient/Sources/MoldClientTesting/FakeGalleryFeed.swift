import Foundation
import MoldClient
import Synchronization

/// A machine's gallery change log, scriptable from a test: the server half of
/// `GET /api/gallery/changes` with the same dirty-set semantics -- one entry
/// per name holding its latest sequence -- and the same reset rules.
///
/// Cursors are `"<epoch>:<seq>"`. A cursor from another epoch, or one past
/// the newest sequence, answers `reset: true` from sequence zero, exactly as
/// the server answers a cursor it cannot honour. `listing()` and
/// `trashListing()` are the same state as the full listings, for comparing
/// the two paths.
public final class FakeGalleryFeed: Sendable {
    struct State {
        var instanceID: String
        var epoch = 1
        var seq = 0
        var rows: [String: GalleryPrint] = [:]
        /// Name -> latest sequence. Removed names stay as tombstones.
        var dirty: [String: Int] = [:]
        var notReady = false
        var pages: [String?] = []
    }

    private let state: Mutex<State>

    public init(instanceID: String = "instance-1", rows: [GalleryPrint] = []) {
        state = Mutex(State(instanceID: instanceID))
        for row in rows { upsert(row) }
    }

    // MARK: Scripting the machine

    public func upsert(_ print: GalleryPrint) {
        state.withLock { state in
            state.seq += 1
            state.rows[print.filename] = print
            state.dirty[print.filename] = state.seq
        }
    }

    public func remove(_ filename: String) {
        state.withLock { state in
            state.seq += 1
            state.rows[filename] = nil
            state.dirty[filename] = state.seq
        }
    }

    /// A restored database or corruption recovery: every cursor is void.
    public func rotateEpoch() { state.withLock { $0.epoch += 1 } }

    /// A different server answering at the same address.
    public func reinstance(_ id: String) { state.withLock { $0.instanceID = id } }

    /// `503 GALLERY_FEED_NOT_READY` while true.
    public func setNotReady(_ notReady: Bool) { state.withLock { $0.notReady = notReady } }

    /// The `since` of every page asked, in order.
    public var requestedCursors: [String?] { state.withLock { $0.pages } }

    public func listing() -> [GalleryPrint] {
        state.withLock { $0.rows.values.filter { $0.trashedAt == nil }.sorted { $0.filename < $1.filename } }
    }

    public func trashListing() -> [GalleryPrint] {
        state.withLock { $0.rows.values.filter { $0.trashedAt != nil }.sorted { $0.filename < $1.filename } }
    }

    // MARK: Answering

    public func page(since cursor: String?, limit: Int) throws -> GalleryChangesPage {
        try state.withLock { state in
            state.pages.append(cursor)
            if state.notReady {
                throw MoldClientError.http(status: 503, code: "GALLERY_FEED_NOT_READY", message: "not ready")
            }
            var after = 0
            var reset = false
            if let cursor {
                let parts = cursor.split(separator: ":").compactMap { Int($0) }
                if parts.count == 2, parts[0] == state.epoch, parts[1] <= state.seq {
                    after = parts[1]
                } else {
                    reset = true
                }
            }
            let pending = state.dirty.filter { $0.value > after }.sorted { $0.value < $1.value }
            let slice = pending.prefix(max(limit, 1))
            let more = pending.count > slice.count
            let last = more ? slice.last!.value : state.seq
            let changes = slice.map { name, _ in
                state.rows[name].map { GalleryChange(filename: name, image: $0) }
                    ?? GalleryChange(filename: name, removed: true)
            }
            return GalleryChangesPage(instanceId: state.instanceID, cursor: "\(state.epoch):\(last)",
                                      more: more, reset: reset, changes: changes)
        }
    }

    /// Answers `backend`'s feed, listing and trash routes from this log.
    public func serve(_ backend: FakeBackend) {
        backend.stub("galleryChanges(since:limit:)") { [self] args in
            try page(since: args[0] as? String, limit: args[1] as? Int ?? 2_000)
        }
        backend.stub("gallery(etag:)") { [self] _ in Fetched.fresh(listing(), etag: nil) }
        backend.stub("trashedPrints(etag:)") { [self] _ in Fetched.fresh(trashListing(), etag: nil) }
    }
}
