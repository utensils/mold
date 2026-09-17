import Foundation

/// The one seam between the UI and wherever generation actually happens.
///
/// `HTTPBackend` is the only conformance, and it covers two of the three
/// intended cases on its own: a remote `mold serve` over the network, and
/// mold's own Rust engine running in-process on loopback. `mold-server` speaks
/// ONE wire contract whether it is bound to a Tailscale address or to
/// `127.0.0.1`, so embedding the engine changes a URL, not this protocol. The
/// third is a fake for previews and tests.
///
/// It carries every route the app calls. It used to carry thirteen, and the
/// rest were reached by downcasting to `HTTPBackend` at the call site -- which
/// in `QueueStore` meant an optional chain where a failed cast was `nil`,
/// nothing threw, and a cancel that never left the machine reported success.
public protocol MoldBackend: Sendable {
    var host: MoldHost { get }

    // MARK: - Status

    func status() async throws -> ServerStatus
    func capabilities() async throws -> Capabilities
    func models() async throws -> [Model]

    // MARK: - Generation

    /// Read-only: reserves nothing, queues nothing.
    func placementPreview(_ request: GenerateRequest, copies: Int) async throws -> PlacementPreview
    /// Admits a batch. Idempotent on `clientBatchId`: re-sending the same one
    /// returns the work already held rather than starting it twice.
    func submit(_ admission: BatchAdmission) async throws -> BatchStatus
    func batchStatus(id: String) async throws -> BatchStatus
    /// Recovers a batch whose admission response was lost, by the id this
    /// client minted for it.
    func batchStatus(clientBatchId: String) async throws -> BatchStatus
    /// Step progress and the denoise preview for one running job.
    func jobPreview(jobId: String) async throws -> JobProgress?
    func cancelBatch(id: String) async throws

    // MARK: - Create

    /// Rewrites a prompt. A family that reads no prompt is answered with the
    /// family guide's own advice rather than an LLM rewrite.
    func expand(_ request: ExpandRequest) async throws -> ExpandResponse
    /// Subject-preserving alternatives. Separate from `expand` so a host too
    /// old to remix fails closed instead of silently expanding.
    func remix(_ request: RemixRequest) async throws -> RemixResponse
    /// Newest first. What somebody typed, not what was made.
    func history(limit: Int) async throws -> HistoryListing
    /// `nil` clears everything; otherwise trims to the most recent N. There
    /// is no per-row delete -- a `HistoryEntry` carries no id to name one.
    func clearHistory(keeping keep: Int?) async throws
    /// Every model's config surface, as `/api/config` reports it. An absent
    /// per-model key means "never configured"; a present key with a null
    /// value means the same thing.
    func config() async throws -> ConfigListing
    /// Sets one key. `models.<name>.<field>` CREATES the model's row.
    @discardableResult
    func setConfig(_ key: String, to value: ConfigScalar) async throws -> ConfigEntry
    /// Drops the DB row so the key falls back to file/env/default.
    @discardableResult
    func resetConfig(_ key: String) async throws -> ConfigEntry

    // MARK: - Queue

    func queue() async throws -> QueueListing
    func cancelJob(id: String) async throws
    func pauseJob(id: String) async throws
    func resumeJob(id: String) async throws
    /// The one route that moves a job BACKWARD, from held to accepted.
    func retryJob(_ entry: QueueEntry, instanceId: String) async throws

    // MARK: - Downloads

    func startDownload(_ request: DownloadRequest) async throws -> DownloadTicket
    func cancelDownload(id: String) async throws

    // MARK: - Machines

    func devices() async throws -> DeviceState
    /// Answers with the device's new state on both the 200 that reached it
    /// and the 202 that is draining or starting.
    @discardableResult
    func setDevice(_ id: String, enabled: Bool) async throws -> DeviceInfo
    /// 503 for about a second after the server boots -- not yet, not a fault.
    func resources() async throws -> ResourceSnapshot
    func resourceStream() -> AsyncThrowingStream<ResourceSnapshot, Error>
    /// What this machine can see on its own network. Per host, never merged.
    func peers() async throws -> [DiscoveryPeer]

    // MARK: - Gallery

    /// Pass the previous `etag` to let the host answer `.notModified`.
    func gallery(etag: String?) async throws -> Fetched<[GalleryPrint]>
    func trashedPrints(etag: String?) async throws -> Fetched<[GalleryPrint]>
    func patch(_ filename: String, with patch: GalleryPatch) async throws
    /// Replay-safe by `operationId`, so a retry cannot double-apply.
    func mutate(_ mutation: GalleryBulkMutation) async throws
    func trash(_ filenames: [String]) async throws
    func restoreFromTrash(_ filenames: [String]) async throws
    /// Permanent. There is no undo on the host side.
    func deleteForever(_ filenames: [String]) async throws
    @discardableResult
    func importPrint(_ item: GalleryImport, as filename: String) async throws -> String
    /// The stored bytes. A trashed print lives behind the trash view, exactly
    /// as the listing does.
    func media(_ filename: String, trashed: Bool) async throws -> Data
    func exportOptions() async throws -> ExportOptions
    /// Converts on the machine that holds the print, so the app needs no
    /// decoder for every container mold can write.
    func export(_ filename: String, format: String) async throws -> Data
    /// A URL a player can open directly, ticketed where the host needs it.
    /// Throws rather than falling back to an unticketed URL: on a keyed host
    /// a failed ticket means the player would 401, not play silently wrong.
    func playableURL(for filename: String) async throws -> URL

    // MARK: - Organization

    func collections() async throws -> [Collection]
    func createCollection(name: String, description: String?) async throws -> Collection
    /// Absent fields are untouched, so renaming does not clear a cover.
    func updateCollection(id: String, change: CollectionChange) async throws -> Collection
    /// Removes the shelf, never its prints.
    func deleteCollection(id: String) async throws
    func tags() async throws -> [TagCount]
    @discardableResult
    func renameTag(_ name: String, to newName: String) async throws -> TagCount
    func deleteTag(_ name: String) async throws
    /// Empties the trash now. Permanent.
    func emptyTrash() async throws

    // MARK: - Streams

    /// Everything this machine reports, as it happens.
    func events() -> AsyncThrowingStream<MoldEvent, Error>
    /// Whole-snapshot frames; safe to reconnect at any point.
    func batchEvents(id: String) -> AsyncThrowingStream<BatchStatus, Error>
    /// Progress for every download on this host.
    func downloadEvents() -> AsyncThrowingStream<DownloadEvent, Error>
}
