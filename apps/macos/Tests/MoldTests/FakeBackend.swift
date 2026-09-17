import Foundation
import MoldClient
@testable import Mold

/// A machine that records what it was asked and answers with what a test
/// planted. A route with nothing planted THROWS, so a store that reaches for
/// an unexpected one fails the test rather than quietly getting a default.
///
/// A class, not a struct of closures: 38 requirements as stored closures would
/// mean every test supplying all 38. `@unchecked Sendable` with
/// `nonisolated(unsafe)` state: the three stream requirements answer
/// synchronously, which a `@MainActor` witness cannot satisfy, and this is
/// only ever touched from the main actor by a `@MainActor` test.
/// Lets a store's own tasks run. A watcher is an unstructured `Task`, so
/// nothing about it has happened yet when the call that started it returns --
/// a test asserts on what it DID, not on the instant it was made.
@MainActor
func settle(until condition: () -> Bool) async {
    for step in 0 ..< 200 {
        if condition() { return }
        if step < 100 { await Task.yield() } else { try? await Task.sleep(for: .milliseconds(5)) }
    }
}

final class FakeBackend: MoldBackend, @unchecked Sendable {
    let host: MoldHost
    nonisolated(unsafe) private(set) var calls: [String] = []
    /// Route names that answer with `.unreachable` however they are planted.
    nonisolated(unsafe) var refuses: Set<String> = []
    nonisolated(unsafe) var prints: [GalleryPrint] = []
    nonisolated(unsafe) var trashedRows: [GalleryPrint] = []
    nonisolated(unsafe) var tagRows: [TagCount] = []
    nonisolated(unsafe) var collectionRows: [Collection] = []
    /// `nil` means nothing was planted, so `queue()` behaves like every other
    /// unplanted route and throws rather than answering with an empty list.
    nonisolated(unsafe) var queueListing: QueueListing?
    nonisolated(unsafe) var serverStatus: ServerStatus?
    nonisolated(unsafe) var capabilityBlock: Capabilities?
    nonisolated(unsafe) var exportBlock: ExportOptions?
    nonisolated(unsafe) var downloadTicket: DownloadTicket?
    /// The live `/api/events` stream, so a test can hand the store a frame
    /// and watch what it does with it. Held open: a stream that finishes
    /// sends the watcher round its reconnect loop, which is a second
    /// `events` call and a wait a test would have to sleep through.
    nonisolated(unsafe) var eventStream: AsyncThrowingStream<MoldEvent, Error>.Continuation?
    nonisolated(unsafe) var downloadStream: AsyncThrowingStream<DownloadEvent, Error>.Continuation?
    /// Set when the download stream's consumer went away.
    nonisolated(unsafe) var downloadStreamEnded = false

    /// Hands the open event stream one frame.
    func emit(_ event: MoldEvent) { eventStream?.yield(event) }

    func callCount(_ route: String) -> Int { calls.filter { $0 == route }.count }

    init(host: MoldHost) { self.host = host }

    private func record(_ route: String) throws {
        calls.append(route)
        if refuses.contains(route) { throw MoldClientError.unreachable("planted") }
    }

    private func notPlanted() -> Error { MoldClientError.unreachable("not planted") }

    // MARK: - Status

    func status() async throws -> ServerStatus {
        try record("status")
        guard let serverStatus else { throw notPlanted() }
        return serverStatus
    }
    func capabilities() async throws -> Capabilities {
        try record("capabilities")
        guard let capabilityBlock else { throw notPlanted() }
        return capabilityBlock
    }
    func models() async throws -> [Model] { try record("models"); return [] }

    // MARK: - Generation

    func placementPreview(_ request: GenerateRequest, copies: Int) async throws -> PlacementPreview {
        try record("placementPreview"); throw notPlanted()
    }
    func submit(_ admission: BatchAdmission) async throws -> BatchStatus {
        try record("submit"); throw notPlanted()
    }
    func batchStatus(id: String) async throws -> BatchStatus {
        try record("batchStatus"); throw notPlanted()
    }
    func batchStatus(clientBatchId: String) async throws -> BatchStatus {
        try record("batchStatusByClientId"); throw notPlanted()
    }
    func jobPreview(jobId: String) async throws -> JobProgress? { try record("jobPreview"); return nil }
    func cancelBatch(id: String) async throws { try record("cancelBatch") }

    // MARK: - Queue

    func queue() async throws -> QueueListing {
        try record("queue")
        guard let queueListing else { throw notPlanted() }
        return queueListing
    }
    func cancelJob(id: String) async throws { try record("cancelJob") }
    func pauseJob(id: String) async throws { try record("pauseJob") }
    func resumeJob(id: String) async throws { try record("resumeJob") }
    func retryJob(_ entry: QueueEntry, instanceId: String) async throws { try record("retryJob") }

    // MARK: - Downloads

    func startDownload(_ request: DownloadRequest) async throws -> DownloadTicket {
        try record("startDownload")
        guard let downloadTicket else { throw notPlanted() }
        return downloadTicket
    }
    func cancelDownload(id: String) async throws { try record("cancelDownload") }

    // MARK: - Gallery

    func gallery(etag: String?) async throws -> Fetched<[GalleryPrint]> {
        try record("gallery")
        return .fresh(prints, etag: "fake-etag")
    }
    func trashedPrints(etag: String?) async throws -> Fetched<[GalleryPrint]> {
        try record("trashedPrints")
        return .fresh(trashedRows, etag: "fake-etag")
    }
    func patch(_ filename: String, with patch: GalleryPatch) async throws { try record("patch") }
    func mutate(_ mutation: GalleryBulkMutation) async throws { try record("mutate") }
    func trash(_ filenames: [String]) async throws { try record("trash") }
    func restoreFromTrash(_ filenames: [String]) async throws { try record("restoreFromTrash") }
    func deleteForever(_ filenames: [String]) async throws { try record("deleteForever") }
    @discardableResult
    func importPrint(_ item: GalleryImport, as filename: String) async throws -> String {
        try record("importPrint")
        return filename
    }
    func media(_ filename: String, trashed: Bool) async throws -> Data {
        try record("media"); throw notPlanted()
    }
    func exportOptions() async throws -> ExportOptions {
        try record("exportOptions")
        guard let exportBlock else { throw notPlanted() }
        return exportBlock
    }
    func export(_ filename: String, format: String) async throws -> Data {
        try record("export"); throw notPlanted()
    }
    /// Not throwing, per the protocol -- so this cannot honor `refuses`, and
    /// always answers with something plausible instead of a planted value.
    func playableURL(for filename: String) async -> URL {
        calls.append("playableURL")
        return host.baseURL.appendingPathComponent(filename)
    }

    // MARK: - Organization

    func collections() async throws -> [Collection] { try record("collections"); return collectionRows }
    func createCollection(name: String, description: String?) async throws -> Collection {
        try record("createCollection"); throw notPlanted()
    }
    func updateCollection(id: String, change: CollectionChange) async throws -> Collection {
        try record("updateCollection"); throw notPlanted()
    }
    func deleteCollection(id: String) async throws { try record("deleteCollection") }
    func tags() async throws -> [TagCount] { try record("tags"); return tagRows }
    @discardableResult
    func renameTag(_ name: String, to newName: String) async throws -> TagCount {
        try record("renameTag")
        return TagCount(name: newName, count: 0)
    }
    func deleteTag(_ name: String) async throws { try record("deleteTag") }
    func emptyTrash() async throws { try record("emptyTrash") }

    // MARK: - Streams

    func events() -> AsyncThrowingStream<MoldEvent, Error> {
        calls.append("events")
        return AsyncThrowingStream { self.eventStream = $0 }
    }
    func batchEvents(id: String) -> AsyncThrowingStream<BatchStatus, Error> {
        calls.append("batchEvents")
        return AsyncThrowingStream { $0.finish() }
    }
    func downloadEvents() -> AsyncThrowingStream<DownloadEvent, Error> {
        calls.append("downloadEvents")
        return AsyncThrowingStream {
            $0.onTermination = { _ in self.downloadStreamEnded = true }
            self.downloadStream = $0
        }
    }
}
