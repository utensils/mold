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
    /// Every route asked, in order. Read and written under `callsLock`: a
    /// store's unstructured task appends while a test reads `callCount`, and
    /// an unguarded array filtered mid-append is an index trap -- one took
    /// the whole app bundle down after five tests on 2026-09-16.
    var calls: [String] { callsLock.withLock { recorded } }
    nonisolated(unsafe) private var recorded: [String] = []
    private let callsLock = NSLock()
    /// Route names that answer with a refusal (a 409) however they are
    /// planted -- a REFUSAL, not an unreachable machine, so a store test can
    /// still pin the verb its report was keyed on.
    nonisolated(unsafe) var refuses: Set<String> = []
    /// Route names that throw a SPECIFIC error instead of what they would
    /// otherwise answer -- `refuses`' fixed 409 can't simulate a `503
    /// HISTORY_UNAVAILABLE` or `503 CONFIG_UNAVAILABLE`, which a store must
    /// tell apart from an ordinary refusal by CODE, not by status alone.
    nonisolated(unsafe) var plantedErrors: [String: Error] = [:]
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
    nonisolated(unsafe) var modelRows: [Model] = []
    /// Every admission `submit` was asked, in call order -- what a batch of
    /// four actually looked like on the wire.
    nonisolated(unsafe) var submittedAdmissions: [BatchAdmission] = []
    nonisolated(unsafe) var submitAnswer: BatchStatus?
    /// Planted per `id`, since a test drives `submit` then reads the same
    /// batch back through `batchStatus(id:)` once its events stream ends.
    nonisolated(unsafe) var batchStatusAnswers: [String: BatchStatus] = [:]
    /// `jobId` asked, in call order -- which child the preview poll followed.
    nonisolated(unsafe) var jobPreviewCalls: [String] = []
    nonisolated(unsafe) var mediaAnswer: Data?

    // MARK: - Machines

    nonisolated(unsafe) var deviceState: DeviceState?
    nonisolated(unsafe) var resourceSnapshot: ResourceSnapshot?
    nonisolated(unsafe) var peerRows: [DiscoveryPeer] = []
    /// Held open like `eventStream`, so a test can push a sample without
    /// the store's watcher going round its reconnect loop. Named apart from
    /// the `resourceStream()` witness below it answers -- a stored property
    /// and a method cannot share one name in Swift.
    nonisolated(unsafe) var resourceStreamContinuation: AsyncThrowingStream<ResourceSnapshot, Error>.Continuation?
    /// What `setDevice` was asked, in call order.
    nonisolated(unsafe) var patchedDevices: [(String, Bool)] = []
    /// Overrides the default enabled/disabled mutation, for a test that needs
    /// to see what a draining or starting answer looks like -- a real machine
    /// answers `setDevice` with its actual admin state, not the boolean it was
    /// asked for.
    nonisolated(unsafe) var setDeviceAnswer: DeviceInfo?
    /// Set when the resource stream's consumer went away, same reason as
    /// `downloadStreamEnded` below.
    nonisolated(unsafe) var resourceStreamEnded = false
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

    func callCount(_ route: String) -> Int { callsLock.withLock { recorded.filter { $0 == route }.count } }

    init(host: MoldHost) { self.host = host }

    private func record(_ route: String) throws {
        callsLock.withLock { recorded.append(route) }
        if let planted = plantedErrors[route] { throw planted }
        if refuses.contains(route) {
            throw MoldClientError.http(status: 409, code: nil, message: "Refused by the fake.")
        }
    }

    private func notPlanted() -> Error { MoldClientError.unreachable("not planted") }

    /// `DeviceInfo` has no public memberwise init -- like `GalleryPrint` and
    /// `QueueEntry` above, it is built the way the wire builds one, by
    /// round-tripping through JSON with the two fields a live toggle changes.
    private static func mutated(_ device: DeviceInfo, enabled: Bool) throws -> DeviceInfo {
        let data = try MoldJSON.encoder.encode(device)
        guard var object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return device
        }
        object["desired_enabled"] = enabled
        object["admin_state"] = enabled ? "enabled" : "disabled"
        let mutated = try JSONSerialization.data(withJSONObject: object)
        return try MoldJSON.decoder.decode(DeviceInfo.self, from: mutated)
    }

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
    func models() async throws -> [Model] { try record("models"); return modelRows }

    // MARK: - Generation

    /// `copies` asked, in call order -- what a batch of four previews as.
    nonisolated(unsafe) var placementCopiesRequested: [Int] = []

    func placementPreview(_ request: GenerateRequest, copies: Int) async throws -> PlacementPreview {
        try record("placementPreview")
        placementCopiesRequested.append(copies)
        throw notPlanted()
    }
    func submit(_ admission: BatchAdmission) async throws -> BatchStatus {
        try record("submit")
        submittedAdmissions.append(admission)
        guard let submitAnswer else { throw notPlanted() }
        return submitAnswer
    }
    func batchStatus(id: String) async throws -> BatchStatus {
        try record("batchStatus")
        guard let status = batchStatusAnswers[id] else { throw notPlanted() }
        return status
    }
    func batchStatus(clientBatchId: String) async throws -> BatchStatus {
        try record("batchStatusByClientId"); throw notPlanted()
    }
    func jobPreview(jobId: String) async throws -> JobProgress? {
        try record("jobPreview")
        jobPreviewCalls.append(jobId)
        return nil
    }
    func cancelBatch(id: String) async throws { try record("cancelBatch") }

    // MARK: - Create

    nonisolated(unsafe) var expandAnswer: ExpandResponse?
    nonisolated(unsafe) var remixAnswer: RemixResponse?
    /// `nil` throws as unplanted; `[]` is a real empty history, same rule as
    /// every other listing on this fake.
    nonisolated(unsafe) var historyRows: [HistoryEntry]?
    /// What `clearHistory` was asked, in call order -- `nil` is "clear
    /// everything", a number is the `keep` it trimmed to.
    nonisolated(unsafe) var historyCleared: [Int?] = []
    nonisolated(unsafe) var configListing: ConfigListing?
    /// What `setConfig` was asked, in call order.
    nonisolated(unsafe) var configWrites: [(String, ConfigScalar)] = []
    /// What `resetConfig` was asked, in call order.
    nonisolated(unsafe) var configResets: [String] = []

    func expand(_ request: ExpandRequest) async throws -> ExpandResponse {
        try record("expand")
        guard let expandAnswer else { throw notPlanted() }
        return expandAnswer
    }
    func remix(_ request: RemixRequest) async throws -> RemixResponse {
        try record("remix")
        guard let remixAnswer else { throw notPlanted() }
        return remixAnswer
    }
    func history(limit: Int) async throws -> HistoryListing {
        try record("history")
        guard let historyRows else { throw notPlanted() }
        return HistoryListing(entries: historyRows)
    }
    func clearHistory(keeping keep: Int?) async throws {
        try record("clearHistory")
        historyCleared.append(keep)
    }
    func config() async throws -> ConfigListing {
        try record("config")
        guard let configListing else { throw notPlanted() }
        return configListing
    }
    @discardableResult
    func setConfig(_ key: String, to value: ConfigScalar) async throws -> ConfigEntry {
        try record("setConfig")
        configWrites.append((key, value))
        return ConfigEntry(key: key, value: value, source: "db")
    }
    @discardableResult
    func resetConfig(_ key: String) async throws -> ConfigEntry {
        try record("resetConfig")
        configResets.append(key)
        return ConfigEntry(key: key, value: .null, source: "default")
    }

    /// Answered per MODEL -- a model absent from this dictionary is
    /// unplanted, the same "throw when nothing was planted" rule every other
    /// route follows. `[]` is a real empty answer: a family with no adapter
    /// support.
    nonisolated(unsafe) var loraRows: [String: [LoraInfo]] = [:]
    /// Per-model overrides, e.g. `400 UNKNOWN_MODEL` -- distinct from
    /// `refuses`'s fixed 409, the way `plantedErrors` already is for a route.
    nonisolated(unsafe) var loraErrors: [String: Error] = [:]
    /// Every model this was asked to list adapters for, in call order.
    nonisolated(unsafe) var loraModelsRequested: [String] = []

    func loras(compatibleWith model: String) async throws -> [LoraInfo] {
        try record("loras")
        loraModelsRequested.append(model)
        if let error = loraErrors[model] { throw error }
        guard let rows = loraRows[model] else { throw notPlanted() }
        return rows
    }

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

    // MARK: - Licences

    /// `nil` throws as unplanted, same rule as every other listing here.
    nonisolated(unsafe) var licenseRows: [ThirdPartyLicense]?
    /// Every `acceptLicenses` call, in order -- what a retry actually sent.
    nonisolated(unsafe) var acceptedLicenses: [[LicenseAcceptance]] = []

    func licenses() async throws -> [ThirdPartyLicense] {
        try record("licenses")
        guard let licenseRows else { throw notPlanted() }
        return licenseRows
    }
    @discardableResult
    func acceptLicenses(_ acceptances: [LicenseAcceptance]) async throws -> [ThirdPartyLicense] {
        try record("acceptLicenses")
        acceptedLicenses.append(acceptances)
        let accepted = Set(acceptances.map(\.id))
        licenseRows = (licenseRows ?? []).map { row in
            accepted.contains(row.id)
                ? ThirdPartyLicense(
                    id: row.id, name: row.name, url: row.url, canonical: row.canonical,
                    sha256: row.sha256, summary: row.summary, accepted: true,
                    requiredBy: row.requiredBy, requiredByStyles: row.requiredByStyles)
                : row
        }
        return licenseRows ?? []
    }

    // MARK: - Machines

    func devices() async throws -> DeviceState {
        try record("devices")
        guard let deviceState else { throw notPlanted() }
        return deviceState
    }
    @discardableResult
    func setDevice(_ id: String, enabled: Bool) async throws -> DeviceInfo {
        try record("setDevice")
        patchedDevices.append((id, enabled))
        if let setDeviceAnswer { return setDeviceAnswer }
        guard let row = deviceState?.devices.first(where: { $0.id == id }) else { throw notPlanted() }
        return try Self.mutated(row, enabled: enabled)
    }
    func resources() async throws -> ResourceSnapshot {
        try record("resources")
        guard let resourceSnapshot else { throw notPlanted() }
        return resourceSnapshot
    }
    func resourceStream() -> AsyncThrowingStream<ResourceSnapshot, Error> {
        callsLock.withLock { recorded.append("resourceStream") }
        return AsyncThrowingStream { continuation in
            self.resourceStreamContinuation = continuation
            continuation.onTermination = { _ in self.resourceStreamEnded = true }
        }
    }
    func peers() async throws -> [DiscoveryPeer] { try record("peers"); return peerRows }

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
        try record("media")
        guard let mediaAnswer else { throw notPlanted() }
        return mediaAnswer
    }
    func exportOptions() async throws -> ExportOptions {
        try record("exportOptions")
        guard let exportBlock else { throw notPlanted() }
        return exportBlock
    }
    func export(_ filename: String, format: String) async throws -> Data {
        try record("export"); throw notPlanted()
    }
    func playableURL(for filename: String) async throws -> URL {
        try record("playableURL")
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
        callsLock.withLock { recorded.append("events") }
        return AsyncThrowingStream { self.eventStream = $0 }
    }
    func batchEvents(id: String) -> AsyncThrowingStream<BatchStatus, Error> {
        callsLock.withLock { recorded.append("batchEvents") }
        return AsyncThrowingStream { $0.finish() }
    }
    func downloadEvents() -> AsyncThrowingStream<DownloadEvent, Error> {
        callsLock.withLock { recorded.append("downloadEvents") }
        return AsyncThrowingStream {
            $0.onTermination = { _ in self.downloadStreamEnded = true }
            self.downloadStream = $0
        }
    }
}
