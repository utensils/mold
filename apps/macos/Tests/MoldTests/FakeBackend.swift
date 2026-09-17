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
/// Settle on the STATE the assertion reads, never on a call count: a fake
/// records the call before the store has applied its answer, so a count can be
/// satisfied by work that has not landed. A budget is NOT a fix for that, and
/// no test here is allowed to pass because this number is big.
///
/// What it is for: a real wait that is longer than a schedule.
/// `aJobFrameReReadsThatMachineOnceForABurst` coalesces 64 frames behind a
/// 500 ms delay -- a delay chosen to outlast the burst, which is the contract
/// -- so the answer legitimately cannot arrive inside the old ~0.5 s tail.
/// Two seconds covers it with room for a main actor shared by fifty-eight
/// suites. It returns the instant the condition holds, so a green run costs
/// exactly what it did before; only a genuinely slow one waits longer.
@MainActor
func settle(until condition: () -> Bool) async {
    for step in 0 ..< 500 {
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
    nonisolated(unsafe) var thumbnailAnswer: Data?
    /// Every mesh export body this double was asked for, in order.
    nonisolated(unsafe) var exportRequests: [MeshExportRequest] = []
    nonisolated(unsafe) var downloadTicket: DownloadTicket?
    /// Every model name `startDownload` was asked to fetch, in call order --
    /// what a licence retry actually resent.
    nonisolated(unsafe) var startedDownloads: [String] = []
    nonisolated(unsafe) var modelRows: [Model] = []
    // MARK: - Models (M5 S1b)

    /// Answered per MODEL, the same "absent is unplanted" rule as `loraRows`.
    nonisolated(unsafe) var componentRows: [String: ModelComponentsResponse] = [:]
    nonisolated(unsafe) var removalAnswers: [String: ModelRemoval] = [:]
    /// Every model `deleteModel` was asked to remove, in call order.
    nonisolated(unsafe) var deletedModels: [String] = []
    /// Every `(model, gpu)` `loadModel` was asked for, in call order.
    nonisolated(unsafe) var loadedModels: [(model: String, gpu: Int?)] = []
    /// Every `(model, gpu)` `unloadModel` was asked for, in call order --
    /// `model == nil` is "unload everything".
    nonisolated(unsafe) var unloadedModels: [(model: String?, gpu: Int?)] = []
    nonisolated(unsafe) var downloadsListing: DownloadsListing?

    // MARK: - Catalog (M5 S1b)

    /// Answered per QUERY STRING -- a test plants what one particular search
    /// answers, not a fixed listing for every call.
    nonisolated(unsafe) var catalogPages: [String: CatalogListing] = [:]
    nonisolated(unsafe) var catalogInstallAnswer: CatalogInstall?
    /// Every id `installCatalogEntry` was asked to install, in call order.
    nonisolated(unsafe) var catalogInstalls: [String] = []
    nonisolated(unsafe) var credentialStatus: CatalogCredentialStatus?
    /// Every `(provider, token)` `setCatalogCredential` was asked to write,
    /// in call order.
    nonisolated(unsafe) var credentialWrites: [(provider: String, token: String)] = []
    /// Every provider `clearCatalogCredential` was asked to clear, in call
    /// order.
    nonisolated(unsafe) var credentialClears: [String] = []
    /// Every admission `submit` was asked, in call order -- what a batch of
    /// four actually looked like on the wire.
    nonisolated(unsafe) var submittedAdmissions: [BatchAdmission] = []
    nonisolated(unsafe) var submitAnswer: BatchStatus?
    /// Consumed FIFO, ahead of `submitAnswer` -- for a test where two
    /// `submit` calls must come back with DIFFERENT ids (e.g. a queued
    /// second batch), since `submitAnswer` alone can only ever answer the
    /// same one. Empty falls back to `submitAnswer`, so every existing test
    /// is unaffected.
    nonisolated(unsafe) var submitAnswers: [BatchStatus] = []
    /// Every batch id `cancelBatch` was asked to stop, in call order -- WHICH
    /// batch a Stop reached, not merely that one did.
    nonisolated(unsafe) var cancelledBatchIds: [String] = []
    /// Set before a `submit` to hold it in the air until `releaseSubmit()`.
    nonisolated(unsafe) var holdsSubmit = false
    nonisolated(unsafe) private var submitGate: (() -> Void)?
    /// A release that arrived before the gate was installed -- `settle` sees
    /// the call RECORDED before `submit` has suspended, so without this (and
    /// without the lock ordering the two) a release lands in the window
    /// between and the test hangs to its timeout.
    nonisolated(unsafe) private var submitReleased = false
    /// `submitAnswers`, `submittedAdmissions` and the gate are all touched
    /// from two cooperative threads the moment two presses overlap. Unguarded
    /// `Array.removeFirst()` from two threads is a data race AND makes which
    /// press gets which answer a coin toss -- which is the real cause of
    /// `aSecondGenerateWhileOneRunsIsAdmittedAndQueued`'s flake, not a
    /// `settle` timing out.
    private let submitLock = NSLock()
    /// Planted per `id`, since a test drives `submit` then reads the same
    /// batch back through `batchStatus(id:)` once its events stream ends.
    nonisolated(unsafe) var batchStatusAnswers: [String: BatchStatus] = [:]
    /// Ids this fake holds `batchEvents(id:)` open for, the same pattern as
    /// `eventStream`: a test pushes frames at its own pace with
    /// `emitBatchEvent(_:for:)` instead of the stream finishing before the
    /// next `submit` has had a chance to land in `queued`. An id absent here
    /// keeps the old behaviour -- an empty stream that finishes at once, so
    /// `follow()` falls through to its one-shot `batchStatus` read.
    nonisolated(unsafe) var batchEventsHeldOpen: Set<String> = []
    nonisolated(unsafe) var batchEventsContinuations:
        [String: AsyncThrowingStream<BatchStatus, Error>.Continuation] = [:]
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

    /// Not `private`: `FakeBackend+Models.swift` (M5 S1b's routes, split out
    /// to keep this file from growing further) calls both from a different
    /// file in the same type.
    /// How long a route takes to answer. For a test that has to make
    /// something happen WHILE a call is in flight -- an edit enqueued during
    /// a drain's trailing re-list, say -- which no planted answer can express.
    /// Awaited by the routes that read it; a route with no entry is instant,
    /// so nothing existing changes.
    nonisolated(unsafe) var delays: [String: Duration] = [:]

    /// Holds a route open for its planted delay. `await`ed, never slept on the
    /// caller's behalf: the store's task suspends exactly where a real round
    /// trip would.
    func pause(_ route: String) async {
        guard let delay = delays[route] else { return }
        try? await Task.sleep(for: delay)
    }

    func record(_ route: String) throws {
        callsLock.withLock { recorded.append(route) }
        if let planted = plantedErrors[route] { throw planted }
        if refuses.contains(route) {
            throw MoldClientError.http(status: 409, code: nil, message: "Refused by the fake.")
        }
    }

    func notPlanted() -> Error { MoldClientError.unreachable("not planted") }

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

    /// Holds `status()` open until `releaseStatus()`, the way a machine that
    /// is off holds a connection until it times out. A test uses it to prove
    /// that asking one machine does not stop the others being asked.
    nonisolated(unsafe) var statusHeldOpen = false {
        // Setting the hold ARMS it: hold → release → hold again is a test
        // this seam should answer, and a latch that is never cleared would
        // silently not hold the second time.
        didSet { if statusHeldOpen { statusReleased = false } }
    }
    nonisolated(unsafe) private var statusWaiters: [CheckedContinuation<Void, Never>] = []
    /// The release is a LATCH, not a broadcast. `releaseStatus()` used to
    /// resume whoever happened to be waiting at that instant, so a `status()`
    /// that had recorded its call but not yet parked its continuation waited
    /// for a wake-up that had already happened -- and `settle`, being bounded,
    /// returned long before the test's own `await tick.value` hung the whole
    /// bundle. Measured: 22 minutes, one suite, no output.
    nonisolated(unsafe) private var statusReleased = false

    func releaseStatus() {
        statusReleased = true
        let waiting = statusWaiters
        statusWaiters = []
        for continuation in waiting { continuation.resume() }
    }

    func status() async throws -> ServerStatus {
        try record("status")
        if statusHeldOpen, !statusReleased {
            await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
                statusWaiters.append(continuation)
            }
        }
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
    /// Every request the probe actually sent -- WHAT a planning read carried,
    /// not merely that one happened (finding 02#5).
    nonisolated(unsafe) var placementRequests: [GenerateRequest] = []

    func placementPreview(_ request: GenerateRequest, copies: Int) async throws -> PlacementPreview {
        try record("placementPreview")
        placementCopiesRequested.append(copies)
        placementRequests.append(request)
        throw notPlanted()
    }
    func submit(_ admission: BatchAdmission) async throws -> BatchStatus {
        try record("submit")
        // The answer is claimed BEFORE any suspension, under the lock, so two
        // overlapping presses get their answers in submission order however
        // their tasks interleave.
        let (claimed, holding) = submitLock.withLock { () -> (BatchStatus?, Bool) in
            submittedAdmissions.append(admission)
            let answer = submitAnswers.isEmpty ? nil : submitAnswers.removeFirst()
            let held = holdsSubmit
            holdsSubmit = false
            return (answer, held)
        }

        // Held open so a test can press Stop while an admission is GENUINELY
        // in the air -- the window finding 02#2 is about.
        if holding {
            await withCheckedContinuation { continuation in
                submitLock.lock()
                if submitReleased {
                    submitReleased = false
                    submitLock.unlock()
                    continuation.resume()
                } else {
                    submitGate = { continuation.resume() }
                    submitLock.unlock()
                }
            }
        }
        if let claimed { return claimed }
        guard let submitAnswer else { throw notPlanted() }
        return submitAnswer
    }

    /// Lets a held-open `submit` answer, whether or not it has suspended yet.
    func releaseSubmit() {
        submitLock.lock()
        let gate = submitGate
        submitGate = nil
        if gate == nil { submitReleased = true }
        submitLock.unlock()
        gate?()
    }
    func batchStatus(id: String) async throws -> BatchStatus {
        try record("batchStatus")
        guard let status = batchStatusAnswers[id] else { throw notPlanted() }
        return status
    }
    /// Answered per CLIENT BATCH ID. Unplanted is a real 404, not
    /// `notPlanted()`'s bare `.unreachable` -- a transfer's `checkPriorAttempt`
    /// reads "nothing landed yet" from exactly that status, and a test that
    /// wants a genuine transport failure here plants one in `plantedErrors`.
    func batchStatus(clientBatchId: String) async throws -> BatchStatus {
        try record("batchStatusByClientId")
        guard let status = batchStatusByClientId[clientBatchId] else {
            throw MoldClientError.http(status: 404, code: nil, message: "No batch with that client id.")
        }
        return status
    }
    func jobPreview(jobId: String) async throws -> JobProgress? {
        try record("jobPreview")
        jobPreviewCalls.append(jobId)
        return nil
    }
    func cancelBatch(id: String) async throws {
        try record("cancelBatch")
        // Two withdrawn submissions can land from two threads.
        submitLock.withLock { cancelledBatchIds.append(id) }
    }

    // MARK: - Create

    nonisolated(unsafe) var expandAnswer: ExpandResponse?
    /// Every rewrite asked for, in call order -- WHICH task the app sent,
    /// not merely that it asked (findings 01#13, 02#12).
    nonisolated(unsafe) var expandRequests: [ExpandRequest] = []
    nonisolated(unsafe) var remixRequests: [RemixRequest] = []
    /// Set before an `expand` to hold it in the air until `releaseExpand()`,
    /// so a test can move the box while a rewrite is in flight (02#13).
    nonisolated(unsafe) var holdsExpand = false
    nonisolated(unsafe) private var expandGate: (() -> Void)?
    /// A release that arrived before the gate was installed. The test's
    /// `settle` sees the call recorded before `expand` has suspended, so
    /// without this -- and without the lock ordering the two -- a release can
    /// land in the window between and hang for ever.
    nonisolated(unsafe) private var expandReleased = false
    private let expandLock = NSLock()
    nonisolated(unsafe) var remixAnswer: RemixResponse?
    /// `nil` throws as unplanted; `[]` is a real empty history, same rule as
    /// every other listing on this fake.
    nonisolated(unsafe) var historyRows: [HistoryEntry]?
    /// What `clearHistory` was asked, in call order -- `nil` is "clear
    /// everything", a number is the `keep` it trimmed to.
    nonisolated(unsafe) var historyCleared: [Int?] = []

    func expand(_ request: ExpandRequest) async throws -> ExpandResponse {
        try record("expand")
        expandRequests.append(request)
        if holdsExpand {
            holdsExpand = false
            await withCheckedContinuation { continuation in
                expandLock.lock()
                if expandReleased {
                    expandReleased = false
                    expandLock.unlock()
                    continuation.resume()
                } else {
                    expandGate = { continuation.resume() }
                    expandLock.unlock()
                }
            }
        }
        guard let expandAnswer else { throw notPlanted() }
        return expandAnswer
    }
    /// Lets a held-open `expand` answer, whether or not it has suspended yet.
    func releaseExpand() {
        expandLock.lock()
        let gate = expandGate
        expandGate = nil
        if gate == nil { expandReleased = true }
        expandLock.unlock()
        gate?()
    }
    func remix(_ request: RemixRequest) async throws -> RemixResponse {
        try record("remix")
        remixRequests.append(request)
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

    // MARK: - Config (M7 S1)

    /// Route implementations live in `FakeBackend+Config.swift` -- an
    /// extension cannot declare stored properties, the same rule
    /// `FakeBackend+Models.swift` documents.
    nonisolated(unsafe) var configListing: ConfigListing?
    /// What `setConfig` was asked, in call order.
    nonisolated(unsafe) var configWrites: [(String, ConfigScalar)] = []
    /// What `resetConfig` was asked, in call order.
    nonisolated(unsafe) var configResets: [String] = []
    nonisolated(unsafe) var profilesAnswer: ConfigProfiles?
    /// Consumed FIFO by `pairingSession()`, so a test can plant a keyed
    /// answer then a keyless one and see a store react to each in turn.
    nonisolated(unsafe) var pairingSessions: [PairingSession] = []
    nonisolated(unsafe) var pairedClientsAnswer: PairedClients?
    /// Every id `revokePairedClient` was asked to revoke, in call order.
    nonisolated(unsafe) var revokedClients: [String] = []

    // MARK: - Queue

    /// Answered per JOB ID, same "absent is unplanted" rule as every other
    /// listing here.
    nonisolated(unsafe) var queueJobDetails: [String: QueueJobDetail] = [:]
    /// Answered per CLIENT BATCH ID by `batchStatus(clientBatchId:)` -- absent
    /// is a real 404 ("nothing landed yet"), not `notPlanted()`.
    nonisolated(unsafe) var batchStatusByClientId: [String: BatchStatus] = [:]
    /// A FIFO per host id, so a test can plant "before" and "after" a retry
    /// and see the store pick up the second answer on its next call.
    nonisolated(unsafe) var batchListings: [BatchStatusListing] = []
    /// Every `(id, position)` `reorderJob` was asked for, in call order.
    nonisolated(unsafe) var reorders: [(id: String, position: Int)] = []
    /// Every `cancelAllQueued` call's answer, planted per call in order --
    /// `nil` throws as unplanted, same rule as everything else on this fake.
    nonisolated(unsafe) var cancelAllAnswer: QueueCancelResult?
    /// Set once `cancelAllQueued` is actually called -- a test asserting it
    /// was NOT called reads this rather than `calls.contains`.
    nonisolated(unsafe) var cancelledAll = false
    /// Every authority `retryJob` was asked to retry, in call order -- what a
    /// retry actually sent, not just that one was sent.
    nonisolated(unsafe) var retriedAuthorities: [QueueAuthority] = []
    /// Every ids array `batchStatuses` was asked for, in call order.
    nonisolated(unsafe) var batchStatusQueries: [[String]] = []
    /// Makes `batchStatuses` actually SUSPEND. Every other route here answers
    /// without one, so an `async` call to it runs straight through and two
    /// "concurrent" callers never interleave at all -- which is the only way
    /// to reproduce two overlapping hydrations. Off by default: it changes
    /// the scheduling of every test that reads a batch.
    nonisolated(unsafe) var batchStatusesYields = false
    /// Answered per JOB ID, same "absent is unplanted" rule as `queueJobDetails`.
    nonisolated(unsafe) var exportBodies: [String: Data] = [:]
    /// Every admission `admitTransfer` was asked for, in call order -- what a
    /// transfer's assembled body actually carried.
    nonisolated(unsafe) var transferAdmissions:
        [(clientBatchId: String, body: Data, destination: String)] = []
    nonisolated(unsafe) var admitAnswer: BatchStatus?
    /// Every authority `completeTransfer` was asked to cancel, in call order.
    nonisolated(unsafe) var completedTransfers: [QueueAuthority] = []

    /// Makes `queue` actually SUSPEND, the same reason as
    /// `batchStatusesYields` above: without one, a second caller arriving
    /// "while a read is in flight" can never actually arrive, because the
    /// first ran straight through. Off by default.
    nonisolated(unsafe) var queueYields = false

    func queue() async throws -> QueueListing {
        try record("queue")
        if queueYields { await Task.yield() }
        guard let queueListing else { throw notPlanted() }
        return queueListing
    }
    func queueJob(id: String) async throws -> QueueJobDetail {
        try record("queueJob")
        guard let detail = queueJobDetails[id] else { throw notPlanted() }
        return detail
    }
    func cancelJob(id: String) async throws { try record("cancelJob") }
    func pauseJob(id: String) async throws { try record("pauseJob") }
    func resumeJob(id: String) async throws { try record("resumeJob") }
    func reorderJob(id: String, position: Int) async throws {
        try record("reorderJob")
        reorders.append((id: id, position: position))
    }
    func retryJob(_ authority: QueueAuthority) async throws {
        try record("retryJob")
        retriedAuthorities.append(authority)
    }
    @discardableResult
    func cancelAllQueued() async throws -> QueueCancelResult {
        try record("cancelAllQueued")
        cancelledAll = true
        guard let cancelAllAnswer else { throw notPlanted() }
        return cancelAllAnswer
    }
    func batchStatuses(batchIds: [String]) async throws -> BatchStatusListing {
        try record("batchStatuses")
        batchStatusQueries.append(batchIds)
        if batchStatusesYields { await Task.yield() }
        guard !batchListings.isEmpty else { throw notPlanted() }
        return batchListings.removeFirst()
    }
    func exportHeldJob(_ authority: QueueAuthority) async throws -> Data {
        try record("exportHeldJob")
        guard let body = exportBodies[authority.jobId] else { throw notPlanted() }
        return body
    }
    func admitTransfer(
        clientBatchId: String, portable: Data, destinationInstance: String
    ) async throws -> BatchStatus {
        try record("admitTransfer")
        transferAdmissions.append(
            (clientBatchId: clientBatchId, body: portable, destination: destinationInstance))
        guard let admitAnswer else { throw notPlanted() }
        return admitAnswer
    }
    func completeTransfer(_ authority: QueueAuthority) async throws {
        try record("completeTransfer")
        completedTransfers.append(authority)
    }

    // MARK: - Downloads

    func startDownload(_ request: DownloadRequest) async throws -> DownloadTicket {
        try record("startDownload")
        startedDownloads.append(request.model)
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
        await pause("gallery")
        return .fresh(prints, etag: "fake-etag")
    }
    func trashedPrints(etag: String?) async throws -> Fetched<[GalleryPrint]> {
        try record("trashedPrints")
        return .fresh(trashedRows, etag: "fake-etag")
    }
    func patch(_ filename: String, with patch: GalleryPatch) async throws { try record("patch") }
    func mutate(_ mutation: GalleryBulkMutation) async throws {
        try record("mutate")
        await pause("mutate")
    }
    func trash(_ filenames: [String]) async throws { try record("trash") }
    func restoreFromTrash(_ filenames: [String]) async throws { try record("restoreFromTrash") }
    func deleteForever(_ filenames: [String]) async throws { try record("deleteForever") }
    /// Every import's filename, in order -- what a BATCH actually sent, which
    /// a call count cannot say.
    nonisolated(unsafe) var importedNames: [String] = []

    @discardableResult
    func importPrint(_ item: GalleryImport, as filename: String) async throws -> String {
        try record("importPrint")
        importedNames.append(filename)
        return filename
    }
    func media(_ filename: String, trashed: Bool) async throws -> Data {
        try record("media")
        guard let mediaAnswer else { throw notPlanted() }
        return mediaAnswer
    }
    func thumbnail(_ filename: String, size: Int, trashed: Bool) async throws -> Data {
        try record("thumbnail")
        guard let thumbnailAnswer else { throw notPlanted() }
        return thumbnailAnswer
    }
    func exportOptions() async throws -> ExportOptions {
        try record("exportOptions")
        guard let exportBlock else { throw notPlanted() }
        return exportBlock
    }
    func export(_ filename: String, format: String) async throws -> Data {
        try record("export"); throw notPlanted()
    }
    /// The body is recorded, not dropped: what a mesh export ASKS for is the
    /// thing worth asserting.
    func export(_ filename: String, request: MeshExportRequest) async throws -> Data {
        exportRequests.append(request)
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
        guard batchEventsHeldOpen.contains(id) else {
            return AsyncThrowingStream { $0.finish() }
        }
        return AsyncThrowingStream { continuation in
            self.batchEventsContinuations[id] = continuation
        }
    }
    /// Pushes one frame into an id's held-open `batchEvents` stream.
    func emitBatchEvent(_ status: BatchStatus, for id: String) {
        batchEventsContinuations[id]?.yield(status)
    }
    /// Ends an id's held-open `batchEvents` stream -- with no settled frame
    /// pushed, `follow()` falls through to its one-shot `batchStatus` read,
    /// exactly like the default unplanted stream.
    func finishBatchEvents(for id: String) {
        batchEventsContinuations[id]?.finish()
    }
    func downloadEvents() -> AsyncThrowingStream<DownloadEvent, Error> {
        callsLock.withLock { recorded.append("downloadEvents") }
        return AsyncThrowingStream {
            $0.onTermination = { _ in self.downloadStreamEnded = true }
            self.downloadStream = $0
        }
    }
}
