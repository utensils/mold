import Foundation
import MoldClient
import MoldClientTesting
import Synchronization
import Testing

@testable import MoldCompanion

/// Models against a fake machine: Get starts a tracked download, a gated
/// model asks for its licence and then tries again, and a held row's Pull
/// retries the job only once the download completed.
@MainActor
struct ModelStoreTests {
    private func setUp() async throws -> (ModelStore, QueueStore, HostStore, FakeBackend) {
        let (queue, hosts, fake) = try await QueueStoreTests.setUp()
        fake.stub("downloads()", returning: DownloadsListing())
        fake.stubStream("downloadEvents()") { _ -> AsyncThrowingStream<DownloadEvent, Error> in AsyncThrowingStream { _ in } }
        return (ModelStore(hosts: hosts, queue: queue), queue, hosts, fake)
    }

    private func frame(_ json: String) throws -> DownloadEvent {
        try QueueStoreTests.decode(DownloadEvent.self, json)
    }

    @Test func unreadInventoryDoesNotClaimNothingIsInstalled() async throws {
        let (models, _, hosts, _) = try await setUp()
        let host = hosts.hosts[0]
        hosts.setModels(nil, for: host.id)
        hosts.setReachability(.down("Offline"), for: host.id)
        #expect(hosts.installed[host.id] == nil)
        #expect(models.emptyInventoryMessage(on: host) == "Connect to \(host.name) to see its installed models.")
        hosts.setReachability(.checking, for: host.id)
        #expect(models.emptyInventoryMessage(on: host) == "Checking installed models on \(host.name)…")
        hosts.setModels([], for: host.id)
        #expect(hosts.installed[host.id] == 0)
        #expect(models.emptyInventoryMessage(on: host).hasPrefix("Nothing installed"))
    }

    @Test func getStartsADownloadAndShowsItsRow() async throws {
        let (models, _, hosts, fake) = try await setUp()
        fake.stub("startDownload(_:)", returning: try QueueStoreTests.decode(DownloadTicket.self, #"{"id":"d1"}"#))
        let id = hosts.hosts[0].id
        await models.install("sdxl", on: id)
        #expect(models.isBusy("sdxl", on: id))
        models.apply(try frame(#"{"type":"progress","id":"d1","bytes_done":50,"bytes_total":100}"#), on: id)
        #expect(models.progress(for: "sdxl", on: id)?.row.fraction == 0.5)
    }

    @Test func aGatedModelAsksForItsLicenceThenTriesAgain() async throws {
        let (models, _, hosts, fake) = try await setUp()
        let refusal = LicenseRefusal(id: "flux-nc", name: "FLUX.1 [dev] Non-Commercial", url: "https://x",
                                     canonical: "https://x", sha256: "abc", summary: "Non-commercial use.")
        let accepted = Mutex(false)
        fake.stub("startDownload(_:)") { _ in
            guard accepted.withLock({ $0 }) else { throw MoldClientError.licenseRequired(refusal, mismatch: false) }
            return try QueueStoreTests.decode(DownloadTicket.self, #"{"id":"d1"}"#)
        }
        fake.stub("acceptLicenses(_:)") { _ in accepted.withLock { $0 = true }; return [ThirdPartyLicense]() }
        let id = hosts.hosts[0].id
        await models.install("flux-dev:q4", on: id)
        let pending = try #require(models.pendingLicense)
        #expect(pending.refusal.id == "flux-nc")
        await models.accept(pending)
        #expect(models.pendingLicense == nil)
        #expect(fake.count("startDownload(_:)") == 2)
        #expect(models.isBusy("flux-dev:q4", on: id))
    }

    @Test func pullRetriesTheHeldJobOnlyAfterTheDownloadCompletes() async throws {
        let (models, queue, hosts, fake) = try await setUp()
        fake.stub("startDownload(_:)", returning: try QueueStoreTests.decode(DownloadTicket.self, #"{"id":"d1"}"#))
        fake.stub("retryJob(_:)") { _ in () }
        let id = hosts.hosts[0].id
        let held = try QueueStoreTests.decode(QueueEntry.self,
            #"{"id":"h1","model":"wan","state":"held","batch_id":"b1","client_batch_id":"c1"}"#)
        fake.stub("queue()", returning: try QueueStoreTests.decode(QueueListing.self, #"{"entries":[{"id":"h1","model":"wan","state":"held","batch_id":"b1","client_batch_id":"c1"}]}"#))
        fake.stub("queueJob(id:)", returning: try QueueStoreTests.decode(QueueJobDetail.self, #"{"job":{"id":"h1","model":"wan","state":"held","batch_id":"b1","client_batch_id":"c1"}}"#))
        await queue.poll(id)
        models.pullThenRetry("wan", entry: held, on: id)
        try await waitUntil { models.isBusy("wan", on: id) }
        #expect(fake.count("retryJob(_:)") == 0, "not while it is still fetching")
        models.apply(try frame(#"{"type":"job_done","id":"d1","model":"wan"}"#), on: id)
        try await waitUntil { fake.count("retryJob(_:)") == 1 }
        _ = queue
    }

    @Test func aFailedPullNeverRetries() async throws {
        let (models, queue, hosts, fake) = try await setUp()
        fake.stub("startDownload(_:)", returning: try QueueStoreTests.decode(DownloadTicket.self, #"{"id":"d1"}"#))
        fake.stub("retryJob(_:)") { _ in () }
        let id = hosts.hosts[0].id
        let held = try QueueStoreTests.decode(QueueEntry.self,
            #"{"id":"h1","model":"wan","state":"held","batch_id":"b1","client_batch_id":"c1"}"#)
        fake.stub("queue()", returning: try QueueStoreTests.decode(QueueListing.self, #"{"entries":[{"id":"h1","model":"wan","state":"held","batch_id":"b1","client_batch_id":"c1"}]}"#))
        fake.stub("queueJob(id:)", returning: try QueueStoreTests.decode(QueueJobDetail.self, #"{"job":{"id":"h1","model":"wan","state":"held","batch_id":"b1","client_batch_id":"c1"}}"#))
        await queue.poll(id)
        models.pullThenRetry("wan", entry: held, on: id)
        try await waitUntil { models.isBusy("wan", on: id) }
        models.apply(try frame(#"{"type":"job_failed","id":"d1","model":"wan","error":"disk full"}"#), on: id)
        try await Task.sleep(for: .milliseconds(600))
        #expect(fake.count("retryJob(_:)") == 0)
        #expect(models.finished[id]?.first?.error == "disk full")
    }

    @Test func loadedInventoryIncludesResidentModelsWithoutDownloadMetadata() async throws {
        let (models, _, hosts, _) = try await setUp()
        let resident = try QueueStoreTests.decode(Model.self,
            #"{"name":"flux-dev:q4","family":"flux","description":"Resident","is_loaded":true}"#)
        hosts.setModels([resident], for: hosts.hosts[0].id)
        #expect(models.loaded(on: hosts.hosts[0].id).map(\.name) == [resident.name])
    }

    @Test func unloadAllTargetsOnlyTheSelectedServerAndRefreshesItsInventory() async throws {
        let (models, _, hosts, fake) = try await setUp()
        fake.stub("unloadModel(model:gpu:)") { arguments in
            #expect((arguments[0] as? String?) == .some(nil))
            #expect((arguments[1] as? Int?) == .some(nil))
            return ()
        }
        let before = fake.count("models()")
        await models.unloadAll(on: hosts.hosts[0].id)
        #expect(fake.count("unloadModel(model:gpu:)") == 1)
        #expect(fake.count("models()") == before + 1)
        #expect(models.changing.isEmpty)
    }

    @Test func unloadRefusalIsVisibleAndClearsInFlightState() async throws {
        let (models, _, hosts, fake) = try await setUp()
        fake.stub("unloadModel(model:gpu:)") { _ in throw MoldClientError.unreachable("Server is busy") }
        await models.unloadAll(on: hosts.hosts[0].id)
        #expect(hosts.failures.contains { $0.doing.contains("unload") })
        #expect(models.changing.isEmpty)
    }

    @Test func anUnloadBlocksConflictingChangesOnThatServerOnly() async throws {
        let (models, _, hosts, fake) = try await setUp()
        let gate = AsyncStream<Void>.makeStream()
        fake.stub("unloadModel(model:gpu:)") { _ in
            for await _ in gate.stream { break }
            return ()
        }
        let id = hosts.hosts[0].id
        let resident = try QueueStoreTests.decode(Model.self,
            #"{"name":"flux-dev:q4","family":"flux","description":"Resident","is_loaded":true}"#)
        let first = Task { await models.unload(resident, on: id) }
        try await waitUntil { fake.count("unloadModel(model:gpu:)") == 1 }
        #expect(models.changing.contains(id))
        await models.unloadAll(on: id)
        await models.load(resident, on: id)
        await models.delete(resident, on: id)
        #expect(fake.count("unloadModel(model:gpu:)") == 1)
        #expect(fake.count("loadModel(_:gpu:)") == 0)
        #expect(fake.count("deleteModel(_:)") == 0)
        let call = try #require(fake.calls.last { $0.route == "unloadModel(model:gpu:)" })
        #expect((call.arguments[0] as? String?) == .some(resident.name))
        #expect((call.arguments[1] as? Int?) == .some(nil))
        try hosts.add(name: "Other", address: "10.0.0.5", apiKey: nil, makeDefault: false)
        let other = try #require(hosts.hosts.first { $0.id != id })
        let second = Task { await models.unloadAll(on: other.id) }
        try await waitUntil { fake.count("unloadModel(model:gpu:)") == 2 }
        #expect(models.changing.contains(other.id))
        gate.continuation.finish()
        await first.value
        await second.value
        #expect(models.changing.isEmpty)
    }

    private func waitUntil(_ condition: () -> Bool) async throws {
        for _ in 0..<200 where !condition() { try await Task.sleep(for: .milliseconds(20)) }
        #expect(condition())
    }
}
