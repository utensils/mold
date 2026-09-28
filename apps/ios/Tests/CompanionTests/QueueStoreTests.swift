import Foundation
import MoldClient
import MoldClientTesting
import Testing

@testable import MoldCompanion

/// The Queue against a fake machine: the badge counts what is rendering or
/// held, a reorder names the machine's own index, a held row is cleared by
/// its own route, and the whole-queue verbs only reach machines that offer
/// them.
@MainActor
struct QueueStoreTests {
    nonisolated static func decode<T: Decodable>(_ type: T.Type, _ json: String) throws -> T {
        try MoldJSON.decoder.decode(T.self, from: Data(json.utf8))
    }

    nonisolated static let listing = #"{"entries":["#
        + #"{"id":"r1","model":"flux-dev:q4","state":"running","position":0},"#
        + #"{"id":"q1","model":"flux-dev:q4","state":"queued","position":1},"#
        + #"{"id":"q2","model":"sdxl","state":"queued","position":2},"#
        + #"{"id":"h1","model":"wan","state":"held","position":3,"held_reason":"Model not installed"}]}"#

    static func setUp(capabilities: String = #"{"queue":{"can_reorder":true,"can_pause":true,"can_cancel_all":true}}"#)
        async throws -> (QueueStore, HostStore, FakeBackend) {
        let fake = FakeBackend()
        fake.stub("status()", returning: try decode(ServerStatus.self,
            #"{"version":"0.32.0","busy":false,"uptime_secs":1,"instance_id":"inst-1"}"#))
        fake.stub("capabilities()", returning: try decode(Capabilities.self, capabilities))
        fake.stub("models()", returning: [Model]())
        fake.stub("queue()", returning: try decode(QueueListing.self, listing))
        let file = HostListFile(url: FileManager.default.temporaryDirectory.appending(path: "h-\(UUID()).json"))
        let hosts = HostStore(list: file, credentials: HostStoreTests.MemoryCredentials(), makeBackend: { _ in fake })
        try hosts.add(name: "workstation", address: "10.0.0.4", apiKey: nil, makeDefault: true)
        await hosts.refreshAll()
        let queue = QueueStore(hosts: hosts)
        await queue.reload()
        return (queue, hosts, fake)
    }

    @Test func theBadgeCountsRenderingAndHeldJobs() async throws {
        let (queue, hosts, _) = try await Self.setUp()
        #expect(queue.badge == 2)
        #expect(queue.groups(for: hosts.hosts[0].id).count == 4)
    }

    @Test func movingDownNamesTheQueuedIndexNotTheScreenRow() async throws {
        let (queue, hosts, fake) = try await Self.setUp()
        fake.stub("reorderJob(id:position:)") { _ in () }
        let id = hosts.hosts[0].id
        let q1 = try #require(queue.listings[id]?.first { $0.id == "q1" })
        await queue.move(q1, up: false, on: id)
        let call = try #require(fake.calls.first { $0.route == "reorderJob(id:position:)" })
        #expect(call.arguments.first as? String == "q1")
        // Among queued rows only: [q1, q2] -> [q2, q1], so index 1.
        #expect(call.arguments.last as? Int == 1)
    }

    @Test func aDragLandsBehindTheNearestReorderableRowNotAHeldOne() async throws {
        let entries = try Self.decode(QueueListing.self, #"{"entries":["#
            + #"{"id":"R","state":"running","position":0},{"id":"A","state":"queued","position":1},"#
            + #"{"id":"H","state":"held","position":2},{"id":"B","state":"queued","position":2}]}"#).merged
        let groups = QueueGroup.build(entries, children: [:])
        // B dropped just below H: [R, A, H | B] -> behind A, never the front.
        var after = groups
        after.removeAll { $0.id == "B" }
        #expect(QueueStore.neighbour(above: 3, in: after) == "A")
        #expect(QueueStore.neighbour(above: 1, in: after) == nil, "below only the running row: the front")
    }

    @Test func aHeldRowIsCancelledByItsOwnRoute() async throws {
        let (queue, hosts, fake) = try await Self.setUp()
        fake.stub("cancelHeldJob(id:)") { _ in true }
        let id = hosts.hosts[0].id
        let held = try #require(queue.listings[id]?.first { $0.id == "h1" })
        await queue.cancel(held, on: id)
        #expect(fake.count("cancelHeldJob(id:)") == 1)
        #expect(fake.count("cancelJob(id:)") == 0)
    }

    @Test func emptyingClearsWaitingJobsThenEachHeldOne() async throws {
        let (queue, hosts, fake) = try await Self.setUp()
        fake.stub("cancelAllQueued()", returning: try Self.decode(QueueCancelResult.self, #"{"cancelled":2}"#))
        fake.stub("cancelHeldJob(id:)") { _ in true }
        await queue.empty([hosts.hosts[0].id])
        #expect(fake.count("cancelAllQueued()") == 1)
        #expect(fake.count("cancelHeldJob(id:)") == 1)
    }

    @Test func aRunningJobHasNoCancelWhereTheMachineCannotStopSafely() async throws {
        let (queue, hosts, _) = try await Self.setUp(capabilities: #"{"queue":{}}"#)
        let id = hosts.hosts[0].id
        let running = try #require(queue.listings[id]?.first { $0.id == "r1" })
        #expect(!queue.canCancel(running, on: id))
        #expect(!queue.canReorder(on: id))
        #expect(queue.gateMachines.isEmpty, "no Pause Queue on a machine that does not offer it")
    }

    @Test func pausingTheQueueRecordsTheMachinesAnswer() async throws {
        let (queue, hosts, fake) = try await Self.setUp()
        fake.stub("pauseQueue()", returning: QueuePauseState(paused: true))
        let id = hosts.hosts[0].id
        #expect(!queue.isQueuePaused(id))
        await queue.setQueuePaused(true, on: [id])
        #expect(queue.isQueuePaused(id))
    }
}
