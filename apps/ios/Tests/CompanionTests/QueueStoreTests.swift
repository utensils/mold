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

    @Test func inputListingFailureCanBeRetriedWithoutReopeningDetails() async throws {
        let (queue, hosts, fake) = try await Self.setUp()
        let id = hosts.hosts[0].id
        let row = try #require(queue.listings[id]?.first { $0.id == "q1" })
        fake.stub("queueInputs(id:)") { _ in throw URLError(.timedOut) }
        await queue.loadSourceThumbnail(for: row, on: id, detailed: true)
        #expect(queue.inputLoadFailed(for: row, on: id))
        fake.stub("queueInputs(id:)", returning: [QueueInput]())
        await queue.loadSourceThumbnail(for: row, on: id, detailed: true, retry: true)
        #expect(!queue.inputLoadFailed(for: row, on: id))
    }

    @Test func cancellingAndOfflineRowsOfferNoMutations() async throws {
        let (queue, hosts, _) = try await Self.setUp(capabilities: #"{"queue":{"can_pause_job":true,"can_cancel_running":true}}"#)
        let host = hosts.hosts[0]
        let cancelling = try Self.decode(QueueEntry.self, #"{"id":"c","state":"cancelling"}"#)
        #expect(!queue.canCancel(cancelling, on: host.id))
        let entry = try #require(queue.listings[host.id]?.first { $0.id == "q1" })
        hosts.setReachability(.down("Offline"), for: host.id)
        #expect(!queue.canCancel(entry, on: host.id))
        #expect(!queue.canPause(entry, on: host.id))
    }

    @Test func everyQueueStateHasOnlyItsValidControls() async throws {
        let (queue, hosts, fake) = try await Self.setUp(capabilities: #"{"queue":{"can_pause_job":true,"cooperative_cancellation":true}}"#)
        let id = hosts.hosts[0].id
        for state in ["queued", "paused", "running", "held", "cancelling", "complete", "failed", "cancelled", "unknown"] {
            let listing = try Self.decode(QueueListing.self, "{\"entries\":[{\"id\":\"s\",\"state\":\"\(state)\",\"batch_id\":\"b\",\"client_batch_id\":\"cb\",\"retryable\":true}]}")
            fake.stub("queue()", returning: listing); await queue.poll(id)
            let row = listing.entries[0]
            #expect(queue.canCancel(row, on: id) == ["queued", "paused", "running", "held"].contains(state))
            #expect(queue.canPause(row, on: id) == ["queued", "paused"].contains(state))
            #expect(queue.canRetry(row, on: id) == (state == "held"))
        }
    }

    @Test func retryRequiresAuthorityAndExplicitRefusalsWin() async throws {
        let (queue, hosts, fake) = try await Self.setUp()
        let id = hosts.hosts[0].id
        for json in [#"{"entries":[{"id":"h","state":"held","retryable":true}]}"#,
                     #"{"entries":[{"id":"h","state":"held","batch_id":"b","client_batch_id":"cb","retryable":false}]}"#] {
            let listing = try Self.decode(QueueListing.self, json)
            fake.stub("queue()", returning: listing); await queue.poll(id)
            #expect(!queue.canRetry(listing.entries[0], on: id))
            await queue.retry(listing.entries[0], on: id)
        }
        #expect(fake.count("retryJob(_:)" ) == 0)
    }

    @Test func oneItemMutationRemainsGuardedThroughItsRefresh() async throws {
        let (queue, hosts, fake) = try await Self.setUp(capabilities: #"{"queue":{"can_pause_job":true}}"#)
        let id = hosts.hosts[0].id
        let row = try #require(queue.listings[id]?.first { $0.id == "q1" })
        fake.stub("pauseJob(id:)") { _ in
            await MainActor.run {
                #expect(queue.isActing(row, on: id))
                #expect(!queue.canPause(row, on: id))
            }
            await queue.setPaused(true, row, on: id)
            return ()
        }
        await queue.setPaused(true, row, on: id)
        #expect(fake.count("pauseJob(id:)") == 1)
        #expect(!queue.isActing(row, on: id))
    }

    @Test func retryDoesNotUseAChangedBatchIdentity() async throws {
        let (queue, hosts, fake) = try await Self.setUp()
        let id = hosts.hosts[0].id
        let original = try Self.decode(QueueListing.self, #"{"entries":[{"id":"h","state":"held","batch_id":"old","client_batch_id":"client","retryable":true}]}"#)
        fake.stub("queue()", returning: original); await queue.poll(id)
        let changed = try Self.decode(QueueListing.self, #"{"entries":[{"id":"h","state":"held","batch_id":"new","client_batch_id":"client","retryable":true}]}"#)
        fake.stub("queue()", returning: changed); await queue.poll(id)
        await queue.retry(original.entries[0], on: id)
        #expect(fake.count("retryJob(_:)" ) == 0)
    }

    @Test func orderedReferencePreviewsSurviveASingleMissingImage() async throws {
        let (queue, hosts, fake) = try await Self.setUp()
        let id = hosts.hosts[0].id
        let entry = try #require(queue.listings[id]?.first)
        fake.stub("queueInputs(id:)", returning: [QueueInput(index: 2, label: "Reference image 1", preview: true), QueueInput(index: 3, label: "Reference image 2", preview: true)])
        fake.stub("queueInputThumbnail(id:index:)") { args in
            if args[1] as? Int == 2 { throw MoldClientError.malformedResponse }
            return Data([4, 5, 6])
        }
        await queue.loadSourceThumbnail(for: entry, on: id, detailed: true)
        let inputs = queue.inputPreviews(for: entry, on: id)
        #expect(inputs.count == 2)
        #expect(inputs[0].bytes == nil)
        #expect(inputs[1].bytes == Data([4, 5, 6]))
        fake.stub("queueInputThumbnail(id:index:)", returning: Data([7, 8, 9]))
        await queue.loadSourceThumbnail(for: entry, on: id, detailed: true)
        let retried = queue.inputPreviews(for: entry, on: id)
        #expect(retried[0].bytes == Data([7, 8, 9]))
        #expect(retried[1].bytes == Data([4, 5, 6]))
    }

    @Test func sourceThumbnailUsesTheQueuedJobsPrivateRoute() async throws {
        let (queue, hosts, fake) = try await Self.setUp()
        let id = hosts.hosts[0].id
        let entry = try #require(queue.listings[id]?.first)
        let bytes = Data([1, 2, 3])
        fake.stub("queueInputs(id:)", returning: [QueueInput(label: "Source", preview: true)])
        fake.stub("queueInputThumbnail(id:)", returning: bytes)
        await queue.loadSourceThumbnail(for: entry, on: id)
        #expect(queue.sourceThumbnail(for: entry, on: id) == bytes)
    }

    @Test func sourcePreviewDisappearsWhenItsJobLeavesTheQueue() async throws {
        let (queue, hosts, fake) = try await Self.setUp()
        let id = hosts.hosts[0].id
        let entry = try #require(queue.listings[id]?.first)
        fake.stub("queueInputs(id:)", returning: [QueueInput(label: "Source", preview: true)])
        fake.stub("queueInputThumbnail(id:)", returning: Data([1, 2, 3]))
        await queue.loadSourceThumbnail(for: entry, on: id)
        #expect(queue.sourceThumbnail(for: entry, on: id) != nil)
        fake.stub("queue()", returning: try Self.decode(QueueListing.self, #"{"entries":[]}"#))
        await queue.poll(id)
        #expect(queue.sourceThumbnail(for: entry, on: id) == nil)
    }

    @Test func anInflightSourcePreviewCannotRestoreACompletedRow() async throws {
        let (queue, hosts, fake) = try await Self.setUp()
        let id = hosts.hosts[0].id
        let entry = try #require(queue.listings[id]?.first)
        fake.stub("queueInputs(id:)", returning: [QueueInput(label: "Source", preview: true)])
        fake.stub("queueInputThumbnail(id:)") { _ in
            fake.stub("queue()", returning: try Self.decode(QueueListing.self, #"{"entries":[]}"#))
            await queue.poll(id)
            return Data([1, 2, 3])
        }
        await queue.loadSourceThumbnail(for: entry, on: id)
        #expect(queue.sourceThumbnail(for: entry, on: id) == nil)
    }

    @Test func emptyQueuesMustHaveAnsweredBeforeTheyAreCalledEmpty() async throws {
        let (queue, hosts, fake) = try await Self.setUp()
        let host = hosts.hosts[0]
        fake.stub("queue()", returning: try Self.decode(QueueListing.self, #"{"entries":[]}"#))
        await queue.reload()
        #expect(queue.isEmpty)
        #expect(queue.unavailableMachines.isEmpty)
        hosts.setReachability(.down("Offline"), for: host.id)
        #expect(queue.unavailableMachines.map(\.id) == [host.id])
    }

    @Test func failedQueueReadInvalidatesAnEarlierEmptyAnswer() async throws {
        let (queue, hosts, fake) = try await Self.setUp()
        fake.stub("queue()") { _ in throw MoldClientError.unreachable("Offline") }
        await queue.reload()
        #expect(queue.unavailableMachines.map(\.id) == hosts.hosts.map(\.id))
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
