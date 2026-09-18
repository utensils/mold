import Foundation
import MoldClient
import Testing

@testable import Mold

/// The queue used to be poll-only: a job that started, ran and finished
/// between two visits to the pane was never seen at all. These pin the
/// event-driven half -- `QueueStore+Live` and `QueueStore+Batches` -- the way
/// `HostStoreLifecycleTests` pins `HostStore+Events`.
@MainActor
struct QueueStoreLiveTests {
    private func machine(_ name: String = "workstation") -> MoldHost {
        MoldHost(name: name, baseURL: URL(string: "http://\(name)")!)
    }

    /// A machine that answers, and advertises `/api/events` unless told
    /// otherwise.
    private func fake(for host: MoldHost, events: Bool = true) -> FakeBackend {
        let fake = FakeBackend(host: host)
        fake.serverStatus = FakeFixtures.serverStatus()
        fake.capabilityBlock = FakeFixtures.capabilities(events: events)
        fake.exportBlock = FakeFixtures.exportOptions()
        return fake
    }

    /// Opens this machine's live stream and waits for it, the same steps
    /// every test below needs before it can hand the fake a frame.
    private func connect(_ host: MoldHost, hosts: HostStore, backend: FakeBackend) async {
        await hosts.refresh(host)
        hosts.reconcileEventStreams()
        await settle { backend.callCount("events") == 1 }
    }

    /// **Fails today**: no frame reaches the store at all, so 64 `job_queued`
    /// frames never re-read the machine even once.
    @Test func aJobFrameReReadsThatMachineOnceForABurst() async {
        let workstation = machine()
        let backend = fake(for: workstation)
        backend.queueListing = FakeFixtures.queueListing(["job-1"])
        let hosts = HostStore(hosts: [workstation]) { _ in backend }
        // Long enough to outlast the burst. At 5 ms the window could elapse
        // while frames were still arriving, so `markDirty` started a SECOND
        // coalescer and a second read genuinely happened -- and nothing
        // bounded that at two. Widening the assertion to absorb it was the
        // wrong repair: the contract is one read, so the delay has to be one
        // that actually covers the burst.
        let queue = QueueStore(hosts: hosts, coalesceDelay: .milliseconds(500))
        await connect(workstation, hosts: hosts, backend: backend)

        for i in 0 ..< 64 { backend.emit(.job(.queued(id: "job-\(i)", model: "flux-dev"))) }

        // Settled on the ROWS, not on `callCount`: the fake records the call
        // before the store has applied its answer, so a count can be
        // satisfied by a read whose result is not in `byHost` yet.
        await settle { !queue.entries(on: workstation.id).isEmpty }
        #expect(queue.entries(on: workstation.id).map(\.id) == ["job-1"])
        #expect(backend.callCount("queue") == 1)
    }

    /// The server emits `generation_states_committed` explicitly so a bulk
    /// commit reconciles once (`types.rs:13163-13167`) -- the coalescer
    /// treats it the same as any other job frame.
    @Test func aBulkCommitFrameReconcilesTheMachineOnce() async {
        let workstation = machine()
        let backend = fake(for: workstation)
        backend.queueListing = FakeFixtures.queueListing(["job-1"])
        let hosts = HostStore(hosts: [workstation]) { _ in backend }
        let queue = QueueStore(hosts: hosts, coalesceDelay: .milliseconds(5))
        await connect(workstation, hosts: hosts, backend: backend)

        backend.emit(.job(.statesCommitted))

        await settle { !queue.entries(on: workstation.id).isEmpty }
        #expect(queue.entries(on: workstation.id).map(\.id) == ["job-1"])
        // ONE frame, so one coalescer, so one read. There is no path to two.
        #expect(backend.callCount("queue") == 1)
    }

    /// **Fails today**: there is no resync handling at all. A long
    /// coalescing delay proves this path skips the wait rather than just
    /// outrunning a short one -- the stream admitted it dropped deltas, and a
    /// delay would only widen the hole.
    @Test func aResyncReReadsWithoutWaiting() async {
        let workstation = machine()
        let backend = fake(for: workstation)
        backend.queueListing = FakeFixtures.queueListing(["job-1"])
        let hosts = HostStore(hosts: [workstation]) { _ in backend }
        let queue = QueueStore(hosts: hosts, coalesceDelay: .seconds(60))
        await connect(workstation, hosts: hosts, backend: backend)

        backend.emit(.resyncRequired)

        // Two different waits, not one polled on a fixed budget: the frame
        // still has to cross the stream to the watcher task before `apply`
        // stores the resync's own refresh in `coalescers[host]` (the same
        // slot a job frame's coalesced re-read uses), but once it is there
        // this AWAITS that exact task rather than polling for its side
        // effect -- a race against a simulated network call under a loaded
        // machine, which is what flaked before.
        await settle { queue.coalescers[workstation.id] != nil }
        await queue.coalescers[workstation.id]?.value
        #expect(backend.callCount("queue") == 1)
        #expect(queue.entries(on: workstation.id).map(\.id) == ["job-1"])
    }

    /// A host whose capabilities never advertised `/api/events` gets no
    /// watcher at all -- `wantsPoll` says so -- but the ordinary listing read
    /// still works regardless, which is the fallback this pins.
    @Test func aMachineThatDoesNotAdvertiseEventsIsStillPolled() async {
        let workstation = machine()
        let backend = fake(for: workstation, events: false)
        backend.queueListing = FakeFixtures.queueListing(["job-1"])
        let hosts = HostStore(hosts: [workstation]) { _ in backend }
        let queue = QueueStore(hosts: hosts)

        await hosts.refresh(workstation)
        hosts.reconcileEventStreams()

        #expect(hosts.watchers[workstation.id] == nil)
        #expect(queue.wantsPoll(workstation.id) == true)

        await queue.refresh(on: workstation.id)

        #expect(queue.entries(on: workstation.id).map(\.id) == ["job-1"])
    }

    // MARK: - A storm of repairs

    /// **Fails today**: `refresh(on:)` has no throttle at all. `.resyncRequired`
    /// cancels the pending coalescer and starts a fresh `refresh`, but neither
    /// `poll` nor `hydrateNow` ever checks `Task.isCancelled`, so `cancel()`
    /// stops nothing: K markers are K concurrent `GET /api/queue` calls and K
    /// chained batch-status reads.
    ///
    /// That composes badly with the stream, which emits one `.resyncRequired`
    /// per DROPPED frame: the repair runs on the main actor, the consumer
    /// falls further behind, more frames drop, more markers arrive. The
    /// mechanism that exists to close a gap is what widens it.
    ///
    /// A marker arriving while a read is in flight means "read once more
    /// after this one" -- never "read N more". One in flight, at most one
    /// queued behind it, however many callers ask.
    @Test func aStormOfRefreshesReadsTheMachineTwiceNotOncePerCaller() async {
        let workstation = machine()
        let backend = fake(for: workstation)
        backend.queueListing = FakeFixtures.queueListing(["job-1"])
        // The read has to SUSPEND, or no second caller can arrive during it.
        backend.queueYields = true
        let hosts = HostStore(hosts: [workstation]) { _ in backend }
        let queue = QueueStore(hosts: hosts)

        // Twelve callers, all created before any of them can run -- exactly
        // the shape a burst of markers takes.
        let callers = (0 ..< 12).map { _ in Task { await queue.refresh(on: workstation.id) } }
        for caller in callers { await caller.value }

        // Two, not twelve and not one: the first read, and the ONE re-read
        // the callers that arrived during it are owed. That second read
        // starts only when the first has finished, so it necessarily starts
        // after the last caller joined it.
        #expect(backend.callCount("queue") == 2)
        #expect(queue.entries(on: workstation.id).map(\.id) == ["job-1"])
        #expect(hosts.failures.isEmpty)
    }

    /// And one caller is one read -- the throttle must not invent a
    /// follow-up nobody asked for.
    @Test func oneRefreshIsOneRead() async {
        let workstation = machine()
        let backend = fake(for: workstation)
        backend.queueListing = FakeFixtures.queueListing(["job-1"])
        backend.queueYields = true
        let hosts = HostStore(hosts: [workstation]) { _ in backend }
        let queue = QueueStore(hosts: hosts)

        await queue.refresh(on: workstation.id)
        await queue.refresh(on: workstation.id)

        // Sequential callers are not a storm: each gets its own read.
        #expect(backend.callCount("queue") == 2)
    }

    /// The consumer side of the same rule, driven the way the stream drives
    /// it. A marker cancels the pending coalesce -- a gap must not wait out a
    /// delay -- and then asks for a read that the throttle above bounds.
    @Test func aBurstOfResyncMarkersRepairsTheMachineWithoutAReadEach() async {
        let workstation = machine()
        let backend = fake(for: workstation)
        backend.queueListing = FakeFixtures.queueListing(["job-1"])
        backend.queueYields = true
        let hosts = HostStore(hosts: [workstation]) { _ in backend }
        let queue = QueueStore(hosts: hosts, coalesceDelay: .seconds(60))

        for _ in 0 ..< 12 { queue.apply(.resyncRequired, from: workstation.id) }
        await queue.coalescers[workstation.id]?.value

        #expect(backend.callCount("queue") <= 2)
        #expect(queue.entries(on: workstation.id).map(\.id) == ["job-1"])
    }

    /// Three rows, three different batches, on screen at once -- their holds
    /// hydrate in the one call `POST /api/generation-batches/status` is for,
    /// not one request per batch.
    @Test func holdsAreHydratedInOneCallForEveryBatchOnScreen() async {
        let workstation = machine()
        let backend = fake(for: workstation)
        backend.queueListing = FakeFixtures.queueListing(entries: [
            FakeFixtures.queueEntry("job-1", batchId: "batch-1"),
            FakeFixtures.queueEntry("job-2", batchId: "batch-2"),
            FakeFixtures.queueEntry("job-3", batchId: "batch-3"),
        ])
        backend.batchListings = [
            FakeFixtures.batchStatusListing([
                FakeFixtures.batchStatus(id: "batch-1", [.init(0, jobId: "job-1")]),
                FakeFixtures.batchStatus(id: "batch-2", [.init(0, jobId: "job-2")]),
                FakeFixtures.batchStatus(id: "batch-3", [.init(0, jobId: "job-3")]),
            ]),
        ]
        let hosts = HostStore(hosts: [workstation]) { _ in backend }
        let queue = QueueStore(hosts: hosts)

        await queue.refresh(on: workstation.id)

        #expect(backend.callCount("batchStatuses") == 1)
        #expect(Set(backend.batchStatusQueries.first ?? []) == ["batch-1", "batch-2", "batch-3"])
        #expect(queue.groups(on: workstation.id).count == 3)
    }

    /// `BatchChild.supersedes(_:)` is what decides this, and the store must
    /// actually ask it: a second hydrate answering with an OLDER revision
    /// -- planted second in the fake's FIFO, as if it were a delayed
    /// response landing after a fresher one -- must not overwrite what the
    /// first hydrate already knew.
    @Test func aLateAnswerNeverOverwritesANewerChild() async {
        let workstation = machine()
        let backend = fake(for: workstation)
        backend.queueListing = FakeFixtures.queueListing(entries: [
            FakeFixtures.queueEntry("job-1", state: "held", batchId: "batch-1"),
        ])
        let newer = FakeFixtures.batchChild("job-1", revision: 5)
        let older = FakeFixtures.batchChild("job-1", revision: 2)
        backend.batchListings = [
            FakeFixtures.batchStatusListing([FakeFixtures.batchStatus(id: "batch-1", children: [newer])]),
            FakeFixtures.batchStatusListing([FakeFixtures.batchStatus(id: "batch-1", children: [older])]),
        ]
        let hosts = HostStore(hosts: [workstation]) { _ in backend }
        let queue = QueueStore(hosts: hosts)

        await queue.refresh(on: workstation.id)
        #expect(queue.children[workstation.id]?["batch-1"]?.first?.revision == 5)

        await queue.refresh(on: workstation.id)
        #expect(queue.children[workstation.id]?["batch-1"]?.first?.revision == 5)
    }
}
