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
    private func machine(_ name: String = "plato") -> MoldHost {
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
        let plato = machine()
        let backend = fake(for: plato)
        backend.queueListing = FakeFixtures.queueListing(["job-1"])
        let hosts = HostStore(hosts: [plato]) { _ in backend }
        let queue = QueueStore(hosts: hosts, coalesceDelay: .milliseconds(5))
        await connect(plato, hosts: hosts, backend: backend)

        for i in 0 ..< 64 { backend.emit(.job(.queued(id: "job-\(i)", model: "flux-dev"))) }

        await settle { backend.callCount("queue") == 1 }
        #expect(backend.callCount("queue") == 1)
        #expect(queue.entries(on: plato.id).map(\.id) == ["job-1"])
    }

    /// The server emits `generation_states_committed` explicitly so a bulk
    /// commit reconciles once (`types.rs:13163-13167`) -- the coalescer
    /// treats it the same as any other job frame.
    @Test func aBulkCommitFrameReconcilesTheMachineOnce() async {
        let plato = machine()
        let backend = fake(for: plato)
        backend.queueListing = FakeFixtures.queueListing(["job-1"])
        let hosts = HostStore(hosts: [plato]) { _ in backend }
        let queue = QueueStore(hosts: hosts, coalesceDelay: .milliseconds(5))
        await connect(plato, hosts: hosts, backend: backend)

        backend.emit(.job(.statesCommitted))

        await settle { backend.callCount("queue") == 1 }
        #expect(backend.callCount("queue") == 1)
        #expect(queue.entries(on: plato.id).map(\.id) == ["job-1"])
    }

    /// **Fails today**: there is no resync handling at all. A long
    /// coalescing delay proves this path skips the wait rather than just
    /// outrunning a short one -- the stream admitted it dropped deltas, and a
    /// delay would only widen the hole.
    @Test func aResyncReReadsWithoutWaiting() async {
        let plato = machine()
        let backend = fake(for: plato)
        backend.queueListing = FakeFixtures.queueListing(["job-1"])
        let hosts = HostStore(hosts: [plato]) { _ in backend }
        let queue = QueueStore(hosts: hosts, coalesceDelay: .seconds(60))
        await connect(plato, hosts: hosts, backend: backend)

        backend.emit(.resyncRequired)

        // Two different waits, not one polled on a fixed budget: the frame
        // still has to cross the stream to the watcher task before `apply`
        // stores the resync's own refresh in `coalescers[host]` (the same
        // slot a job frame's coalesced re-read uses), but once it is there
        // this AWAITS that exact task rather than polling for its side
        // effect -- a race against a simulated network call under a loaded
        // machine, which is what flaked before.
        await settle { queue.coalescers[plato.id] != nil }
        await queue.coalescers[plato.id]?.value
        #expect(backend.callCount("queue") == 1)
        #expect(queue.entries(on: plato.id).map(\.id) == ["job-1"])
    }

    /// A host whose capabilities never advertised `/api/events` gets no
    /// watcher at all -- `wantsPoll` says so -- but the ordinary listing read
    /// still works regardless, which is the fallback this pins.
    @Test func aMachineThatDoesNotAdvertiseEventsIsStillPolled() async {
        let plato = machine()
        let backend = fake(for: plato, events: false)
        backend.queueListing = FakeFixtures.queueListing(["job-1"])
        let hosts = HostStore(hosts: [plato]) { _ in backend }
        let queue = QueueStore(hosts: hosts)

        await hosts.refresh(plato)
        hosts.reconcileEventStreams()

        #expect(hosts.watchers[plato.id] == nil)
        #expect(queue.wantsPoll(plato.id) == true)

        await queue.refresh(on: plato.id)

        #expect(queue.entries(on: plato.id).map(\.id) == ["job-1"])
    }

    /// Three rows, three different batches, on screen at once -- their holds
    /// hydrate in the one call `POST /api/generation-batches/status` is for,
    /// not one request per batch.
    @Test func holdsAreHydratedInOneCallForEveryBatchOnScreen() async {
        let plato = machine()
        let backend = fake(for: plato)
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
        let hosts = HostStore(hosts: [plato]) { _ in backend }
        let queue = QueueStore(hosts: hosts)

        await queue.refresh(on: plato.id)

        #expect(backend.callCount("batchStatuses") == 1)
        #expect(Set(backend.batchStatusQueries.first ?? []) == ["batch-1", "batch-2", "batch-3"])
        #expect(queue.groups(on: plato.id).count == 3)
    }

    /// `BatchChild.supersedes(_:)` is what decides this, and the store must
    /// actually ask it: a second hydrate answering with an OLDER revision
    /// -- planted second in the fake's FIFO, as if it were a delayed
    /// response landing after a fresher one -- must not overwrite what the
    /// first hydrate already knew.
    @Test func aLateAnswerNeverOverwritesANewerChild() async {
        let plato = machine()
        let backend = fake(for: plato)
        backend.queueListing = FakeFixtures.queueListing(entries: [
            FakeFixtures.queueEntry("job-1", state: "held", batchId: "batch-1"),
        ])
        let newer = FakeFixtures.batchChild("job-1", revision: 5)
        let older = FakeFixtures.batchChild("job-1", revision: 2)
        backend.batchListings = [
            FakeFixtures.batchStatusListing([FakeFixtures.batchStatus(id: "batch-1", children: [newer])]),
            FakeFixtures.batchStatusListing([FakeFixtures.batchStatus(id: "batch-1", children: [older])]),
        ]
        let hosts = HostStore(hosts: [plato]) { _ in backend }
        let queue = QueueStore(hosts: hosts)

        await queue.refresh(on: plato.id)
        #expect(queue.children[plato.id]?["batch-1"]?.first?.revision == 5)

        await queue.refresh(on: plato.id)
        #expect(queue.children[plato.id]?["batch-1"]?.first?.revision == 5)
    }
}
