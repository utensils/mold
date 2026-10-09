import Foundation
import MoldClient
import Testing

@testable import Mold

/// `QueueStore` used to reach a machine through a downcast to `HTTPBackend`;
/// a failed cast was `nil`, nothing threw, and a cancel that never left the
/// machine still reported success. `MoldBackend` now carries `cancelJob`
/// directly, so there is no cast left to swallow -- this pins the outcome
/// that mattered: a refusal is REPORTED, not silently treated as done.
@MainActor
struct QueueStoreTests {
    @Test func cancellingAJobTheMachineRefusesIsNotReportedAsDone() async {
        let machine = MoldHost(name: "workstation", baseURL: URL(string: "http://workstation")!)
        let fake = FakeBackend(host: machine)
        fake.refuses = ["cancelJob"]
        fake.serverStatus = FakeFixtures.serverStatus()
        fake.capabilityBlock = FakeFixtures.capabilities()
        fake.queueListing = FakeFixtures.queueListing(["job-1"])
        let hosts = HostStore(hosts: [machine]) { _ in fake }
        let queue = QueueStore(hosts: hosts)
        let entry = FakeFixtures.queueEntry("job-1")

        await hosts.refresh(machine)
        await queue.poll(machine.id)
        await queue.cancel(entry, on: machine.id)

        #expect(fake.calls.contains("cancelJob"))
        #expect(hosts.failures.contains { $0.host == machine.id })
    }

    /// **Fails today**: `refresh`'s `?? []` writes an empty array for a
    /// machine whose fetch failed, so a transient hiccup blanks rows that
    /// were showing a second ago.
    @Test func aMachineThatCannotListItsQueueKeepsTheRowsItLastShowed() async {
        let machine = MoldHost(name: "workstation", baseURL: URL(string: "http://workstation")!)
        let fake = FakeBackend(host: machine)
        let hosts = HostStore(hosts: [machine]) { _ in fake }
        let queue = QueueStore(hosts: hosts)

        fake.queueListing = FakeFixtures.queueListing(["job-1"])
        await queue.refresh()
        #expect(queue.entries(on: machine.id).map(\.id) == ["job-1"])

        fake.refuses = ["queue"]
        await queue.refresh()

        #expect(queue.entries(on: machine.id).map(\.id) == ["job-1"])
        #expect(hosts.failures.contains { $0.host == machine.id && $0.verb == "list its queue" })
    }

    /// `hasLoaded` is what tells the Machines page "None installed" from "we
    /// haven't asked yet" -- a never-listed host has no key in `byHost` at all.
    @Test func aHostThatHasNeverBeenListedHasNotLoaded() async {
        let machine = MoldHost(name: "workstation", baseURL: URL(string: "http://workstation")!)
        let hosts = HostStore(hosts: [machine])
        let queue = QueueStore(hosts: hosts)

        #expect(queue.hasLoaded(on: machine.id) == false)
    }

    @Test func refreshingOneHostLoadsOnlyThatHostsQueue() async {
        let workstation = MoldHost(name: "workstation", baseURL: URL(string: "http://workstation")!)
        let fake = FakeBackend(host: workstation)
        fake.queueListing = FakeFixtures.queueListing(["job-1"])
        let hosts = HostStore(hosts: [workstation]) { _ in fake }
        let queue = QueueStore(hosts: hosts)

        await queue.refresh(on: workstation.id)

        #expect(queue.hasLoaded(on: workstation.id) == true)
        #expect(queue.entries(on: workstation.id).map(\.id) == ["job-1"])
    }
    @Test func heldCancellationCannotStopAJobThatStartedAfterTheRowWasDrawn() async throws {
        let machine = MoldHost(name: "workstation", baseURL: URL(string: "http://workstation")!)
        let fake = FakeBackend(host: machine)
        fake.serverStatus = FakeFixtures.serverStatus()
        fake.capabilityBlock = FakeFixtures.capabilities()
        let hosts = HostStore(hosts: [machine]) { _ in fake }
        await hosts.refresh(machine)
        let queue = QueueStore(hosts: hosts)
        let held = try MoldJSON.decoder.decode(QueueListing.self, from: Data(#"{"entries":[{"id":"h","state":"held"}]}"#.utf8))
        fake.queueListing = held
        await queue.poll(machine.id)
        await queue.cancel(held.entries[0], on: machine.id)
        #expect(fake.callCount("cancelHeldJob") == 1)
        #expect(fake.callCount("cancelJob") == 0)
        fake.queueListing = try MoldJSON.decoder.decode(QueueListing.self, from: Data(#"{"entries":[{"id":"h","state":"running"}]}"#.utf8))
        await queue.poll(machine.id)
        await queue.cancel(held.entries[0], on: machine.id)
        #expect(fake.callCount("cancelHeldJob") == 1)
        #expect(fake.callCount("cancelJob") == 0)
    }

    @Test func stalePauseResumeAndRetryDoNotMutateAChangedRow() async {
        let machine = MoldHost(name: "workstation", baseURL: URL(string: "http://workstation")!)
        let fake = FakeBackend(host: machine)
        fake.serverStatus = FakeFixtures.serverStatus(instanceId: "run-1")
        fake.capabilityBlock = FakeFixtures.capabilities(canPauseJob: true)
        let hosts = HostStore(hosts: [machine]) { _ in fake }
        await hosts.refresh(machine)
        let queue = QueueStore(hosts: hosts)
        queue.byHost[machine.id] = [FakeFixtures.queueEntry("job", state: "running")]
        await queue.pause(FakeFixtures.queueEntry("job"), on: machine.id)
        await queue.resume(FakeFixtures.queueEntry("job", state: "paused"), on: machine.id)
        await queue.retry(FakeFixtures.queueEntry("job", state: "held", batchId: "b", clientBatchId: "c", batchIndex: 0), on: machine.id)
        #expect(fake.callCount("pauseJob") == 0)
        #expect(fake.callCount("resumeJob") == 0)
        #expect(fake.callCount("retryJob") == 0)
    }

    @Test func rowActionsFollowLiveAuthorityWithoutMetadata() async {
        let machine = MoldHost(name: "workstation", baseURL: URL(string: "http://workstation")!)
        let fake = FakeBackend(host: machine)
        let hosts = HostStore(hosts: [machine]) { _ in fake }
        let queue = QueueStore(hosts: hosts)
        let held = FakeFixtures.queueEntry("h", state: "held", batchId: "b", clientBatchId: "c", retryable: true)
        queue.byHost[machine.id] = [held]
        #expect(queue.actions(for: held, on: machine.id) == QueueRowActions())
        hosts.reachability[machine.id] = .up(FakeFixtures.serverStatus(instanceId: "run-1"))
        #expect(queue.actions(for: held, on: machine.id).retry)
        #expect(queue.actions(for: held, on: machine.id).cancel)
        #expect(queue.canTransfer(held, on: machine.id))
        queue.acting[machine.id] = [held.id]
        #expect(queue.actions(for: held, on: machine.id) == QueueRowActions())
        #expect(!queue.canTransfer(held, on: machine.id))
        queue.acting[machine.id] = []
        queue.byHost[machine.id] = [FakeFixtures.queueEntry("h", state: "held", retryable: true)]
        #expect(!queue.actions(for: held, on: machine.id).retry)
        #expect(!queue.canTransfer(held, on: machine.id))
        #expect(queue.actions(for: held, on: machine.id).cancel)
    }

    @Test func actionMatrixHonorsCapabilitiesAndTerminalStates() {
        let machine = MoldHost(name: "workstation", baseURL: URL(string: "http://workstation")!)
        let hosts = HostStore(hosts: [machine])
        hosts.reachability[machine.id] = .up(FakeFixtures.serverStatus(instanceId: "run-1"))
        let queue = QueueStore(hosts: hosts)
        for caps in [nil, FakeFixtures.capabilities(), FakeFixtures.capabilities(canPauseJob: true, cooperativeCancellation: true)] {
            hosts.capabilities[machine.id] = caps
            for state in ["queued", "paused", "running", "failed", "complete", "cancelled", "cancelling", "unknown"] {
                let row = FakeFixtures.queueEntry("j", state: state)
                queue.byHost[machine.id] = [row]
                let actions = queue.actions(for: row, on: machine.id)
                #expect(actions.pause == (state == "queued" && caps?.canPauseOneJob == true))
                #expect(actions.resume == (state == "paused" && caps?.canPauseOneJob == true))
                #expect(actions.cancel == (["queued", "paused"].contains(state) || (state == "running" && caps?.canCancelRunningJob == true)))
                #expect(!actions.retry)
            }
        }
    }

    @Test func duplicateCancellationIsReservedThroughRefresh() async {
        let machine = MoldHost(name: "workstation", baseURL: URL(string: "http://workstation")!)
        let fake = FakeBackend(host: machine)
        let hosts = HostStore(hosts: [machine]) { _ in fake }
        hosts.reachability[machine.id] = .up(FakeFixtures.serverStatus(instanceId: "run-1"))
        let queue = QueueStore(hosts: hosts)
        let row = FakeFixtures.queueEntry("h", state: "held")
        queue.byHost[machine.id] = [row]
        fake.queueListing = FakeFixtures.queueListing(entries: [row])
        fake.delays["queue"] = .milliseconds(100)
        let first = Task { await queue.cancel(row, on: machine.id) }
        await fake.entered("queue")
        #expect(queue.isActing(row, on: machine.id))
        await queue.cancel(row, on: machine.id)
        #expect(fake.callCount("cancelHeldJob") == 1)
        await first.value
        #expect(!queue.isActing(row, on: machine.id))
    }

}
