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
        let machine = MoldHost(name: "plato", baseURL: URL(string: "http://plato")!)
        let fake = FakeBackend(host: machine)
        fake.refuses = ["cancelJob"]
        let hosts = HostStore(hosts: [machine]) { _ in fake }
        let queue = QueueStore(hosts: hosts)
        let entry = FakeFixtures.queueEntry("job-1")

        await queue.cancel(entry, on: machine.id)

        #expect(fake.calls.contains("cancelJob"))
        #expect(hosts.failures.contains { $0.host == machine.id })
    }

    /// **Fails today**: `refresh`'s `?? []` writes an empty array for a
    /// machine whose fetch failed, so a transient hiccup blanks rows that
    /// were showing a second ago.
    @Test func aMachineThatCannotListItsQueueKeepsTheRowsItLastShowed() async {
        let machine = MoldHost(name: "plato", baseURL: URL(string: "http://plato")!)
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
}
