import Foundation
import MoldClient
import Testing
@testable import Mold

@MainActor
struct LibraryBulkTests {
    private func bench(_ count: Int) async -> (LibraryStore, FakeBackend, MoldHost) {
        let host = MoldHost(name: "bulk", baseURL: URL(string: "http://bulk")!)
        let fake = FakeBackend(host: host)
        fake.prints = (0..<count).map { FakeFixtures.print("print-\($0).png") }
        let hosts = HostStore(hosts: [host]) { _ in fake }
        let store = LibraryStore(hosts: hosts)
        await store.refresh()
        return (store, fake, host)
    }

    @Test func largeTrashUsesBoundedRequestsAndReconciles() async {
        let (store, fake, _) = await bench(35)
        await store.moveToTrash(store.items)
        #expect(fake.trashRequests.map(\.count) == [16, 16, 3])
        #expect(store.items.isEmpty)
        #expect(store.trashed.count == 35)
        #expect(store.bulkProgress == nil)
        #expect(store.bulkResult?.contains("35 of 35") == true)
    }

    @Test func statusIsVisibleWhileAwaitingAndStopFinishesCurrentBatch() async {
        let (store, fake, _) = await bench(35)
        fake.delays["trash"] = .milliseconds(150)
        let task = Task { await store.moveToTrash(store.items) }
        await fake.entered("trash")
        #expect(store.bulkRunning)
        #expect(store.bulkProgress?.contains("bulk") == true)
        store.bulkStopRequested = true
        // A second action cannot overlap the active one.
        await store.deleteForever(store.items)
        await task.value
        #expect(fake.trashRequests.count == 1)
        #expect(fake.callCount("deleteForever") == 0)
        #expect(store.items.count == 19)
        #expect(store.bulkResult?.contains("Stopped") == true)
    }

    @Test func stoppingDoesNotProbeUntouchedMachines() async {
        let (store, first, _) = await bench(35)
        let host = MoldHost(name: "untouched", baseURL: URL(string: "http://untouched")!)
        let second = FakeBackend(host: host)
        second.prints = [FakeFixtures.print("other.png")]
        let original = store.hosts.hosts[0]
        let hosts = HostStore(hosts: [original, host]) { $0.id == original.id ? first : second }
        let library = LibraryStore(hosts: hosts)
        await library.refresh()
        first.delays["trash"] = .milliseconds(150)
        let task = Task { await library.moveToTrash(library.items) }
        await first.entered("trash")
        library.bulkStopRequested = true
        await task.value
        #expect(second.callCount("trash") == 0)
        #expect(second.callCount("gallery") == 1)
        #expect(second.callCount("trashedPrints") == 0)
    }

    @Test func failedChunkKeepsAcknowledgedWorkAndReadsHostTruth() async {
        let (store, fake, host) = await bench(35)
        fake.trashFailureOnCall = 2
        fake.trashPartialFailureCount = 2
        await store.moveToTrash(store.items)
        #expect(store.items.count == 17)
        #expect(store.trashed.count == 18)
        #expect(fake.callCount("gallery") == 2)
        #expect(fake.callCount("trashedPrints") == 1)
        #expect(store.hosts.failures.contains { $0.host == host.id })
        #expect(store.bulkResult?.contains("could not be confirmed") == true)
    }

    @Test func removedHostCannotReceiveStaleResponseOrNextBatch() async {
        let (store, fake, host) = await bench(35)
        fake.delays["trash"] = .milliseconds(150)
        let task = Task { await store.moveToTrash(store.items) }
        await fake.entered("trash")
        store.hosts.hosts = []
        store.prune(to: [])
        await task.value
        #expect(fake.trashRequests.count == 1)
        #expect(store.perHost[host.id] == nil)
        #expect(store.trashPerHost[host.id] == nil)
    }


    @Test func replacedHostDiscardsOldReply() async {
        let (store, fake, host) = await bench(35)
        fake.delays["trash"] = .milliseconds(150)
        let task = Task { await store.moveToTrash(store.items) }
        await fake.entered("trash")
        var replacement = host
        replacement.baseURL = URL(string: "http://replacement")!
        store.hosts.hosts = [replacement]
        await task.value
        #expect(fake.trashRequests.count == 1)
        #expect(store.perHost[host.id]?.count == 35)
        #expect(fake.callCount("gallery") == 1)
    }

    @Test func ownBulkEventsDoNotRebuildPerPrintButUnrelatedEventsStillApply() async {
        let (store, _, host) = await bench(35)
        store.bulkTargets = Set(store.items.map(\.id))
        let revision = store.rows.value
        for entry in store.items {
            store.live.apply(.gallery(.trashed(filename: entry.print.filename)), from: host.id, in: store)
            store.live.apply(.gallery(.restored(filename: entry.print.filename, row: entry.print)),
                             from: host.id, in: store)
        }
        #expect(store.rows.value == revision)
        store.live.apply(.gallery(.added(filename: "new.png", row: FakeFixtures.print("new.png"))),
                         from: host.id, in: store)
        #expect(store.items.count == 36)
    }

    @Test func emptyTrashUsesTrashOnlyEndpoint() async {
        let (store, fake, _) = await bench(0)
        fake.trashedRows = (0..<35).map { FakeFixtures.print("trash-\($0).png") }
        await store.emptyTrash()
        #expect(fake.callCount("deleteForever") == 0)
        #expect(fake.callCount("emptyTrash") == 1)
        #expect(store.trashed.isEmpty)
    }
    @Test func emptyTrashNeverUsesStaleNamesToDeleteLivePrints() async {
        let (store, fake, host) = await bench(1)
        store.trashPerHost[host.id] = [LibraryEntry(host: host, print: fake.prints[0])]
        store.rebuildTrash()
        // The displayed trash row is stale: that print has been restored.
        fake.trashedRows = []
        await store.emptyTrash()
        #expect(fake.prints.count == 1)
        #expect(store.items.count == 1)
        #expect(fake.callCount("deleteForever") == 0)
    }

}
