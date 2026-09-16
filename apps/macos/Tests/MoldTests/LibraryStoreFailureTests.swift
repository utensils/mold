import Foundation
import MoldClient
import Testing

@testable import Mold

/// Pins three defects the one-failure-policy slice replaces: a rollback that
/// reached across machines, and a tag delete that swallowed a real refusal.
/// `QueueStoreTests` carries the third -- a refresh that blanked a machine's
/// rows instead of keeping the ones it last showed.
@MainActor
struct LibraryStoreFailureTests {
    private func host(_ name: String) -> MoldHost {
        MoldHost(name: name, baseURL: URL(string: "http://\(name)")!)
    }

    /// **Fails today**: `moveToTrash` snapshots ALL of `perHost` before the
    /// await and restores the whole thing on any one machine's refusal, so a
    /// machine that succeeded loses the rows it just trashed too.
    @Test func oneMachineRefusingTheTrashDoesNotResurrectAnotherMachinesRows() async {
        let willing = host("willing")
        let refusing = host("refusing")
        let willingBackend = FakeBackend(host: willing)
        let refusingBackend = FakeBackend(host: refusing)
        refusingBackend.refuses = ["trash"]
        let fakes: [MoldHost.ID: FakeBackend] = [willing.id: willingBackend, refusing.id: refusingBackend]
        let hosts = HostStore(hosts: [willing, refusing]) { fakes[$0.id]! }
        let library = LibraryStore(hosts: hosts)

        let willingEntry = LibraryEntry(host: willing, print: FakeFixtures.print("a.png"))
        let refusingEntry = LibraryEntry(host: refusing, print: FakeFixtures.print("b.png"))
        library.perHost[willing.id] = [willingEntry]
        library.perHost[refusing.id] = [refusingEntry]
        library.rebuild()

        await library.moveToTrash([willingEntry, refusingEntry])

        // The willing machine's row stays trashed even though the other
        // machine refused.
        #expect(library.perHost[willing.id]?.isEmpty == true)
        // The refusing machine's own row is restored -- and ONLY its own.
        #expect(library.perHost[refusing.id]?.map(\.print.filename) == ["b.png"])
        #expect(hosts.failures.contains { $0.host == refusing.id })
    }

    /// **Fails today**: `deleteTag` sends with `try?` and says nothing when a
    /// machine refuses.
    @Test func deletingATagAMachineRefusesIsReported() async throws {
        let machine = host("plato")
        let fake = FakeBackend(host: machine)
        fake.refuses = ["deleteTag"]
        let hosts = HostStore(hosts: [machine]) { _ in fake }
        let library = LibraryStore(hosts: hosts)

        library.deleteTag("owls")
        try await waitUntil { fake.calls.contains("deleteTag") }
        try await waitUntil { !hosts.failures.isEmpty }

        #expect(hosts.failures.contains { $0.host == machine.id && $0.verb == "delete the tag “owls”" })
    }
}
