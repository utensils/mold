import Foundation
import MoldClient
import Testing

@testable import Mold

/// What a machine says while this app is sending it something.
///
/// **Fails today**: `LibraryStore+Live.swift:38` drops EVERY gallery frame
/// from a machine with anything queued for it, and nothing re-lists once the
/// chain empties -- so a print landing during a star is invisible until ⌘R.
@MainActor
struct LibraryStoreLiveTests {
    private func host(_ name: String) -> MoldHost {
        MoldHost(name: name, baseURL: URL(string: "http://\(name)")!)
    }

    @Test func aPrintLandingDuringOneOfOurEditsStillAppears() async {
        let machine = host("plato")
        let fake = FakeBackend(host: machine)
        let hosts = HostStore(hosts: [machine]) { _ in fake }
        let library = LibraryStore(hosts: hosts)
        library.perHost[machine.id] = [LibraryEntry(host: machine,
                                                    print: FakeFixtures.print("star.png"))]

        // A star on its way to the machine, and a render landing in the window.
        library.outbox.enqueue(PrintEdit(change: .favorite(true),
                                         targets: [machine.id: ["star.png"]]))
        hosts.listeners.forEach {
            $0(machine.id, .gallery(.added(filename: "new.png",
                                           row: FakeFixtures.print("new.png"))))
        }

        #expect(library.perHost[machine.id]?.map(\.print.filename).sorted()
            == ["new.png", "star.png"])
    }

    /// The echo itself is still skipped: the star is already on screen, and
    /// re-applying it would be a wasted round trip.
    @Test func theMachineEchoingOurOwnEditBackIsStillSkipped() async {
        let machine = host("plato")
        let fake = FakeBackend(host: machine)
        let hosts = HostStore(hosts: [machine]) { _ in fake }
        let library = LibraryStore(hosts: hosts)
        library.perHost[machine.id] = [LibraryEntry(host: machine,
                                                    print: FakeFixtures.print("star.png"))]
        library.outbox.enqueue(PrintEdit(change: .favorite(true),
                                         targets: [machine.id: ["star.png"]]))

        // `row: nil` means "go and read" -- which, taken, would be a re-list.
        hosts.listeners.forEach {
            $0(machine.id, .gallery(.updated(filename: "star.png", row: nil)))
        }
        await settle(until: { fake.calls.contains("gallery") })

        #expect(fake.callCount("gallery") == 0)
    }

    /// A frame skipped as our own echo may ALSO have been another client
    /// editing that row. Once our chain is settled, that is the only thing it
    /// could still have been saying, so the machine is read again.
    @Test func aMachineWhoseFrameWasSkippedIsReadAgainOnceTheChainDrains() async throws {
        let machine = host("plato")
        let fake = FakeBackend(host: machine)
        fake.prints = [FakeFixtures.print("star.png")]
        let hosts = HostStore(hosts: [machine]) { _ in fake }
        let library = LibraryStore(hosts: hosts)
        library.perHost[machine.id] = [LibraryEntry(host: machine,
                                                    print: FakeFixtures.print("star.png"))]

        library.setFavorite(true, on: library.perHost[machine.id] ?? [])
        hosts.listeners.forEach {
            $0(machine.id, .gallery(.updated(filename: "star.png", row: nil)))
        }

        await settle(until: { fake.calls.contains("gallery") })
        #expect(fake.calls.contains("mutate"))
        #expect(fake.callCount("gallery") == 1)
    }
}
