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

    /// **Fails today**: `drain` holds `draining` until its whole task ends --
    /// including the trailing `relist` and `reloadCollections`, both full round
    /// trips. An edit enqueued in that window calls `drain`, hits the guard and
    /// returns, and the running task never looks at the outbox again: the star
    /// sits on screen forever and never reaches the machine.
    @Test func anEditMadeDuringTheTrailingRelistStillReachesTheMachine() async {
        let machine = host("plato")
        let fake = FakeBackend(host: machine)
        fake.prints = [FakeFixtures.print("star.png"), FakeFixtures.print("second.png")]
        let hosts = HostStore(hosts: [machine]) { _ in fake }
        let library = LibraryStore(hosts: hosts)
        let rows = [FakeFixtures.print("star.png"), FakeFixtures.print("second.png")]
            .map { LibraryEntry(host: machine, print: $0) }
        library.perHost[machine.id] = rows

        // The trailing re-list is held open and the second star is made while
        // it is in flight -- the only moment that reproduces this.
        fake.delays["gallery"] = .milliseconds(200)
        library.setFavorite(true, on: [rows[0]])
        hosts.listeners.forEach {
            $0(machine.id, .gallery(.updated(filename: "star.png", row: nil)))
        }
        await settle(until: { fake.calls.contains("gallery") })
        library.setFavorite(true, on: [rows[1]])

        await settle(until: { fake.calls.filter { $0 == "mutate" }.count == 2 })
        #expect(fake.calls.filter { $0 == "mutate" }.count == 2)
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

        // The mutation is held open and the frame is delivered while it is in
        // flight, so the frame provably arrives while the chain is pending --
        // asserting one `gallery` call alone could not tell the
        // skip-then-repair from a drain that had already finished and simply
        // took the frame's own `row: nil` re-list.
        fake.delays["mutate"] = .milliseconds(200)
        library.setFavorite(true, on: library.perHost[machine.id] ?? [])
        await settle(until: { fake.calls.contains("mutate") })
        hosts.listeners.forEach {
            $0(machine.id, .gallery(.updated(filename: "star.png", row: nil)))
        }

        await settle(until: { fake.calls.contains("gallery") })
        // The frame was SKIPPED (no re-list while the chain ran) and the
        // machine was read exactly once, after the mutation settled.
        #expect(fake.calls == ["mutate", "gallery"])
    }
}
