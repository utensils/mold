import Foundation
import MoldClient
import Testing

@testable import Mold

/// What a burst of `resync_required` markers costs.
///
/// **Fails today**: `LibraryStore+Live.swift:25` fires a bare
/// `Task { await relist(host) }` per marker, so K markers are K concurrent
/// reads of one machine's whole index, each assigning `perHost` wholesale in
/// whatever order they return.
@MainActor
struct RelistGateTests {
    private func host(_ name: String) -> MoldHost {
        MoldHost(name: name, baseURL: URL(string: "http://\(name)")!)
    }

    /// Five markers in one burst: one read, and ONE more behind it.
    @Test func aBurstOfMarkersIsOneReadAndAtMostOneMore() async {
        let machine = host("workstation")
        let fake = FakeBackend(host: machine)
        fake.prints = [FakeFixtures.print("a.png")]
        fake.delays["gallery"] = .milliseconds(150)
        let hosts = HostStore(hosts: [machine]) { _ in fake }
        let library = LibraryStore(hosts: hosts)

        for _ in 0 ..< 5 {
            hosts.listeners.forEach { $0(machine.id, .resyncRequired) }
        }

        await settle(until: { library.relists.reads[machine.id] == 2 })
        #expect(library.relists.reads[machine.id] == 2)
        #expect(fake.callCount("gallery") <= 2)
    }

    /// The queued read STARTS after the last marker, which is the only thing
    /// "current" can mean: a marker delivered while the first read is in
    /// flight is answered by a read begun afterwards.
    @Test func theLastReadStartsAfterTheLastMarker() async {
        let machine = host("workstation")
        let fake = FakeBackend(host: machine)
        fake.prints = [FakeFixtures.print("a.png")]
        fake.delays["gallery"] = .milliseconds(150)
        let hosts = HostStore(hosts: [machine]) { _ in fake }
        let library = LibraryStore(hosts: hosts)

        hosts.listeners.forEach { $0(machine.id, .resyncRequired) }
        await settle(until: { library.relists.reads[machine.id] == 1 })
        // Mid-read: the answer in flight was asked for before this marker, so
        // it cannot be the repair the marker is asking for.
        hosts.listeners.forEach { $0(machine.id, .resyncRequired) }
        #expect(library.relists.reads[machine.id] == 1)

        await settle(until: { library.relists.reads[machine.id] == 2 })
        #expect(library.relists.reads[machine.id] == 2)
    }

    /// Markers with nothing in flight are each answered in turn -- the gate
    /// collapses a burst, it does not swallow a later one.
    @Test func aMarkerArrivingWhenTheGateIsIdleIsAlwaysRead() async {
        let machine = host("workstation")
        let fake = FakeBackend(host: machine)
        fake.prints = [FakeFixtures.print("a.png")]
        let hosts = HostStore(hosts: [machine]) { _ in fake }
        let library = LibraryStore(hosts: hosts)

        hosts.listeners.forEach { $0(machine.id, .resyncRequired) }
        await settle(until: { library.relists.reads[machine.id] == 1 })
        hosts.listeners.forEach { $0(machine.id, .resyncRequired) }
        await settle(until: { library.relists.reads[machine.id] == 2 })

        #expect(library.relists.reads[machine.id] == 2)
    }

    /// Two machines are two gates' worth of work, not one queue.
    @Test func oneMachinesReadDoesNotHoldAnothersUp() async {
        let workstation = host("workstation"), hal = host("hal9000")
        let fakes = [workstation.id: FakeBackend(host: workstation), hal.id: FakeBackend(host: hal)]
        fakes.values.forEach {
            $0.prints = [FakeFixtures.print("a.png")]
            $0.delays["gallery"] = .milliseconds(100)
        }
        let hosts = HostStore(hosts: [workstation, hal]) { fakes[$0.id]! }
        let library = LibraryStore(hosts: hosts)

        hosts.listeners.forEach { $0(workstation.id, .resyncRequired) }
        hosts.listeners.forEach { $0(hal.id, .resyncRequired) }

        await settle(until: { library.relists.reads.count == 2 })
        #expect(library.relists.reads[workstation.id] == 1)
        #expect(library.relists.reads[hal.id] == 1)
    }
}
