import Foundation
import MoldClient
import Testing

@testable import Mold

/// `MachineStore` is one store for devices, telemetry and peers -- all "what
/// this machine is", fetched per host through `HostStore`. These pin its four
/// decisions: a failed listing keeps what it last showed, a lifecycle change
/// shows what the machine answers rather than what was asked, an invalidation
/// frame refetches, and at most one telemetry stream exists at a time.
@MainActor
struct MachineStoreTests {
    private func machine(_ name: String = "plato") -> MoldHost {
        MoldHost(name: name, baseURL: URL(string: "http://\(name)")!)
    }

    @Test func aMachineThatCannotListItsGpusReportsItAndKeepsWhatItShowed() async {
        let plato = machine()
        let fake = FakeBackend(host: plato)
        fake.deviceState = FakeFixtures.deviceState()
        let hosts = HostStore(hosts: [plato]) { _ in fake }
        let machines = MachineStore(hosts: hosts)

        await machines.refreshDevices(on: plato.id)
        #expect(machines.devices(on: plato.id).map(\.id) == ["cuda:0"])

        fake.refuses = ["devices"]
        await machines.refreshDevices(on: plato.id)

        #expect(machines.devices(on: plato.id).map(\.id) == ["cuda:0"])
        #expect(hosts.failures.contains { $0.host == plato.id && $0.verb == "list its GPUs" })
    }

    @Test func flippingASwitchShowsWhatTheMachineSaysAndNotWhatWasAsked() async {
        let plato = machine()
        let fake = FakeBackend(host: plato)
        let device = FakeFixtures.deviceInfo("cuda:0", ordinal: 0)
        fake.deviceState = FakeFixtures.deviceState([device])
        let hosts = HostStore(hosts: [plato]) { _ in fake }
        let machines = MachineStore(hosts: hosts)
        await machines.refreshDevices(on: plato.id)

        // The machine answers a drain in progress, not "off" -- the row must
        // show that, not the `enabled: false` the person asked for.
        fake.setDeviceAnswer = FakeFixtures.deviceInfo(
            "cuda:0", ordinal: 0, adminState: "draining", desiredEnabled: false)

        await machines.setDevice(device, enabled: false, on: plato.id)

        let row = machines.devices(on: plato.id)[0]
        #expect(row.adminState == .draining)
        #expect(row.desiredEnabled == false)
        #expect(machines.isChanging(device) == false)
    }

    @Test func aRefusedLifecycleChangeLeavesTheRowAloneAndSaysSo() async {
        let plato = machine()
        let fake = FakeBackend(host: plato)
        fake.deviceState = FakeFixtures.deviceState()
        let hosts = HostStore(hosts: [plato]) { _ in fake }
        let machines = MachineStore(hosts: hosts)
        await machines.refreshDevices(on: plato.id)
        let before = machines.devices(on: plato.id)[0]

        fake.refuses = ["setDevice"]
        await machines.setDevice(before, enabled: false, on: plato.id)

        #expect(machines.devices(on: plato.id)[0] == before)
        #expect(hosts.failures.contains { $0.host == plato.id && $0.verb == "change that GPU" })
        #expect(machines.isChanging(before) == false)
    }

    @Test func aDeviceStateChangedFrameMakesTheStoreGoAndRead() async {
        let plato = machine()
        let fake = FakeBackend(host: plato)
        fake.deviceState = FakeFixtures.deviceState()
        let hosts = HostStore(hosts: [plato]) { _ in fake }
        let machines = MachineStore(hosts: hosts)

        await machines.refreshDevices(on: plato.id)
        #expect(fake.callCount("devices") == 1)

        hosts.listeners.forEach { $0(plato.id, .deviceStateChanged) }
        await settle { fake.callCount("devices") == 2 }

        #expect(fake.callCount("devices") == 2)
    }

    @Test func watchingASecondMachineStopsWatchingTheFirst() async {
        let a = machine("a")
        let b = machine("b")
        let fakeA = FakeBackend(host: a)
        let fakeB = FakeBackend(host: b)
        let fakes: [MoldHost.ID: FakeBackend] = [a.id: fakeA, b.id: fakeB]
        let hosts = HostStore(hosts: [a, b]) { fakes[$0.id]! }
        let machines = MachineStore(hosts: hosts)

        machines.watchResources(on: a.id)
        await settle { fakeA.callCount("resourceStream") == 1 }

        machines.watchResources(on: b.id)
        await settle { fakeB.callCount("resourceStream") == 1 }

        await settle { fakeA.resourceStreamEnded }
        #expect(fakeA.resourceStreamEnded)
        #expect(!fakeB.resourceStreamEnded)
    }

    @Test func refreshPeersFillsInWhatThatMachineFoundNearby() async {
        let plato = machine()
        let fake = FakeBackend(host: plato)
        fake.peerRows = [FakeFixtures.discoveryPeer("bender", url: "http://bender:7680")]
        let hosts = HostStore(hosts: [plato]) { _ in fake }
        let machines = MachineStore(hosts: hosts)

        await machines.refreshPeers(on: plato.id)

        #expect(machines.peers[plato.id]?.map(\.name) == ["bender"])
    }

    @Test func aSnapshotFillsInTheLiveFiguresForTheRightCard() async {
        let plato = machine()
        let fake = FakeBackend(host: plato)
        let device0 = FakeFixtures.deviceInfo("cuda:0", ordinal: 0)
        let device1 = FakeFixtures.deviceInfo("cuda:1", ordinal: 1)
        fake.deviceState = FakeFixtures.deviceState([device0, device1])
        let hosts = HostStore(hosts: [plato]) { _ in fake }
        let machines = MachineStore(hosts: hosts)
        await machines.refreshDevices(on: plato.id)

        machines.watchResources(on: plato.id)
        await settle { fake.callCount("resourceStream") == 1 }

        fake.resourceStreamContinuation?.yield(FakeFixtures.resourceSnapshot([(ordinal: 1, vramUsed: 555)]))
        await settle { machines.sample(for: device1, on: plato.id) != nil }

        #expect(machines.sample(for: device1, on: plato.id)?.vramUsed == 555)
        #expect(machines.sample(for: device0, on: plato.id) == nil)
    }
}
