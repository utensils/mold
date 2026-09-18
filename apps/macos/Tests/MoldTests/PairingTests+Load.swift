import Foundation
import MoldClient
import Testing

@testable import Mold

/// `PairingStore.load(on:fixture:)` -- the one load the Machines pane
/// calls. It exists because the section could not load itself: its empty
/// state is `EmptyView`, and a `.task` hung off an `EmptyView` never runs,
/// which left the section dead on every machine until M7 UAT read the page.
extension PairingTests {
    @Test func loadingWithAFixtureSeedsAndAsksTheMachineNothing() async {
        let workstation = machine()
        let fake = FakeBackend(host: workstation)
        let hosts = HostStore(hosts: [workstation]) { _ in fake }
        let store = PairingStore(hosts: hosts)
        let fixture = PairingStore.Fixture(hosts: [
            "workstation": PairingStore.HostFixture(
                clients: PairedClients(authRequired: true, pairingAvailable: true, clients: []))
        ])

        await store.load(on: workstation.id, fixture: fixture)

        #expect(store.isSeeded)
        #expect(store.byHost[workstation.id]?.authRequired == true)
        #expect(fake.calls.isEmpty)
    }

    @Test func loadingWithoutAFixtureAsksTheMachineOnce() async {
        let workstation = machine()
        let fake = FakeBackend(host: workstation)
        fake.pairedClientsAnswer = PairedClients(authRequired: false, pairingAvailable: true, clients: [])
        let hosts = HostStore(hosts: [workstation]) { _ in fake }
        let store = PairingStore(hosts: hosts)

        await store.load(on: workstation.id, fixture: nil)
        await store.load(on: workstation.id, fixture: nil)

        #expect(!store.isSeeded)
        #expect(store.byHost[workstation.id]?.authRequired == false)
        #expect(fake.callCount("pairedClients") == 2)
    }
}
