import Foundation
import MoldClient
import Testing

@testable import Mold

/// `hasLoaded` is what tells the Machines page "None installed" from "we
/// haven't asked this host yet" -- a never-listed host has no key in
/// `byHost` at all, which is the fact `MachineFigures` reads.
@MainActor
struct ModelStoreTests {
    @Test func aHostThatHasNeverBeenListedHasNotLoaded() async {
        let machine = MoldHost(name: "plato", baseURL: URL(string: "http://plato")!)
        let hosts = HostStore(hosts: [machine])
        let models = ModelStore(hosts: hosts)

        #expect(models.hasLoaded(on: machine.id) == false)
    }

    @Test func refreshingOneHostLoadsOnlyThatHostsModels() async {
        let plato = MoldHost(name: "plato", baseURL: URL(string: "http://plato")!)
        let fake = FakeBackend(host: plato)
        fake.modelRows = [FakeFixtures.model("flux-dev:q4")]
        let hosts = HostStore(hosts: [plato]) { _ in fake }
        let models = ModelStore(hosts: hosts)

        await models.refresh(on: plato.id)

        #expect(models.hasLoaded(on: plato.id) == true)
        #expect(models.model(named: "flux-dev:q4", on: plato.id) != nil)
    }

    @Test func aRefusedListingIsReportedAndStillNotLoaded() async {
        let plato = MoldHost(name: "plato", baseURL: URL(string: "http://plato")!)
        let fake = FakeBackend(host: plato)
        fake.refuses = ["models"]
        let hosts = HostStore(hosts: [plato]) { _ in fake }
        let models = ModelStore(hosts: hosts)

        await models.refresh(on: plato.id)

        #expect(models.hasLoaded(on: plato.id) == false)
        #expect(hosts.failures.contains { $0.host == plato.id && $0.verb == "list its models" })
    }
}
