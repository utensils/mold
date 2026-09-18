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
        let machine = MoldHost(name: "workstation", baseURL: URL(string: "http://workstation")!)
        let hosts = HostStore(hosts: [machine])
        let models = ModelStore(hosts: hosts)

        #expect(models.hasLoaded(on: machine.id) == false)
    }

    @Test func refreshingOneHostLoadsOnlyThatHostsModels() async {
        let workstation = MoldHost(name: "workstation", baseURL: URL(string: "http://workstation")!)
        let fake = FakeBackend(host: workstation)
        fake.modelRows = [FakeFixtures.model("flux-dev:q4")]
        let hosts = HostStore(hosts: [workstation]) { _ in fake }
        let models = ModelStore(hosts: hosts)

        await models.refresh(on: workstation.id)

        #expect(models.hasLoaded(on: workstation.id) == true)
        #expect(models.model(named: "flux-dev:q4", on: workstation.id) != nil)
    }

    @Test func aRefusedListingIsReportedAndStillNotLoaded() async {
        let workstation = MoldHost(name: "workstation", baseURL: URL(string: "http://workstation")!)
        let fake = FakeBackend(host: workstation)
        fake.refuses = ["models"]
        let hosts = HostStore(hosts: [workstation]) { _ in fake }
        let models = ModelStore(hosts: hosts)

        await models.refresh(on: workstation.id)

        #expect(models.hasLoaded(on: workstation.id) == false)
        #expect(hosts.failures.contains { $0.host == workstation.id && $0.verb == "list its models" })
    }
}
