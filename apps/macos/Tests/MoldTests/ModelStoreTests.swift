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

@Test func namedViewMeshModelsAreReachableInGeneratePicker() async throws {
    let machine = MoldHost(name: "Fixture", baseURL: URL(string: "http://fixture")!)
    let fake = FakeBackend(host: machine)
    let hosts = HostStore(hosts: [machine]) { _ in fake }
    let models = ModelStore(hosts: hosts)
    let model = try MoldJSON.decoder.decode(Model.self, from: Data("""
    {"name":"hunyuan3d-2mv:fp16","family":"hunyuan3d","description":"multiview",
     "downloaded":true,"runtime_available":true,
     "generation_profile":{"schema_version":1,"profile_id":"mesh","profile_hash":"h","default_recipe_id":"default","recipes":[
      {"id":"default","label":"Default","defaults":{"width":512,"height":512,"steps":5,"guidance":1},
       "resolution":{"domain":"dynamic","alignment":16,"min_width":256,"min_height":256},
       "steps":{"default":5,"min":1,"max":50,"step":1,"mode":"adjustable"},
       "guidance":{"default":1,"min":1,"max":1,"step":0.1,"mode":"fixed"},"capabilities":{"mesh":{"named_views":
       {"mode":"adjustable","roles":["front","left","right","back"],"min_count":1,"max_count":4}}}}]}}
    """.utf8))
    fake.modelRows = [model]
    await models.refresh(on: machine.id)
    #expect(models.ready(on: machine.id).map(\.name) == [model.name])
}
}
