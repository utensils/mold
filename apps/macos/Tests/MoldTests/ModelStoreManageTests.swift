import Foundation
import MoldClient
import Testing

@testable import Mold

/// `ModelStore`'s mutations -- delete, load, unload, components -- and the
/// management listing a picker's `ready(on:)` deliberately does not answer.
@MainActor
struct ModelStoreManageTests {
    private func machine(_ name: String = "workstation") -> MoldHost {
        MoldHost(name: name, baseURL: URL(string: "http://\(name)")!)
    }

    /// **Fails today**: there is no `installed(on:)`.
    @Test func installedKeepsTheUpscalerThePickerDrops() async {
        let workstation = machine()
        let fake = FakeBackend(host: workstation)
        fake.modelRows = [
            FakeFixtures.model("flux-dev:q4", downloaded: true),
            FakeFixtures.model("real-esrgan-x4plus:fp16", family: "upscaler", downloaded: true),
        ]
        let hosts = HostStore(hosts: [workstation]) { _ in fake }
        let models = ModelStore(hosts: hosts)
        await models.refresh(on: workstation.id)

        #expect(models.ready(on: workstation.id).map(\.name) == ["flux-dev:q4"])
        #expect(Set(models.installed(on: workstation.id).map(\.name)) == ["flux-dev:q4", "real-esrgan-x4plus:fp16"])
    }

    @Test func aModelNeverFetchedIsNotCountedAsInstalled() async {
        let workstation = machine()
        let fake = FakeBackend(host: workstation)
        fake.modelRows = [FakeFixtures.model("flux-dev:bf16", downloaded: false)]
        let hosts = HostStore(hosts: [workstation]) { _ in fake }
        let models = ModelStore(hosts: hosts)
        await models.refresh(on: workstation.id)

        #expect(models.installed(on: workstation.id).isEmpty)
        #expect(models.all(on: workstation.id).count == 1)
    }

    @Test func aDeleteTheMachineRefusesBecauseTheModelIsLoadedIsReportedInItsWords() async {
        let workstation = machine()
        let fake = FakeBackend(host: workstation)
        let model = FakeFixtures.model("flux-dev:q4", downloaded: true)
        fake.plantedErrors["deleteModel"] = MoldClientError.http(
            status: 409, code: "MODEL_LOADED", message: "Unload flux-dev:q4 first.")
        let hosts = HostStore(hosts: [workstation]) { _ in fake }
        let models = ModelStore(hosts: hosts)

        let removal = await models.delete(model, on: workstation.id)

        #expect(removal == nil)
        let failure = hosts.failures.first { $0.host == workstation.id && $0.verb == "delete flux-dev:q4" }
        #expect(failure?.sentence.localizedCaseInsensitiveContains("unload flux-dev:q4 first") == true)
    }

    /// A second machine's own listing must not be touched by a delete on the
    /// first.
    @Test func deletingAModelReReadsThatMachineAndNoOther() async {
        let workstation = machine("workstation")
        let hal = machine("hal9000")
        let workstationFake = FakeBackend(host: workstation)
        workstationFake.modelRows = [FakeFixtures.model("flux-dev:q4", downloaded: true)]
        workstationFake.removalAnswers["flux-dev:q4"] = FakeFixtures.modelRemoval(removed: ["flux-dev:q4"])
        let halFake = FakeBackend(host: hal)
        halFake.modelRows = [FakeFixtures.model("sd15:fp16", downloaded: true)]
        let hosts = HostStore(hosts: [workstation, hal]) { host in host.name == "workstation" ? workstationFake : halFake }
        let models = ModelStore(hosts: hosts)
        await models.refresh(on: workstation.id)
        await models.refresh(on: hal.id)

        let removal = await models.delete(FakeFixtures.model("flux-dev:q4", downloaded: true), on: workstation.id)

        #expect(removal?.removed == ["flux-dev:q4"])
        #expect(workstationFake.callCount("models") == 2) // initial load + the post-delete refresh
        #expect(halFake.callCount("models") == 1) // untouched
    }

    @Test func loadAndUnloadEachRefreshTheHostAndClearBusyOnCompletion() async {
        let workstation = machine()
        let fake = FakeBackend(host: workstation)
        let model = FakeFixtures.model("flux-dev:q4", downloaded: true)
        fake.modelRows = [model]
        let hosts = HostStore(hosts: [workstation]) { _ in fake }
        let models = ModelStore(hosts: hosts)

        await models.load(model, gpu: nil, on: workstation.id)
        #expect(fake.loadedModels.map { $0.model } == ["flux-dev:q4"])
        #expect(fake.callCount("models") == 1)
        #expect(models.isBusy(with: model, on: workstation.id) == false)

        await models.unload(model, on: workstation.id)
        #expect(fake.unloadedModels.map { $0.model } == ["flux-dev:q4"])
        #expect(fake.callCount("models") == 2)
        #expect(models.isBusy(with: model, on: workstation.id) == false)
    }

    /// **Fails today**: `components(of:on:)` does not exist.
    @Test func componentsAreCachedUntilThatHostRefreshes() async {
        let workstation = machine()
        let fake = FakeBackend(host: workstation)
        let model = FakeFixtures.model("flux-dev:q4", downloaded: true)
        fake.modelRows = [model]
        fake.componentRows["flux-dev:q4"] = FakeFixtures.modelComponents(
            "flux-dev:q4", rows: [(kind: "transformer", name: "model.safetensors", present: true)])
        let hosts = HostStore(hosts: [workstation]) { _ in fake }
        let models = ModelStore(hosts: hosts)

        _ = await models.components(of: model, on: workstation.id)
        _ = await models.components(of: model, on: workstation.id)
        #expect(fake.callCount("modelComponents") == 1)

        await models.refresh(on: workstation.id)
        _ = await models.components(of: model, on: workstation.id)
        #expect(fake.callCount("modelComponents") == 2)
    }
}
