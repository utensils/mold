import Foundation
import MoldClient
import Testing

@testable import Mold

/// `GenerateController.machineChoice` (M8 design, decision 2): `nil` is
/// Auto, and a real choice is persisted in the suite under
/// `generateMachine` the same way `HostStore.defaultMachine` is persisted
/// under `defaultMachine` (`HostStore+Default.swift`).
///
/// Reads and writes `AppStorageSuite.defaults` directly, the SHARED suite
/// every test in the run uses under `MOLD_NATIVE_FRESH` -- so, exactly like
/// `DefaultMachineTests`, every test here clears the key first.
@MainActor
struct GenerateMachineTests {
    private func machine(_ name: String = "plato") -> MoldHost {
        MoldHost(name: name, baseURL: URL(string: "http://\(name)")!)
    }

    private func reset() {
        AppStorageSuite.defaults.removeObject(forKey: "generateMachine")
    }

    private func controller() -> GenerateController {
        let hosts = HostStore(hosts: [])
        return GenerateController(hosts: hosts, defaults: ConfigStore(hosts: hosts))
    }

    @Test func aFreshControllerReadsAutoByDefault() {
        reset()
        #expect(controller().machineChoice == nil)
    }

    @Test func adoptingWithKeepingDraftPinsTheMachine() {
        reset()
        let plato = machine()
        let model = FakeFixtures.model("flux-dev:q8")
        let c = controller()

        c.adopt(model: model, on: plato.id, keepingDraft: true)

        #expect(c.machineChoice == plato.id)
    }

    @Test func settingItThenBuildingASecondControllerReadsItBack() {
        reset()
        let plato = machine()
        let first = controller()

        first.machineChoice = plato.id

        #expect(controller().machineChoice == plato.id)
    }

    @Test func settingNilClearsIt() {
        reset()
        let plato = machine()
        let c = controller()
        c.machineChoice = plato.id

        c.machineChoice = nil

        #expect(c.machineChoice == nil)
        #expect(AppStorageSuite.defaults.string(forKey: "generateMachine") == nil)
    }
}
