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
    private func machine(_ name: String = "workstation") -> MoldHost {
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
        let workstation = machine()
        let model = FakeFixtures.model("flux-dev:q8")
        let c = controller()

        c.adopt(model: model, on: workstation.id, keepingDraft: true)

        #expect(c.machineChoice == workstation.id)
    }

    @Test func settingItThenBuildingASecondControllerReadsItBack() {
        reset()
        let workstation = machine()
        let first = controller()

        first.machineChoice = workstation.id

        #expect(controller().machineChoice == workstation.id)
    }

    @Test func settingNilClearsIt() {
        reset()
        let workstation = machine()
        let c = controller()
        c.machineChoice = workstation.id

        c.machineChoice = nil

        #expect(c.machineChoice == nil)
        #expect(AppStorageSuite.defaults.string(forKey: "generateMachine") == nil)
    }
}
