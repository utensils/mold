import Foundation
import MoldClient
import Testing

@testable import Mold

/// `MachineControl.rows` is the menu behind the capsule's new Machine
/// control (M8 design, decision 2): Auto plus every up machine that
/// generates, with the CHOSEN machine always listed so the choice stays
/// visible and changeable even while it can't be reached.
@MainActor
struct MachineControlTests {
    private func host(_ name: String) -> MoldHost {
        MoldHost(name: name, baseURL: URL(string: "http://\(name)")!)
    }

    @Test func aDownMachineIsLeftOut() {
        let workstation = host("workstation")
        let rows = MachineControl.rows(
            hosts: [workstation], chosen: nil, preferred: nil,
            isUp: { _ in false }, generates: { _ in true }
        )
        #expect(rows.machines.isEmpty)
    }

    @Test func aMachineThatIsUpButDoesNotGenerateIsLeftOut() {
        let workstation = host("workstation")
        let rows = MachineControl.rows(
            hosts: [workstation], chosen: nil, preferred: nil,
            isUp: { _ in true }, generates: { _ in false }
        )
        #expect(rows.machines.isEmpty)
    }

    @Test func theChosenMachineIsListedEvenWhenDown() {
        let workstation = host("workstation")
        let rows = MachineControl.rows(
            hosts: [workstation], chosen: workstation.id, preferred: nil,
            isUp: { _ in false }, generates: { _ in false }
        )
        #expect(rows.machines.map(\.id) == [workstation.id])
        #expect(rows.machines.first?.caption == "workstation — can't be reached")
    }

    @Test func labelNamesAutosPreferredMachine() {
        let workstation = host("workstation")
        let rows = MachineControl.rows(
            hosts: [workstation], chosen: nil, preferred: workstation,
            isUp: { _ in true }, generates: { _ in true }
        )
        #expect(rows.label == "Auto · workstation")
    }

    @Test func labelIsPlainAutoWithNoPreferredMachine() {
        let rows = MachineControl.rows(
            hosts: [], chosen: nil, preferred: nil, isUp: { _ in false }, generates: { _ in false }
        )
        #expect(rows.label == "Auto")
    }

    @Test func labelNamesTheChosenMachine() {
        let workstation = host("workstation")
        let hal9000 = host("hal9000")
        let rows = MachineControl.rows(
            hosts: [workstation, hal9000], chosen: hal9000.id, preferred: workstation,
            isUp: { _ in true }, generates: { _ in true }
        )
        #expect(rows.label == "hal9000")
    }

    @Test func modelAfterKeepsTheSameNameWhenItIsReadyThere() {
        let flux = FakeFixtures.model("flux-dev:q8")
        let other = FakeFixtures.model("flux-schnell:q8")
        #expect(MachineControl.model(after: "flux-dev:q8", readyThere: [other, flux])?.name == "flux-dev:q8")
    }

    @Test func modelAfterFallsBackToTheFirstReadyModel() {
        let first = FakeFixtures.model("flux-schnell:q8")
        let second = FakeFixtures.model("flux-dev:q8")
        #expect(MachineControl.model(after: "sdxl:q8", readyThere: [first, second])?.name == "flux-schnell:q8")
    }

    @Test func modelAfterIsNilOnAnEmptyList() {
        #expect(MachineControl.model(after: "flux-dev:q8", readyThere: []) == nil)
    }
}
