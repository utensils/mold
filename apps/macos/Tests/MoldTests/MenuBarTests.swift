import Foundation
import MoldClient
import Testing

@testable import Mold

/// The two pure resolvers behind the Machine menu and View ▸ Larger/Smaller
/// Thumbnails (design M7 S6) -- pinned with no view rendered, the same way
/// `QueueSelection.offeredTitles` is.
@MainActor
struct MenuBarTests {
    private func machine(_ name: String) -> MoldHost {
        MoldHost(name: name, baseURL: URL(string: "http://\(name)")!)
    }

    @Test func everyMachineAppearsInTheMachineMenuWithTheDefaultTicked() {
        let workstation = machine("workstation")
        let hal9000 = machine("hal9000")
        let selection = MachineSelection(
            machines: [workstation, hal9000], selected: workstation.id, defaultID: hal9000.id,
            offered: MachineCardActions.offered(isThisMac: false, isDefault: false),
            choose: { _ in }, perform: { _ in })

        #expect(selection.rows == [
            .init(name: "workstation", isDefault: false),
            .init(name: "hal9000", isDefault: true),
        ])
    }

    @Test func noMachineIsTickedWhenNothingHasBeenChosenYet() {
        let workstation = machine("workstation")
        let selection = MachineSelection(
            machines: [workstation], selected: workstation.id, defaultID: nil,
            offered: MachineCardActions.offered(isThisMac: false, isDefault: false),
            choose: { _ in }, perform: { _ in })

        #expect(selection.rows == [.init(name: "workstation", isDefault: false)])
    }

    @Test func theThumbnailStepsStopAtTheSlidersOwnEnds() {
        #expect(ThumbnailStep.apply(88, delta: -24) == 88)
        #expect(ThumbnailStep.apply(260, delta: 24) == 260)
        #expect(ThumbnailStep.apply(132, delta: 24) == 156)
        #expect(ThumbnailStep.apply(132, delta: -24) == 108)
    }

    @Test func sameActionModelsStillReplaceTheirMenuClosures() {
        let host = UUID()
        let items = [RowAction(kind: ModelActions.Kind.load, title: "Load")]
        let first = ModelSelection(
            target: .init(host: host, model: "first:q4"), items: items, perform: { _ in })
        let second = ModelSelection(
            target: .init(host: host, model: "second:q4"), items: items, perform: { _ in })

        #expect(first != second)
    }
}
