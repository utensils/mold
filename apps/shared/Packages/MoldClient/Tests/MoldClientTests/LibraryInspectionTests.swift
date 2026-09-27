import Foundation
import Testing

@testable import MoldClient

// What the right-hand column is ABOUT, as one derivation.
//
// **Fails today**: there is no such derivation. `LibraryPane.swift:119` hands
// the inspector `showing.selected` and nothing else, and opening a print does
// not select it -- a double-click fires only the `count: 2` gesture, a tile's
// right-click Open acts on a print the click never selected, and `step`
// (`LibraryPane+Wiring.swift:40-45`) moves the viewer without touching the
// selection at all. So the inspector reads "Nothing selected" while a print
// fills the window, and the Library menu acts on whatever the grid still holds
// rather than on what you are looking at.

private let workstation = UUID()

private func entry(_ name: String, at seconds: UInt64 = 1_000) -> LibraryEntry {
    PrintFixtures.entry(name, host: workstation, timestamp: seconds)
}

private func showing(_ pool: [LibraryEntry], selecting: Set<PrintID> = []) -> LibraryShowing {
    LibraryShowing(pool: pool, query: LibraryQuery(), selection: selecting)
}

@Test func theOpenPrintIsWhatTheInspectorIsAboutEvenWithNothingSelected() {
    let open = entry("robot.png", at: 300)
    let other = entry("turtle.png", at: 200)

    let inspected = showing([open, other]).inspected(viewing: open.id)

    #expect(inspected.map(\.id) == [open.id])
}

/// A right-click Open acts on the clicked tile without selecting it, so the
/// two can genuinely disagree -- and the picture on screen is the answer.
@Test func theOpenPrintWinsOverASelectionMadeElsewhere() {
    let open = entry("robot.png", at: 300)
    let selected = entry("turtle.png", at: 200)

    let inspected = showing([open, selected], selecting: [selected.id])
        .inspected(viewing: open.id)

    #expect(inspected.map(\.id) == [open.id])
}

@Test func steppingToTheNextPrintCarriesTheInspectorWithIt() {
    let first = entry("robot.png", at: 300)
    let second = entry("turtle.png", at: 200)
    let list = showing([first, second], selecting: [first.id])

    #expect(list.inspected(viewing: first.id).map(\.id) == [first.id])
    #expect(list.inspected(viewing: second.id).map(\.id) == [second.id])
}

@Test func withTheViewerClosedItIsWhateverTheGridHasSelected() {
    let one = entry("robot.png", at: 300)
    let two = entry("turtle.png", at: 200)
    let three = entry("chair.png", at: 100)
    let list = showing([one, two, three], selecting: [one.id, three.id])

    #expect(list.inspected(viewing: nil).map(\.id) == list.selected.map(\.id))
    #expect(list.inspected(viewing: nil).count == 2)
}

/// The viewer itself falls back to the grid when the print it named is not in
/// the list any more (`LibraryPane.swift:137`), so the inspector has to fall
/// back at the same moment -- otherwise it would go on describing a print that
/// is no longer on screen.
@Test func aPrintTheQueryNoLongerShowsIsNotWhatTheInspectorIsAbout() {
    let shown = entry("robot.png", at: 300)
    let gone = entry("turtle.png", at: 200)
    let list = showing([shown], selecting: [shown.id])

    #expect(list.inspected(viewing: gone.id).map(\.id) == [shown.id])
}

@Test func nothingSelectedAndNothingOpenIsStillNothing() {
    #expect(showing([entry("robot.png")]).inspected(viewing: nil).isEmpty)
}
