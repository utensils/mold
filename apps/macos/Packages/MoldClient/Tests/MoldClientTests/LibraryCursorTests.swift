import Foundation
import Testing

@testable import MoldClient

private let host = UUID()

private func entry(_ n: Int, day: Int) -> LibraryEntry {
    // Day 0 is today; each further day is 24h earlier. Within a day, later n
    // means older, so the grid order is n ascending.
    let base = UInt64(Date.now.timeIntervalSince1970) - UInt64(day * 86_400)
    let meta = try! MoldJSON.decoder.decode(OutputMetadata.self, from: Data("{}".utf8))
    return LibraryEntry(hostID: host, hostName: "h", print: GalleryPrint(
        filename: "d\(day)-\(n).png", metadata: meta, timestamp: base - UInt64(n),
        format: "png", sizeBytes: 1, mediaVersion: "v", title: nil, tags: nil,
        favorite: nil, collections: nil, trashedAt: nil, purgeAt: nil))
}

/// Two days: 5 prints today, 3 yesterday. Three columns.
private func cursor(columns: Int = 3) -> LibraryCursor {
    let items = (0..<5).map { entry($0, day: 0) } + (0..<3).map { entry($0, day: 1) }
    return LibraryCursor(sections: LibraryGrouping.byDay(items), columns: columns)
}

private func id(_ n: Int, day: Int) -> PrintID {
    PrintID(host: host, filename: "d\(day)-\(n).png")
}

@Test func aPlainClickReplacesTheSelection() {
    let c = cursor()
    var selection = c.clicking(id(0, day: 0), .none, from: .empty)
    selection = c.clicking(id(2, day: 0), .none, from: selection)
    #expect(selection.items == [id(2, day: 0)])
    #expect(selection.lead == id(2, day: 0))
}

@Test func commandClickAddsAndRemoves() {
    let c = cursor()
    var selection = c.clicking(id(0, day: 0), .none, from: .empty)
    selection = c.clicking(id(2, day: 0), .toggle, from: selection)
    #expect(selection.items == [id(0, day: 0), id(2, day: 0)])
    selection = c.clicking(id(0, day: 0), .toggle, from: selection)
    #expect(selection.items == [id(2, day: 0)])
}

@Test func deselectingTheLeadDoesNotLeaveAnUnselectedLead() {
    let c = cursor()
    var selection = c.clicking(id(0, day: 0), .none, from: .empty)
    selection = c.clicking(id(1, day: 0), .toggle, from: selection)
    selection = c.clicking(id(1, day: 0), .toggle, from: selection)
    // Whatever the lead is now, it must be something that is selected.
    #expect(selection.lead.map { selection.items.contains($0) } ?? true)
}

@Test func shiftClickSelectsTheRunFromTheAnchor() {
    let c = cursor()
    var selection = c.clicking(id(1, day: 0), .none, from: .empty)
    selection = c.clicking(id(3, day: 0), .extend, from: selection)
    #expect(selection.items == [id(1, day: 0), id(2, day: 0), id(3, day: 0)])
    // Extending the other way from the same anchor replaces, not accumulates.
    selection = c.clicking(id(0, day: 0), .extend, from: selection)
    #expect(selection.items == [id(0, day: 0), id(1, day: 0)])
}

@Test func arrowRightWalksTheGridOrder() {
    let c = cursor()
    var selection = c.clicking(id(0, day: 0), .none, from: .empty)
    selection = c.moving(.right, .none, from: selection)
    #expect(selection.lead == id(1, day: 0))
}

@Test func arrowDownStepsAWholeRow() {
    let c = cursor()
    var selection = c.clicking(id(0, day: 0), .none, from: .empty)
    selection = c.moving(.down, .none, from: selection)
    // Three columns, so one row down from index 0 is index 3.
    #expect(selection.lead == id(3, day: 0))
}

@Test func downFromTheLastRowOfADayLandsInTheNextDayAtTheSameColumn() {
    let c = cursor()
    // Today has 5 items: row 0 is 0,1,2 and row 1 is 3,4. Index 4 is column 1.
    var selection = c.clicking(id(4, day: 0), .none, from: .empty)
    selection = c.moving(.down, .none, from: selection)
    // Yesterday's row 0 is 0,1,2 -- column 1 is its index 1.
    #expect(selection.lead == id(1, day: 1))
}

@Test func upFromTheFirstRowOfADayLandsInThePreviousDaysLastRow() {
    let c = cursor()
    var selection = c.clicking(id(1, day: 1), .none, from: .empty)
    selection = c.moving(.up, .none, from: selection)
    #expect(selection.lead == id(4, day: 0))
}

@Test func downClampsToWhatExistsRatherThanFallingOffTheEnd() {
    let c = cursor()
    var selection = c.clicking(id(2, day: 1), .none, from: .empty)
    selection = c.moving(.down, .none, from: selection)
    // Nothing below the last day, so it stays put instead of going nowhere.
    #expect(selection.lead == id(2, day: 1))
}

@Test func downIntoAShorterRowClampsToItsLastItem() {
    // Today 3, yesterday 2, four columns: column 2 has nothing under it.
    let items = (0..<3).map { entry($0, day: 0) } + (0..<2).map { entry($0, day: 1) }
    let c = LibraryCursor(sections: LibraryGrouping.byDay(items), columns: 4)
    var selection = c.clicking(id(2, day: 0), .none, from: .empty)
    selection = c.moving(.down, .none, from: selection)
    #expect(selection.lead == id(1, day: 1))
}

@Test func shiftArrowGrowsTheRunInsteadOfMoving() {
    let c = cursor()
    var selection = c.clicking(id(0, day: 0), .none, from: .empty)
    selection = c.moving(.right, .extend, from: selection)
    selection = c.moving(.right, .extend, from: selection)
    #expect(selection.items == [id(0, day: 0), id(1, day: 0), id(2, day: 0)])
}

@Test func anArrowWithNothingSelectedStartsAtTheBeginning() {
    let c = cursor()
    let selection = c.moving(.down, .none, from: .empty)
    #expect(selection.lead == id(0, day: 0))
}
