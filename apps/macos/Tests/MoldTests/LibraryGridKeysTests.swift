import MoldClient
import SwiftUI
import Testing

@testable import Mold

/// What a key press means to the grid.
///
/// **Fails today**: the grid reads `ClickModifiers.current` for an arrow --
/// the last left-mouse-DOWN's flags (`LibraryGrid.swift:126`) -- and binds
/// `.onKeyPress(.delete)`, which matches regardless of modifiers
/// (`LibraryGrid.swift:71`). There is no map to ask.
@MainActor
struct LibraryGridKeysTests {

    @Test func aBareArrowMovesTheSelection() {
        #expect(LibraryGridKeys.action(for: .leftArrow, modifiers: [])
            == .move(.left, LibraryCursor.Modifier.none))
        #expect(LibraryGridKeys.action(for: .downArrow, modifiers: [])
            == .move(.down, LibraryCursor.Modifier.none))
    }

    /// The modifier comes from THIS key press, not from whatever the last
    /// click happened to be holding.
    @Test func shiftExtendsFromTheKeyPressItself() {
        #expect(LibraryGridKeys.action(for: .rightArrow, modifiers: .shift)
            == .move(.right, .extend))
    }

    /// ⌘← is Back on a Mac and ⌥← is a word jump. Neither is the grid's, and
    /// neither is a selection toggle.
    @Test func anArrowWithAnythingElseHeldIsNotTheGridsToAnswer() {
        #expect(LibraryGridKeys.action(for: .leftArrow, modifiers: .command) == nil)
        #expect(LibraryGridKeys.action(for: .leftArrow, modifiers: .option) == nil)
    }

    @Test func aBareBackspaceDoesNothing() {
        #expect(LibraryGridKeys.action(for: .delete, modifiers: []) == nil)
    }

    /// What the menu item and the README both promise.
    @Test func commandBackspaceTrashesTheSelection() {
        #expect(LibraryGridKeys.action(for: .delete, modifiers: .command) == .trash)
        #expect(LibraryGridKeys.action(for: .delete, modifiers: [.command, .shift]) == nil)
    }

    @Test func spaceIsQuickLookAndReturnOpens() {
        #expect(LibraryGridKeys.action(for: .space, modifiers: []) == .quickLook)
        #expect(LibraryGridKeys.action(for: .return, modifiers: []) == .open)
        #expect(LibraryGridKeys.action(for: .space, modifiers: .command) == nil)
    }

    @Test func escapeClearsTheGridSelection() {
        #expect(LibraryGridKeys.action(for: .escape, modifiers: []) == .clearSelection)
        #expect(LibraryGridKeys.action(for: .escape, modifiers: .command) == nil)
    }

    /// The binding and the map cannot drift: every key the grid asks for has
    /// a meaning, and nothing else does.
    @Test func everyBoundKeyMeansSomething() {
        for key in LibraryGridKeys.keys {
            #expect(LibraryGridKeys.action(for: key, modifiers: []) != nil
                || key == .delete)
        }
        #expect(LibraryGridKeys.action(for: "a", modifiers: []) == nil)
    }
    @Test func regroupingKeepsTheVisiblePrintDeepInTheLibrary() throws {
        let host = MoldHost(name: "Fixture", baseURL: URL(string: "http://fixture.invalid")!)
        let entries = try (0..<1000).map { index in
            let json = "{\"filename\":\"\(index).png\",\"metadata\":{\"width\":1024,\"height\":1024},\"timestamp\":\(1000 + (index / 100) * 86400)}"
            return LibraryEntry(host: host, print: try MoldJSON.decoder.decode(GalleryPrint.self, from: Data(json.utf8)))
        }
        let layout = JustifiedLibraryLayout()
        let grouped = layout.resolve(LibraryGrouping.byDay(entries), width: 760, targetHeight: 132)
        let visibleRow = try #require(grouped[5].rows.dropFirst(3).first)
        let anchor = try #require(LibraryGrid.visibleAnchor(in: entries, visibleRowIDs: [visibleRow.id]))
        #expect(anchor == visibleRow.id)
        #expect(anchor != entries[0].id)
        let continuous = layout.resolve(LibraryGrouping.ungrouped(entries), width: 760, targetHeight: 132)
        let newRow = try #require(continuous[0].rows.first { row in
            row.items.contains { continuous[0].source.items[$0.index].id == anchor }
        })
        #expect(newRow.items.contains { continuous[0].source.items[$0.index].id == anchor })
    }

}
