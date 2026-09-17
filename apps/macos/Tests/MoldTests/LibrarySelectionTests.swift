import Foundation
import MoldClient
import Testing

@testable import Mold

/// What the Library menu is offered, pinned with no view rendered -- the same
/// way `MachineSelection.rows` is.
///
/// **Fails today**: `LibrarySelection` has no notion of a caret, so the Quick
/// Look item -- whose key equivalent is a BARE space
/// (`LibraryCommands.swift:39-41`) -- is enabled whenever something is
/// selected, which is exactly when the Title field and "Add a tag" are
/// reachable. AppKit offers a key equivalent to the main menu before the field
/// editor sees it, so typing "my cat" as a title fires Quick Look at the space.
@MainActor
struct LibrarySelectionTests {
    private func selection(count: Int, editing: Bool) -> LibrarySelection {
        LibrarySelection(
            count: count, allFavorite: false, scope: .all, shelves: [],
            enclosingShelf: nil, isEditingText: editing, share: [],
            quickLook: {}, favorite: { _ in }, file: { _ in }, unfile: { _ in },
            trash: {}, putBack: {}, deleteForever: {}, emptyTrash: {})
    }

    @Test func quickLookIsOfferedForASelectionWithNoCaretInTheWindow() {
        #expect(selection(count: 2, editing: false).canQuickLook)
    }

    @Test func quickLookStandsDownWhileTextIsBeingEdited() {
        #expect(!selection(count: 2, editing: true).canQuickLook)
    }

    @Test func quickLookIsNotOfferedForNothing() {
        #expect(!selection(count: 0, editing: false).canQuickLook)
    }

    /// The menu redraws when the words on its items change -- and now also
    /// when a field takes or gives up the caret, which changes whether one of
    /// them is enabled.
    @Test func aCaretChangeIsAChangeTheMenuHasToSee() {
        #expect(selection(count: 2, editing: false) != selection(count: 2, editing: true))
    }
}
