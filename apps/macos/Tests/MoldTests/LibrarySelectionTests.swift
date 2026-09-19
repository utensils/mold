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
    private func selection(count: Int, editing: Bool,
                           performed: Performed = Performed()) -> LibrarySelection {
        LibrarySelection(
            count: count, allFavorite: false, scope: .all, shelves: [],
            enclosingShelf: nil, isEditingText: editing, exportFormats: [],
            meshExports: nil, trashCount: 0, name: nil, canReuse: count == 1,
            canUseAsSource: false, canAddReference: false,
            canUpscale: false, upscalers: [], share: [],
            perform: { performed.actions.append($0) })
    }

    /// What the menu asked for, without a menu.
    @MainActor private final class Performed {
        var actions: [LibraryAction] = []
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

    /// The menu bar draws the SAME list a tile's right-click menu draws, and
    /// performs it through the same door.
    @Test func theMenuBarOffersThePlanAndNothingOfItsOwn() {
        let performed = Performed()
        let selection = selection(count: 1, editing: false, performed: performed)

        #expect(selection.plan.items.compactMap(\.kind)
            == [.open, .quickLook, .reuse, .favorite(true), .copy, .save, .trash])

        selection.perform(.quickLook)
        #expect(performed.actions == [.quickLook])
    }

    /// The menu redraws when the words on its items change -- and now also
    /// when a field takes or gives up the caret, which changes whether one of
    /// them is enabled.
    @Test func aCaretChangeIsAChangeTheMenuHasToSee() {
        #expect(selection(count: 2, editing: false) != selection(count: 2, editing: true))
    }
}
