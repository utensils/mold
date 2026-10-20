import AppKit
import SwiftUI
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
    private func selection(count: Int, editing: Bool, targets: [PrintID] = [],
                           performed: Performed = Performed()) -> LibrarySelection {
        LibrarySelection(
            targets: targets, count: count, allFavorite: false, scope: .all, shelves: [],
            enclosingShelf: nil, isEditingText: editing, exportFormats: [],
            meshExports: nil, trashCount: 0, localSaveCount: 0, trashLocationName: nil,
            name: nil, canReuse: count == 1,
            canUseAsSource: false, canAddReference: false,
            canUpscale: false, upscalers: [], share: [],
            perform: { performed.actions.append($0) })
    }

    /// What the menu asked for, without a menu.
    @MainActor private final class Performed {
        var actions: [LibraryAction] = []
    }

    @Test func authoredClipContextMenuOffersContinueAlongsideMenuBar() throws {
        let host = MoldHost(name: "Plato", baseURL: URL(string: "http://fixture.invalid")!)
        let hosts = HostStore(hosts: [host]) { _ in FakeBackend(host: host) }
        var actions = LibraryActions(hosts: hosts, library: LibraryStore(hosts: hosts))
        actions.continueVideo = { _ in }
        let print = try MoldJSON.decoder.decode(GalleryPrint.self, from: Data(#"{"filename":"authored.mp4","format":"mp4","metadata":{"chain_job_id":"source"},"timestamp":1000}"#.utf8))
        let entry = LibraryEntry(host: host, print: print)
        let menu = LibraryMenu(targets: [entry], scope: .all, actions: actions,
            shelves: [], enclosingShelf: nil, trashCount: 0, open: {})
        #expect(menu.items.compactMap(\.kind).contains(.continueVideo))
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
            == [.open, .quickLook, .reuse, .favorite(true), .copy, .copyFileName, .save, .trash])

        selection.perform(.quickLook)
        #expect(performed.actions == [.quickLook])
    }

    /// The menu redraws when the words on its items change -- and now also
    /// when a field takes or gives up the caret, which changes whether one of
    /// them is enabled.
    @Test func aCaretChangeIsAChangeTheMenuHasToSee() {
        #expect(selection(count: 2, editing: false) != selection(count: 2, editing: true))
    }

    @Test func copyMenuUsesCapturedDestinationsWithoutAWindowEnvironment() {
        let host = MoldHost(name: "Fixture", baseURL: URL(string: "http://fixture.invalid")!)
        var called: MoldHost?
        var offered = selection(count: 1, editing: false)
        let original = offered
        offered.copyDestinations = [host]
        offered.copyTo = { called = $0 }
        #expect(offered != original)
        let menu = LibraryCopyToMenu(destinations: offered.copyDestinations, copy: offered.copyTo)
        let view = NSHostingView(rootView: menu)
        view.layoutSubtreeIfNeeded()
        #expect(view.fittingSize.height > 0)
        menu.copy(host)
        #expect(called == host)
    }

    @Test func sameShapedPrintSelectionsStillReplaceTheirActionClosures() {
        let host = UUID()
        let first = PrintID(host: host, filename: "first.png")
        let second = PrintID(host: host, filename: "second.png")

        #expect(selection(count: 1, editing: false, targets: [first])
            != selection(count: 1, editing: false, targets: [second]))
    }

    @Test func sameShapedFileSelectionsStillReplaceTheirActionClosures() {
        let host = UUID()
        let first = PrintID(host: host, filename: "first.png")
        let second = PrintID(host: host, filename: "second.png")
        let firstFile = LibraryFile(
            targets: [first], count: 1, exportFormats: ["png"], meshExports: nil,
            save: {}, export: { _ in })
        let secondFile = LibraryFile(
            targets: [second], count: 1, exportFormats: ["png"], meshExports: nil,
            save: {}, export: { _ in })

        #expect(firstFile != secondFile)
    }
}
