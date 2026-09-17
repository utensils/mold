import Foundation
import Testing

@testable import MoldClient

/// What the Library offers for a selection, declared once.
///
/// **Fails today**: `LibraryMenu` (the tile) and `LibraryCommands` (the menu
/// bar) are two hand-written lists that disagree -- Move to Collection and
/// Remove from ⟨shelf⟩ only in the menu bar, Open, Use These Settings and Copy
/// only on the tile, "Favorite" against "Add to Favorites". There is no plan
/// to assert on.
@Suite struct LibraryMenuPlanSuite {
    private func shelf(_ name: String, hidden: Bool = false) -> CollectionShelf {
        CollectionShelf(slug: name.lowercased(), name: name, count: 0,
                        hidden: hidden, hosts: [:])
    }

    private func plan(scope: LibraryScopeKind = .prints, count: Int = 1,
                      allFavorite: Bool = false, shelves: [CollectionShelf] = [],
                      enclosing: CollectionShelf? = nil, formats: [String] = [],
                      canReuse: Bool = true, trashCount: Int = 0) -> LibraryMenuPlan {
        LibraryMenuPlan(scope: scope, count: count, allFavorite: allFavorite,
                        name: "robot.png", shelves: shelves, enclosingShelf: enclosing,
                        exportFormats: formats, canReuse: canReuse, trashCount: trashCount)
    }

    private func ids(_ plan: LibraryMenuPlan) -> [String] {
        plan.items.filter { !$0.isDivider }.map(\.id)
    }

    /// The whole offer, in order -- what BOTH menus draw.
    @Test func oneSelectedPrintIsOfferedEverythingInOneOrder() {
        let offered = ids(plan(shelves: [shelf("Smurfs")], formats: ["mp4"]))
        #expect(offered == ["open", "quickLook", "reuse", "favorite", "file",
                            "copy", "save", "export", "trash"])
    }

    /// Destructive last, after a divider, and marked.
    @Test func whatDestroysSomethingComesLastAndSaysSo() {
        let items = plan().items
        let last = items.last
        #expect(last?.id == "trash")
        #expect(last?.isDestructive == true)
        #expect(items.dropLast().last?.isDivider == true)
        #expect(items.filter(\.isDestructive).allSatisfy { $0.id == "trash" })
    }

    /// A row with nothing applicable gets NO menu, never an empty one.
    @Test func nothingSelectedIsNoMenuAtAll() {
        #expect(plan(count: 0, canReuse: false).items.isEmpty)
        #expect(plan(scope: .trash, count: 0).items.isEmpty)
    }

    @Test func severalSelectedPrintsDropTheOnesThatOnlyMakeSenseForOne() {
        let offered = ids(plan(count: 3, formats: ["mp4"]))
        #expect(!offered.contains("reuse"))
        #expect(!offered.contains("export"))
        #expect(offered.contains("trash"))
    }

    @Test func theTrashOffersItsOwnThings() {
        let offered = ids(plan(scope: .trash, trashCount: 4))
        #expect(offered == ["putBack", "deleteForever", "emptyTrash"])
        #expect(plan(scope: .trash, trashCount: 4).items.filter(\.isDestructive)
            .map(\.id) == ["deleteForever", "emptyTrash"])
    }

    /// The shelf's own three, which lived only in the sidebar's right-click
    /// menu and so were unreachable from the keyboard.
    @Test func showingACollectionOffersWhatToDoWithIt() {
        let smurfs = shelf("Smurfs")
        let offered = ids(plan(scope: .collection, shelves: [smurfs], enclosing: smurfs))
        #expect(offered.contains("unfile"))
        #expect(offered.suffix(3) == ["renameCollection", "hideCollection", "deleteCollection"])
    }

    @Test func aHiddenCollectionIsOfferedTheOtherHalfOfTheToggle() {
        let hidden = shelf("Smurfs", hidden: true)
        let items = plan(scope: .collection, enclosing: hidden).items
        #expect(items.first { $0.id == "hideCollection" }?.title == "Show in All Prints")
        #expect(items.first { $0.id == "hideCollection" }?.action == .setCollectionHidden(false))
    }

    // MARK: - One wording

    @Test func favouriteIsSpeltTheWayTheSidebarSpellsIt() {
        #expect(plan().items.first { $0.id == "favorite" }?.title == "Add to Favourites")
        #expect(plan(allFavorite: true).items.first { $0.id == "favorite" }?.title
            == "Remove from Favourites")
    }

    @Test func quickLookNamesWhatItIsAbout() {
        #expect(plan().items.first { $0.id == "quickLook" }?.title == "Quick Look “robot.png”")
        #expect(plan(count: 4).items.first { $0.id == "quickLook" }?.title
            == "Quick Look 4 Prints")
    }

    @Test func savingSeveralSaysHowMany() {
        #expect(plan(count: 3).items.first { $0.id == "save" }?.title == "Save 3 Copies…")
    }

    // MARK: - Dividers

    @Test func noMenuStartsOrEndsWithASeparatorOrDoublesOne() {
        for one in [plan(), plan(count: 3), plan(canReuse: false),
                    plan(scope: .trash, trashCount: 2), plan(scope: .trash)] {
            let items = one.items
            #expect(items.first?.isDivider != true)
            #expect(items.last?.isDivider != true)
            #expect(!zip(items, items.dropFirst()).contains { $0.isDivider && $1.isDivider })
        }
    }

    /// Every submenu has something in it -- a "Move to Collection" with no
    /// collections is a dead end.
    @Test func aSubmenuIsOnlyOfferedWhenItHasEntries() {
        #expect(!ids(plan(shelves: [])).contains("file"))
        #expect(!ids(plan(formats: [])).contains("export"))
        #expect(plan(shelves: [shelf("Smurfs")]).items.first { $0.id == "file" }?
            .children.map(\.action) == [.file(slug: "smurfs")])
    }
}
