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
                      canReuse: Bool = true, canSource: Bool = false,
                      canReference: Bool = false, trashCount: Int = 0) -> LibraryMenuPlan {
        LibraryMenuPlan(scope: scope, count: count, allFavorite: allFavorite,
                        name: "robot.png", shelves: shelves, enclosingShelf: enclosing,
                        exportFormats: formats, canReuse: canReuse,
                        canUseAsSource: canSource, canAddReference: canReference,
                        trashCount: trashCount)
    }

    /// Every row that is not a divider, by the words on it -- which pins the
    /// WORDING as well as the order, and is the same question a submenu can
    /// be asked (it has no action of its own to be named by).
    private func titles(_ plan: LibraryMenuPlan) -> [String] {
        plan.items.filter { !$0.isSeparator }.map(\.title)
    }

    private func kinds(_ plan: LibraryMenuPlan) -> [LibraryAction] {
        plan.items.compactMap(\.kind)
    }

    /// The whole offer, in order -- what BOTH menus draw.
    @Test func oneSelectedPrintIsOfferedEverythingInOneOrder() {
        let offered = plan(shelves: [shelf("Smurfs")], formats: ["mp4"])
        #expect(titles(offered) == ["Open", "Quick Look “robot.png”", "Use These Settings",
                                    "Add to Favourites", "Move to Collection", "Copy",
                                    "Save a Copy…", "Export…", "Move to Trash"])
        #expect(kinds(offered) == [.open, .quickLook, .reuse, .favorite(true),
                                   .copy, .save, .trash])
    }

    /// Destructive last, after a divider, and marked.
    @Test func whatDestroysSomethingComesLastAndSaysSo() {
        let items = plan().items
        let last = items.last
        #expect(last?.kind == .trash)
        #expect(last?.isDestructive == true)
        #expect(items.dropLast().last?.isSeparator == true)
        #expect(items.filter(\.isDestructive).map(\.kind) == [.trash])
    }

    /// A row with nothing applicable gets NO menu, never an empty one.
    @Test func nothingSelectedIsNoMenuAtAll() {
        #expect(plan(count: 0, canReuse: false).items.isEmpty)
        #expect(plan(scope: .trash, count: 0).items.isEmpty)
        #expect(!RowAction.offersMenu(plan(count: 0, canReuse: false).items))
    }

    @Test func severalSelectedPrintsDropTheOnesThatOnlyMakeSenseForOne() {
        let offered = plan(count: 3, formats: ["mp4"])
        #expect(!kinds(offered).contains(.reuse))
        #expect(!titles(offered).contains("Export…"))
        #expect(kinds(offered).contains(.trash))
    }

    @Test func oneRasterOffersOnlyTheSupportedDestinationAttachments() {
        let both = plan(canSource: true, canReference: true)
        #expect(kinds(both).contains(.useAsSourceImage))
        #expect(kinds(both).contains(.addAsReference))

        let sourceOnly = plan(canSource: true)
        #expect(kinds(sourceOnly).contains(.useAsSourceImage))
        #expect(!kinds(sourceOnly).contains(.addAsReference))
        #expect(!kinds(plan(count: 2, canSource: true, canReference: true))
            .contains(.useAsSourceImage))
    }

    @Test func theTrashOffersItsOwnThings() {
        let offered = plan(scope: .trash, trashCount: 4)
        #expect(kinds(offered) == [.putBack, .deleteForever, .emptyTrash])
        #expect(offered.items.filter(\.isDestructive).map(\.kind)
            == [.deleteForever, .emptyTrash])
    }

    /// The shelf's own three, which lived only in the sidebar's right-click
    /// menu and so were unreachable from the keyboard.
    @Test func showingACollectionOffersWhatToDoWithIt() {
        let smurfs = shelf("Smurfs")
        let offered = plan(scope: .collection, shelves: [smurfs], enclosing: smurfs)
        #expect(kinds(offered).contains(.unfile(slug: "smurfs")))
        #expect(titles(offered).suffix(3)
            == ["Rename “Smurfs”…", "Hide from All Prints", "Delete Collection…"])
    }

    @Test func aHiddenCollectionIsOfferedTheOtherHalfOfTheToggle() {
        let hidden = shelf("Smurfs", hidden: true)
        let items = plan(scope: .collection, enclosing: hidden).items
        let toggle = items.first { $0.kind == .setCollectionHidden(false) }
        #expect(toggle?.title == "Show in All Prints")
    }

    // MARK: - One wording

    @Test func favouriteIsSpeltTheWayTheSidebarSpellsIt() {
        #expect(plan().items.first { $0.kind == .favorite(true) }?.title == "Add to Favourites")
        #expect(plan(allFavorite: true).items.first { $0.kind == .favorite(false) }?.title
            == "Remove from Favourites")
    }

    @Test func quickLookNamesWhatItIsAbout() {
        #expect(plan().items.first { $0.kind == .quickLook }?.title == "Quick Look “robot.png”")
        #expect(plan(count: 4).items.first { $0.kind == .quickLook }?.title
            == "Quick Look 4 Prints")
    }

    @Test func savingSeveralSaysHowMany() {
        #expect(plan(count: 3).items.first { $0.kind == .save }?.title == "Save 3 Copies…")
    }

    // MARK: - Dividers

    @Test func noMenuStartsOrEndsWithASeparatorOrDoublesOne() {
        for one in [plan(), plan(count: 3), plan(canReuse: false),
                    plan(scope: .trash, trashCount: 2), plan(scope: .trash)] {
            let items = one.items
            #expect(items.first?.isSeparator != true)
            #expect(items.last?.isSeparator != true)
            #expect(!zip(items, items.dropFirst()).contains { $0.isSeparator && $1.isSeparator })
        }
    }

    /// Every submenu has something in it -- a "Move to Collection" with no
    /// collections is a dead end.
    @Test func aSubmenuIsOnlyOfferedWhenItHasEntries() {
        #expect(!titles(plan(shelves: [])).contains("Move to Collection"))
        #expect(!titles(plan(formats: [])).contains("Export…"))
        #expect(plan(shelves: [shelf("Smurfs")]).items.first { $0.isSubmenu }?
            .children.map(\.kind) == [.file(slug: "smurfs")])
    }
}
