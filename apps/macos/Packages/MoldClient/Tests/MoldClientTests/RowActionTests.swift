import Foundation
import Testing

@testable import MoldClient

/// The house rules every menu in the app goes through, pinned once at the
/// shared type rather than fifteen times at the surfaces.
///
/// **Fails today**: there were THREE menu models -- `RowAction` (no submenus,
/// no explicit grouping), `GenerateMenuItems` and `LibraryMenuItem` -- each
/// with its own copy of "destructive last" and its own answer for an empty
/// row, and a hand-written fourth in `QueueHoldRow` besides.
@Suite struct RowActionSuite {
    private typealias Item = RowAction<String>

    /// Whatever order a surface declares them in, everything destructive ends
    /// up last, behind a divider -- so a menu's bottom item is always the one
    /// that cannot be taken back, and a right-click never lands the cursor on
    /// it by accident.
    @Test func destructiveItemsComeLastBehindASeparator() {
        let drawn = RowAction.rendered([
            Item(kind: "remove", title: "Remove…", isDestructive: true),
            Item(kind: "edit", title: "Edit…"),
            Item(kind: "check", title: "Check Now"),
        ])

        #expect(drawn.map(\.kind) == ["edit", "check", nil, "remove"])
        #expect(drawn.dropLast().last?.isSeparator == true)
        // Nothing destructive: no divider to draw.
        #expect(RowAction.rendered([Item(kind: "edit", title: "Edit…")]).count == 1)
    }

    /// A right-click that opens an empty menu says there is something here and
    /// then does not say what. A disabled placeholder is the same lie with an
    /// extra row.
    @Test func aRowWithNothingApplicableCarriesNoMenu() {
        #expect(!Item.offersMenu([]))
        #expect(!Item.offersMenu([.separator, .separator]))
        #expect(RowAction.offersMenu([Item(kind: "clear", title: "Clear")]))
        // Disabled is still OFFERED -- present and inert, said plainly.
        #expect(RowAction.offersMenu([Item(kind: "edit", title: "Edit…", isDisabled: true)]))
    }

    /// A surface that groups its own list is left alone: the Library's
    /// separators are where they are on purpose, and sorting its destructive
    /// rows together would move Move to Trash down beside Delete Collection.
    @Test func aListThatGroupsItselfKeepsItsOwnOrder() {
        let declared: [Item] = [
            Item(kind: "trash", title: "Move to Trash", isDestructive: true),
            .separator,
            Item(kind: "rename", title: "Rename…"),
        ]
        #expect(RowAction.rendered(declared).map(\.title)
            == ["Move to Trash", "", "Rename…"])
    }

    @Test func noMenuOpensClosesOrDoublesOnADivider() {
        let drawn = RowAction.rendered([
            .separator, .separator,
            Item(kind: "open", title: "Open"),
            .separator, .separator,
            Item(kind: "copy", title: "Copy"),
            .separator,
        ])
        #expect(drawn.map(\.title) == ["Open", "", "Copy"])
    }

    /// A submenu with nothing enabled in it is a dead end, and a gate at every
    /// caller is a gate somebody forgets.
    @Test func anEmptySubmenuIsNotOffered() {
        let shelves = Item(title: "Move to Collection", children: [])
        let full = Item(title: "Export…", children: [Item(kind: "png", title: "PNG")])
        let inert = Item(title: "Send to", children: [Item(kind: "x", title: "X", isDisabled: true)])

        #expect(RowAction.rendered([shelves, full, inert]).map(\.title) == ["Export…"])
        #expect(!RowAction.offersMenu([shelves]))
    }

    /// A submenu has no action to be identified by, so it is identified by
    /// its words -- two of them never share a title.
    @Test func everyDrawnRowHasItsOwnIdentity() {
        let drawn = RowAction.rendered([
            Item(kind: "open", title: "Open"),
            Item(title: "Move to Collection", children: [Item(kind: "a", title: "A")]),
            Item(title: "Export…", children: [Item(kind: "png", title: "PNG")]),
        ])
        #expect(Set(drawn.map(\.id)).count == drawn.count)
    }
}
