import MoldClient

/// What a machine row in the SIDEBAR offers.
///
/// Named apart from Settings ▸ Machines' own row actions because they are
/// different surfaces: that one edits and removes a saved machine, this one
/// is the fleet list you navigate by. What they share is the rule -- the
/// items are declared once and both the inline surface and the right-click
/// menu draw from that declaration.
///
/// Here the other surface is the MENU BAR. `MachineCommands` spells the same
/// two items, so the words live in this file and both read them; the row
/// hardcoding "Check Now" while the menu hardcoded "Check Now" was two
/// spellings of one action waiting to disagree.
enum SidebarMachineActions {
    enum Kind: Hashable {
        case checkNow, setDefault, showInLibrary
    }

    /// `HostStore.refresh(_:)` -- one machine asked what it is, a narrower
    /// question than ⌘R's whole-pane refresh (`MachineCommands.swift`).
    static let checkNow = "Check Now"
    /// `HostStore.setDefault(_:)`.
    static let setAsDefault = "Set as Default"
    /// The Library's, not the Machine menu's -- which is why it sits behind
    /// a divider rather than among the first two.
    static let showInLibrary = "Show in Library"

    /// The Machine menu's own two items first, in its order and its words,
    /// then the Library's. Nothing here is destructive: a machine row cannot
    /// remove anything, so `RowActionMenu` draws no divider and the
    /// separation is by meaning alone.
    static func offered() -> [RowAction<Kind>] {
        RowAction.ordered([
            RowAction(kind: .checkNow, title: checkNow),
            RowAction(kind: .setDefault, title: setAsDefault),
            RowAction(kind: .showInLibrary, title: showInLibrary),
        ])
    }
}
