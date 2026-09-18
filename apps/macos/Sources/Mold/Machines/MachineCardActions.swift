import MoldClient

/// What one machine's card on the fleet overview offers.
///
/// ONE declaration, three surfaces: the card's right-click menu, the card's
/// own inline control, and the Machine menu in the menu bar
/// (`MachineCommands.swift`). The two words the sidebar's machine row already
/// spells come from `SidebarMachineActions` rather than being written a third
/// time here -- renaming one there renames it everywhere.
enum MachineCardActions {
    enum Kind: Hashable {
        case open, checkNow, setDefault, copyAddress, edit, remove
    }

    static let open = "Open"
    static let copyAddress = "Copy Address"
    static let edit = "Edit…"
    /// The ellipsis is the promise the dialog keeps: this asks first, and says
    /// that the machine's key goes with it (`MachineRemoval.swift`).
    static let remove = "Remove…"

    /// `isThisMac` is the in-process engine: its address is whatever port the
    /// engine bound and it is not a saved row, so there is nothing to edit and
    /// nothing to remove. Those two are ABSENT here rather than disabled --
    /// this is a card, not a table row, and an inert item on it is a control
    /// that never becomes anything.
    ///
    /// Set as Default is absent on the machine that already IS the default,
    /// for the same reason.
    static func offered(isThisMac: Bool, isDefault: Bool) -> [RowAction<Kind>] {
        var actions = [
            RowAction(kind: Kind.open, title: open),
            RowAction(kind: .checkNow, title: SidebarMachineActions.checkNow),
        ]
        if !isDefault {
            actions.append(RowAction(kind: .setDefault, title: SidebarMachineActions.setAsDefault))
        }
        actions.append(RowAction(kind: .copyAddress, title: copyAddress))
        if !isThisMac {
            actions.append(RowAction(kind: .edit, title: edit))
            actions.append(RowAction(kind: .remove, title: remove, isDestructive: true))
        }
        return RowAction.ordered(actions)
    }

    /// The same list with nothing to act on -- what the Machine menu draws
    /// when no machine is picked. Present and inert, so Help ▸ Search finds
    /// "Set as Default" from every pane, which is the rule the menu already
    /// followed before the card list fed it.
    static func unavailable() -> [RowAction<Kind>] {
        offered(isThisMac: false, isDefault: false).map {
            RowAction(kind: $0.kind, title: $0.title, isDestructive: $0.isDestructive,
                      isDisabled: true)
        }
    }
}
