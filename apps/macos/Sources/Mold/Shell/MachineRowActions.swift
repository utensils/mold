import MoldClient

/// What one machine row in Settings ▸ Machines can do.
///
/// Declared once and rendered twice: the footer's Edit and Remove buttons
/// act on the selected row through the same two closures the right-click menu
/// does. Before this, the only way to edit a machine was a double-click and
/// the only way to remove one was a "−" button whose meaning you had to
/// already know.
enum MachineRowActions {
    enum Kind: Hashable {
        case edit, checkNow, copyAddress, setDefault, remove
    }

    /// `isManaged` is false for This Mac's in-process engine: its address is
    /// whatever port the engine bound and it is not a saved row, so there is
    /// nothing to edit and nothing to remove (`MachinesSettings.isManaged`).
    /// Those two are DISABLED rather than absent -- a menu that changes shape
    /// per row is one nobody learns.
    static func offered(isManaged: Bool, isDefault: Bool) -> [RowAction<Kind>] {
        RowAction.ordered([
            RowAction(kind: .edit, title: "Edit…", isDisabled: !isManaged),
            RowAction(kind: .checkNow, title: "Check Now"),
            RowAction(kind: .copyAddress, title: "Copy Address"),
            RowAction(kind: .setDefault,
                      title: isDefault ? "Already the Default Machine" : "Set as Default",
                      isDisabled: isDefault),
            RowAction(kind: .remove, title: "Remove…", isDestructive: true, isDisabled: !isManaged),
        ])
    }
}
