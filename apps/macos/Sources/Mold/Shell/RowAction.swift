import SwiftUI

/// One entry in a row's menu.
///
/// The shape `ModelsPane+Table`'s own `menuItems(for:)` already uses, named
/// once so the Machines list, the Advanced table, a curated Settings row and
/// an Accounts provider can all declare their actions the same way -- and so
/// each one's right-click menu is built from the SAME list its inline controls
/// are, rather than a second opinion that drifts.
struct RowAction<Kind: Hashable>: Identifiable {
    let kind: Kind
    let title: String
    /// Drawn last, behind a divider: anything that cannot be taken back.
    var isDestructive = false
    /// Present but inert -- a row that cannot do this, said plainly rather
    /// than by the item quietly not being there.
    var isDisabled = false

    var id: Kind { kind }
}

/// A row's actions as menu items, ordinary ones first and everything
/// destructive last behind a divider, whatever order they were declared in.
struct RowActionMenu<Kind: Hashable>: View {
    let actions: [RowAction<Kind>]
    let perform: (Kind) -> Void

    var body: some View {
        ForEach(actions.filter { !$0.isDestructive }) { item(for: $0) }
        let destructive = actions.filter(\.isDestructive)
        if !destructive.isEmpty {
            Divider()
            ForEach(destructive) { item(for: $0) }
        }
    }

    private func item(for action: RowAction<Kind>) -> some View {
        Button(action.title, role: action.isDestructive ? .destructive : nil) {
            perform(action.kind)
        }
        .disabled(action.isDisabled)
    }
}

extension RowAction {
    /// What the menu draws, in order -- pure, so a test can pin a menu
    /// without rendering one, the same way `MachineSelection.rows` does for
    /// the Machine menu.
    static func ordered(_ actions: [RowAction]) -> [RowAction] {
        actions.filter { !$0.isDestructive } + actions.filter(\.isDestructive)
    }

    /// Whether the row carries a contextual menu AT ALL.
    ///
    /// A right-click that opens an empty menu is worse than one that opens
    /// nothing: it says there is something here and then does not say what.
    /// A disabled placeholder is the same lie with an extra row. So a row
    /// with no applicable action gets no `.contextMenu` attached -- and that
    /// belongs here rather than at one call site, because every caller has
    /// the case.
    static func offersMenu(_ actions: [RowAction]) -> Bool { !actions.isEmpty }
}

extension View {
    /// A row's contextual menu, or none. THE door: no caller attaches
    /// `.contextMenu` around a `RowActionMenu` itself.
    @ViewBuilder
    func rowActionMenu<Kind: Hashable>(
        _ actions: [RowAction<Kind>], perform: @escaping (Kind) -> Void
    ) -> some View {
        if RowAction.offersMenu(actions) {
            contextMenu { RowActionMenu(actions: actions, perform: perform) }
        } else {
            self
        }
    }
}

extension TableRowContent {
    /// The same door for a `Table`'s rows. `TableRow` is not a `View`, so it
    /// needs its own overload rather than a second copy of the rule -- both
    /// ask `RowAction.offersMenu`.
    @TableRowBuilder<TableRowValue>
    func rowActionMenu<Kind: Hashable>(
        _ actions: [RowAction<Kind>], perform: @escaping (Kind) -> Void
    ) -> some TableRowContent<TableRowValue> {
        if RowAction.offersMenu(actions) {
            contextMenu { RowActionMenu(actions: actions, perform: perform) }
        } else {
            self
        }
    }
}
