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
}
