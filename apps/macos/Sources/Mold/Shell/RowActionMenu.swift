import MoldClient
import SwiftUI

/// A declared list of actions, as menu rows.
///
/// THE renderer. A contextual menu, a menu-bar menu and a click-to-open
/// `Menu` differ in WHERE they appear and in what they are given, never in
/// what is offered or what it is called -- so all three draw this, and the
/// order, the dividers and the empty case are `RowAction.rendered`'s answer
/// rather than each surface's.
struct RowActionMenu<Kind: Hashable>: View {
    let actions: [RowAction<Kind>]
    let perform: (Kind) -> Void
    /// The menu bar's chords, asked per item. A contextual menu carries none,
    /// which is why this answers nothing by default -- and why a chord is a
    /// property of the SURFACE rather than of the action: the same Cancel Job
    /// is ⌘⌫ in the Queue menu and bare on a row.
    var shortcut: (Kind) -> KeyboardShortcut? = { _ in nil }

    var body: some View {
        ForEach(RowAction.rendered(actions)) { action in
            if action.isSeparator {
                Divider()
            } else if action.isSubmenu {
                Menu(action.title) {
                    RowActionMenu(actions: action.children, perform: perform, shortcut: shortcut)
                }
            } else if let kind = action.kind {
                Button(action.title, role: action.isDestructive ? .destructive : nil) {
                    perform(kind)
                }
                .disabled(action.isDisabled)
                .keyboardShortcut(shortcut(kind))
            }
        }
    }
}

extension View {
    /// A row's contextual menu, or none. THE door: no caller attaches
    /// `.contextMenu` of its own -- `MenuSurfaceTests` scans for it.
    ///
    /// `extra` is for the system controls a `RowAction` cannot model -- a
    /// `ShareLink` is AirDrop, Messages and Mail, not something this app
    /// performs -- and it rides along rather than deciding whether a menu
    /// appears, because a menu holding nothing but a Share sheet is the empty
    /// menu by another name.
    @ViewBuilder
    func rowActionMenu<Kind: Hashable, Extra: View>(
        _ actions: [RowAction<Kind>],
        perform: @escaping (Kind) -> Void,
        @ViewBuilder extra: @escaping () -> Extra = { EmptyView() }
    ) -> some View {
        if RowAction.offersMenu(actions) {
            contextMenu {
                RowActionMenu(actions: actions, perform: perform)
                extra()
            }
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
