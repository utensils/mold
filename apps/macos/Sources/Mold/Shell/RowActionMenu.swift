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
    /// The system controls a `RowAction` cannot model, drawn with the
    /// ordinary items at `RowAction.extraInsertionIndex` so the destructive
    /// tail stays last. Type-erased: a menu row's identity is its position.
    var extra: () -> AnyView = { AnyView(EmptyView()) }

    /// Keyed by POSITION, not by content: a drawn menu repeats itself --
    /// every separator is the same value, and two submenus can share a title
    /// -- so a content-derived identity handed SwiftUI "ID used by multiple
    /// child views" for any menu with two dividers. The Queue menu draws up
    /// to four.
    var body: some View {
        let drawn = RowAction.rendered(actions)
        let insertion = RowAction.extraInsertionIndex(drawn)
        ForEach(Array(drawn.enumerated()), id: \.offset) { offset, action in
            if offset == insertion { extra() }
            row(action)
        }
        if insertion == drawn.count { extra() }
    }

    @ViewBuilder
    private func row(_ action: RowAction<Kind>) -> some View {
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

extension View {
    /// A large grid can attach menus to thousands of rows. Build the plan
    /// only when the person opens one, rather than on every selection redraw.
    func rowActionMenu<Kind: Hashable, Extra: View>(
        lazy actions: @escaping () -> [RowAction<Kind>],
        perform: @escaping (Kind) -> Void,
        @ViewBuilder extra: @escaping () -> Extra = { EmptyView() }
    ) -> some View {
        ContextualRowActionOwner(content: self, actions: actions, perform: perform,
                                 extra: { AnyView(extra()) })
            .equatable()
    }

    /// A row's contextual menu, or none. THE door: no caller attaches
    /// `.contextMenu` of its own -- `MenuSurfaceTests` scans for it.
    ///
    /// `extra` is for the system controls a `RowAction` cannot model -- a
    /// `ShareLink` is AirDrop, Messages and Mail, not something this app
    /// performs -- and it rides along rather than deciding whether a menu
    /// appears, because a menu holding nothing but a Share sheet is the empty
    /// menu by another name. It is drawn among the ordinary items, never
    /// under the destructive tail.
    @ViewBuilder
    func rowActionMenu<Kind: Hashable, Extra: View>(
        _ actions: [RowAction<Kind>],
        perform: @escaping (Kind) -> Void,
        @ViewBuilder extra: @escaping () -> Extra = { EmptyView() }
    ) -> some View {
        if RowAction.offersMenu(actions) {
            ContextualRowActionOwner(content: self, actions: { actions }, perform: perform,
                                     extra: { AnyView(extra()) })
                .equatable()
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
            contextMenu { ContextualRowActionMenu(actions: actions, perform: perform).equatable() }
        } else {
            self
        }
    }
}

/// Freeze the owner together with its context-menu attachment. Freezing only
/// the menu's rows still lets a source redraw replace SwiftUI's AppKit bridge
/// and dismiss an otherwise unchanged native menu.
struct ContextualRowActionOwner<Content: View, Kind: Hashable>: View, Equatable {
    let content: Content
    let actions: () -> [RowAction<Kind>]
    let perform: (Kind) -> Void
    let tracking: ContextMenuTracking
    let extra: () -> AnyView
    /// Fresh per value; persistent @State storage here retains stale row inputs.
    let latest: ContextMenuOwnerInputs<Content, Kind>
    @State private var completionRevision = 0

    init(content: Content, actions: @escaping () -> [RowAction<Kind>],
         perform: @escaping (Kind) -> Void, tracking: ContextMenuTracking = .shared,
         extra: @escaping () -> AnyView = { AnyView(EmptyView()) }) {
        self.content = content
        self.actions = actions
        self.perform = perform
        self.tracking = tracking
        self.extra = extra
        latest = ContextMenuOwnerInputs(
            content: content, actions: actions, perform: perform, extra: extra)
    }

    nonisolated static func == (lhs: Self, rhs: Self) -> Bool {
        MainActor.assumeIsolated {
            lhs.latest.update(content: rhs.content, actions: rhs.actions,
                              perform: rhs.perform, extra: rhs.extra)
            return lhs.tracking === rhs.tracking && lhs.tracking.isTracking
        }
    }

    var body: some View {
        // Capture the pending inputs as values so an equal comparison cannot
        // change the closures belonging to the menu that is already open.
        let content = latest.content
        let actions = latest.actions
        let perform = latest.perform
        let extra = latest.extra
        let _ = completionRevision
        return content.contextMenu {
            ContextualRowActionMenuProvider(actions: actions, perform: perform,
                                           tracking: tracking, extra: extra)
        }
        .onReceive(tracking.didFinishTracking) { completionRevision = $0 }
    }
}
