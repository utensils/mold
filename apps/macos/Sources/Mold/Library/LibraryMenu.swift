import MoldClient
import SwiftUI

/// The right-click menu for a print.
///
/// Acts on the whole selection when the clicked print is part of it, and on
/// just that print when it isn't -- which is what every Mac app does and what
/// stops a stray right-click throwing away a careful selection.
///
/// WHAT it offers is `LibraryMenuPlan`'s, not this view's: the menu bar draws
/// the same list, in the same order, with the same words.
struct LibraryMenu: View {
    let targets: [LibraryEntry]
    let scope: LibraryScope
    let actions: LibraryActions
    let shelves: [CollectionShelf]
    let enclosingShelf: CollectionShelf?
    let trashCount: Int
    let open: (() -> Void)?

    var body: some View {
        LibraryMenuItems(items: plan.items) { action in
            actions.perform(action, on: targets, scope: scope, open: open)
        }
        // Outside the plan: a real `ShareLink` -- AirDrop, Messages, Mail,
        // Photos, Save to Files -- which is a system control and not something
        // this app performs. The file is fetched when the sheet asks for it.
        if !scope.isTrash, !targets.isEmpty {
            ShareLink(items: targets.map(actions.draggable)) { print in
                SharePreview(print.filename)
            }
        }
    }

    private var plan: LibraryMenuPlan {
        LibraryMenuPlan(
            scope: scope.menuKind,
            count: targets.count,
            allFavorite: !targets.isEmpty && targets.allSatisfy(\.print.isFavorite),
            name: targets.count == 1 ? targets[0].print.displayName : nil,
            shelves: shelves,
            enclosingShelf: enclosingShelf,
            exportFormats: targets.count == 1 ? actions.exportFormats(for: targets[0]) : [],
            canReuse: actions.reuse != nil && open != nil,
            trashCount: trashCount
        )
    }
}

extension LibraryScope {
    /// What the plan needs to know about which shelf is showing.
    var menuKind: LibraryScopeKind {
        switch self {
        case .trash: .trash
        case .collection: .collection
        case .all, .favorites: .prints
        }
    }
}
