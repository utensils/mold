import MoldClient
import SwiftUI

/// The right-click menu for a print.
///
/// Acts on the whole selection when the clicked print is part of it, and on
/// just that print when it isn't -- which is what every Mac app does and what
/// stops a stray right-click throwing away a careful selection.
///
/// WHAT it offers is `LibraryMenuPlan`'s, not this type's: the menu bar draws
/// the same list, in the same order, with the same words, through the same
/// `RowActionMenu`.
struct LibraryMenu {
    let targets: [LibraryEntry]
    let scope: LibraryScope
    let actions: LibraryActions
    let shelves: [CollectionShelf]
    let enclosingShelf: CollectionShelf?
    let trashCount: Int
    let open: (() -> Void)?

    var items: [LibraryMenuPlan.Item] { plan.items }

    func perform(_ action: LibraryAction) {
        actions.perform(action, on: targets, scope: scope, open: open)
    }

    /// Outside the plan: a real `ShareLink` -- AirDrop, Messages, Mail,
    /// Photos, Save to Files -- which is a system control and not something
    /// this app performs. The file is fetched when the sheet asks for it.
    /// Empty where there is nothing to share, which draws no item.
    var share: [DraggablePrint] {
        guard !scope.isTrash else { return [] }
        return targets.map(actions.draggable)
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
            meshExports: targets.count == 1 && targets[0].print.isMesh
                ? actions.meshExports(for: targets[0]) : nil,
            canOpen: open != nil,
            canReuse: actions.reuse != nil && !scope.isTrash,
            canUseAsSource: canAttach(using: actions.useAsSource),
            canAddReference: canAttach(using: actions.addAsReference),
            canUpscale: actions.canUpscale(targets),
            upscalers: actions.upscalerOptions(for: targets),
            trashCount: trashCount
        )
    }

    private func canAttach(using action: ((LibraryEntry) -> Void)?) -> Bool {
        guard action != nil, !scope.isTrash, targets.count == 1 else { return false }
        return targets[0].isAttachableRaster
    }
}

extension View {
    /// One print's menu, from the one list.
    func libraryMenu(_ menu: LibraryMenu) -> some View {
        rowActionMenu(menu.items, perform: menu.perform) {
            if !menu.share.isEmpty {
                ShareLink(items: menu.share) { print in SharePreview(print.filename) }
            }
        }
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
