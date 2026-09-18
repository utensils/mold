import MoldClient
import SwiftUI

// One place turns a `LibraryAction` into the call that performs it, so the
// tile's menu and the menu bar's cannot mean different things by the same
// item -- which is the other half of both drawing one plan.
@MainActor
extension LibraryActions {

    func perform(_ action: LibraryAction, on targets: [LibraryEntry],
                 scope: LibraryScope, open: (() -> Void)? = nil) {
        switch action {
        case .open:
            open?()
        case .reuse:
            if let entry = targets.first { reuse?(entry) }
        case .quickLook:
            quickLook(targets)
        case let .favorite(on):
            library.setFavorite(on, on: targets)
        case let .file(slug):
            if let shelf = library.shelf(slug: slug) { library.file(targets, into: shelf) }
        case let .unfile(slug):
            if let shelf = library.shelf(slug: slug) { library.unfile(targets, from: shelf) }
        case let .upscale(model):
            upscale(targets, using: model)
        case .copy:
            copy(targets)
        case .save:
            save(targets)
        case let .export(format):
            if let entry = targets.first { requestExport(entry, as: format) }
        case .exportTurntable:
            if let entry = targets.first { requestTurntable(entry) }
        case .trash:
            moveToTrash(targets)
        case .putBack:
            restore(targets)
        case .deleteForever:
            deleteForever(targets)
        case .emptyTrash:
            emptyTrash()
        case .renameCollection, .setCollectionHidden, .deleteCollection:
            // The shelf's own three. They are declared in the plan so both
            // menus offer them, and answered by whoever owns the sheet and the
            // confirm -- the sidebar row, or the pane's own handler.
            collectionAction?(action)
        }
    }
}
