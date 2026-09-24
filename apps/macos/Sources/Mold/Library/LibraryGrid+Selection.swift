import MoldClient
import SwiftUI

// What a click and a key press do to the selection. Split from the drawing
// half for size; the decisions themselves are `LibraryCursor`'s and
// `LibraryGridKeys`', both pure and tested away from any view.
extension LibraryGrid {

    func click(_ entry: LibraryEntry) {
        selection = cursor.clicking(entry.id, ClickModifiers.current, from: selection)
    }

    func perform(_ action: LibraryGridAction?) -> KeyPress.Result {
        switch action {
        case let .move(move, modifier):
            selection = cursor.moving(move, modifier, from: selection)
            return .handled
        case .open: return openLead()
        case .quickLook: return quickLookSelection()
        case .clearSelection:
            guard !selection.items.isEmpty else { return .ignored }
            selection = .empty
            return .handled
        case .trash: return trashSelection()
        case nil: return .ignored
        }
    }

    private func openLead() -> KeyPress.Result {
        guard let lead = selection.lead else { return .ignored }
        onOpen(lead)
        return .handled
    }

    private func quickLookSelection() -> KeyPress.Result {
        let targets = entries.filter { selection.items.contains($0.id) }
        guard !targets.isEmpty else { return .ignored }
        actions.quickLook(targets)
        return .handled
    }

    private func trashSelection() -> KeyPress.Result {
        let targets = entries.filter { selection.items.contains($0.id) }
        guard !targets.isEmpty else { return .ignored }
        if scope.isTrash { actions.deleteForever(targets) } else { actions.moveToTrash(targets) }
        selection = .empty
        return .handled
    }
}
