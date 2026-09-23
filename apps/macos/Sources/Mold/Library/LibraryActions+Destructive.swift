import MoldClient
import SwiftUI

// The actions that cannot be undone. Split from the rest for size, and
// because asking a question first is their whole distinguishing feature.
@MainActor
extension LibraryActions {
    /// Asks, then destroys. There is no undo on the host side and none here.
    func deleteForever(_ entries: [LibraryEntry]) {
        guard !entries.isEmpty, let ask = confirmDestruction else { return }
        let noun = entries.count == 1
            ? "“\(entries[0].print.title ?? entries[0].print.filename)”"
            : "\(entries.count.formatted()) prints"
        let locations = Set(entries.map(\.hostName)).sorted().joined(separator: ", ")
        ask(Destruction(
            title: "Delete \(noun) immediately?",
            message: "This cannot be undone. Delete from \(locations). "
                + "Copies on other machines remain in their Libraries.",
            verb: "Delete Immediately"
        ) {
            Task {
                await library.deleteForever(entries)
                await library.refreshTrash()
            }
        })
    }

    /// Empties every machine's trash at once.
    func emptyTrash() {
        let waiting = library.trashed.count
        guard waiting > 0, let ask = confirmDestruction else { return }
        ask(Destruction(
            title: "Empty the Trash?",
            message: "\(waiting.formatted()) "
                + (waiting == 1 ? "print" : "prints")
                + " will be deleted from every machine. This cannot be undone.",
            verb: "Empty Trash"
        ) {
            Task { await library.emptyTrash() }
        })
    }
}
