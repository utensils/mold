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
        // Every copy: a merged print is deleted from every machine holding
        // it, so the sentence has to name every one of them.
        let machines = Dictionary(entries.flatMap(\.everyCopy).map { ($0.hostID, $0.hostName) },
                                  uniquingKeysWith: { first, _ in first })
        let names = Dictionary(grouping: machines.values, by: { $0 }).mapValues(\.count)
        let locations = machines.map { id, name in
            guard names[name, default: 0] > 1 else { return name }
            let address = hosts.host(id)?.baseURL.host ?? "unknown address"
            return "\(name) (\(address), \(id.uuidString.prefix(8)))"
        }.sorted().joined(separator: ", ")
        ask(Destruction(
            title: "Delete \(noun) immediately?",
            message: "This cannot be undone. Delete from \(locations).",
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
