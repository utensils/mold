import MoldClient
import SwiftUI

/// A print opened from a widget or a notification: the Library's own viewer,
/// full screen, once the Library has it -- or a sentence when it is gone.
struct LinkedPrint: View {
    @Environment(\.dismiss) private var dismiss
    @Environment(LibraryStore.self) private var library
    let id: PrintID
    @State private var looked = false

    var body: some View {
        NavigationStack {
            Group {
                if let entry = library.pool.first(where: { $0.everyCopy.contains { $0.id == id } }) {
                    PrintViewer(start: entry.id, entries: [entry], trashed: false)
                } else if looked {
                    EmptyState(title: String(localized: "Not in the Library"), symbol: Destination.library.symbol,
                               message: String(localized: "That print was deleted, or its machine isn't answering."))
                } else {
                    ProgressView()
                }
            }
            .toolbar { ToolbarItem(placement: .cancellationAction) { Button("Done") { dismiss() } } }
        }
        .task {
            await library.reload(id.host)
            looked = true
        }
    }
}
