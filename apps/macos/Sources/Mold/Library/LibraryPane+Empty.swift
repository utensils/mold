import MoldClient
import SwiftUI

// What the pane says when there is nothing to draw, and what it says about
// what there is. Split from the pane for size.
extension LibraryPane {

    var subtitle: String {
        if selected.count > 1 { return "\(selected.count.formatted()) selected" }
        let shown = visible.count
        let noun = navigation.scope.isTrash ? "in the trash" : "prints"
        guard navigation.query.isNarrowed else { return "\(shown.formatted()) \(noun)" }
        return "\(shown.formatted()) of \(pool.count.formatted()) \(noun)"
    }

    var pool: [LibraryEntry] {
        navigation.scope.isTrash ? library.trashed : library.items
    }

    @ViewBuilder var empty: some View {
        if library.isLoading, library.items.isEmpty {
            ProgressView("Loading prints…")
        } else if let failure = library.failures.values.compactMap(\.self).first {
            ContentUnavailableView("Can't load the library",
                                   systemImage: "exclamationmark.triangle",
                                   description: Text(failure))
        } else if navigation.query.isNarrowed {
            // A narrowed library that shows nothing is a search result, not an
            // empty shelf -- and the way out is to widen, not to make a print.
            ContentUnavailableView {
                Label("No Matches", systemImage: "magnifyingglass")
            } description: {
                Text("Nothing here matches what you are looking for.")
            } actions: {
                Button("Clear Search") {
                    navigation.query.text = ""
                    navigation.query.tokens = []
                }
            }
        } else {
            ContentUnavailableView(navigation.scope.title(in: library.shelves),
                                   systemImage: navigation.scope.symbol,
                                   description: Text(navigation.scope.emptyMessage))
        }
    }

    /// What the trash promises, in the machines' own terms.
    var retentionSentence: String? {
        guard navigation.scope.isTrash else { return nil }
        return TrashRetention.sentence(for: hosts.hosts, capabilities: hosts.capabilities)
    }
}
