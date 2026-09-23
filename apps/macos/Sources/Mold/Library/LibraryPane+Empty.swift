import MoldClient
import SwiftUI

// What the pane says when there is nothing to draw, and what it says about
// what there is. Split from the pane for size.
extension LibraryPane {

    func subtitle(_ showing: LibraryShowing) -> String {
        if let progress = library.localSaveProgress { return progress }
        if showing.selected.count > 1 { return "\(showing.selected.count.formatted()) selected" }
        let shown = showing.visible.count
        // "1 prints" is the tell of a string built by concatenation. The noun
        // agrees with the LAST number in the sentence, which is the pool's
        // when the query has narrowed one count out of another; the trash
        // reads "in the trash" either way, being a phrase and not a count.
        let counted = navigation.query.isNarrowed ? showing.pool.count : shown
        let noun = navigation.scope.isTrash ? "in the trash" : (counted == 1 ? "print" : "prints")
        guard navigation.query.isNarrowed else { return "\(shown.formatted()) \(noun)" }
        return "\(shown.formatted()) of \(showing.pool.count.formatted()) \(noun)"
    }

    var pool: [LibraryEntry] {
        navigation.scope.isTrash ? library.trashed : library.items
    }

    @ViewBuilder func empty(_ showing: LibraryShowing) -> some View {
        if library.isLoading, library.items.isEmpty {
            ProgressView("Loading prints…")
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
