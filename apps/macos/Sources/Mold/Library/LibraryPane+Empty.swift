import MoldClient
import SwiftUI

// What the pane says when there is nothing to draw, and what it says about
// what there is. Split from the pane for size.
extension LibraryPane {

    func subtitle(_ showing: LibraryShowing) -> String {
        // Transfer progress belongs to LibraryActivityStatus, which observes it
        // without invalidating the grid or its tracked context menu.
        if showing.selected.count > 1 { return "\(showing.selected.count.formatted()) selected" }
        let shown = showing.visible.count
        let baseline = navigation.scope.baselineCount(in: showing.pool,
            machines: navigation.query.machineIDs, shelves: library.shelves,
            hiddenIDs: library.hiddenCollectionIDs)
        let counted = shown == baseline ? shown : baseline
        let noun = navigation.scope.isTrash ? "in the trash" : (counted == 1 ? "print" : "prints")
        guard shown != baseline else { return "\(shown.formatted()) \(noun)" }
        return "\(shown.formatted()) of \(baseline.formatted()) \(noun)"
    }

    var pool: [LibraryEntry] {
        navigation.scope.isTrash ? library.trashed : library.items
    }

    @ViewBuilder func empty(_ showing: LibraryShowing) -> some View {
        if library.isLoading, library.items.isEmpty {
            ProgressView("Loading prints…")
        } else if let slug = navigation.scope.collectionSlug,
                  let shelf = library.shelf(slug: slug),
                  shelf.presence(on: navigation.query.machineIDs,
                    available: library.collectionInventoryAvailable.intersection(Set(hosts.hosts.filter(hosts.isUp).map(\.id)))) != .present {
            let unavailable = shelf.presence(on: navigation.query.machineIDs,
                available: library.collectionInventoryAvailable.intersection(Set(hosts.hosts.filter(hosts.isUp).map(\.id)))) == .unavailable
            ContentUnavailableView(unavailable ? "Collection Unavailable" : "Collection Not on This Machine",
                systemImage: "rectangle.stack",
                description: Text(unavailable ? "Connect to the selected machine to confirm its collections. Saved prints remain available." : "This collection exists on other machines. File prints here to create its copy on this machine."))
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
