import MoldClient
import SwiftUI

// The library's toolbar. Split from the pane purely for size.
//
// What is NOT here matters: the shelf picker moved to the sidebar, where a
// collection is an ordinary row, and the machine picker became a search token,
// because filtering by machine is the same kind of act as filtering by tag and
// there is no reason for it to have its own control.
extension LibraryPane {

    @ToolbarContentBuilder var toolbar: some ToolbarContent {
        // Its own binding: `@Bindable` in `body` is local to `body`, and the
        // toolbar lives out here for size.
        @Bindable var navigation = navigation
        ToolbarItem {
            Menu {
                Picker("Sort By", selection: $navigation.query.sort) {
                    ForEach(LibrarySort.allCases, id: \.self) { order in
                        Text(order.title).tag(order)
                    }
                }
                .pickerStyle(.inline)
            } label: {
                Label("Sort", systemImage: "arrow.up.arrow.down")
            }
            .help("How the prints are ordered")
        }
        ToolbarItem {
            Slider(value: $navigation.edge, in: 88...260) { Text("Thumbnail size") }
                .frame(width: 110)
                .help("Thumbnail size")
                .onChange(of: navigation.edge) { _, _ in navigation.rememberEdge() }
        }
        ToolbarItem {
            Button { showsInspector.toggle() } label: {
                Label("Inspector", systemImage: "sidebar.trailing")
            }
            .help(showsInspector ? "Hide the inspector" : "Show the inspector")
        }
    }

    /// Chips offered under the search field as you type -- and the one a
    /// Return commits. `LibrarySearchSyntax` owns the vocabulary (`is:`,
    /// `tag:`, `on:`); this only feeds it what the library actually holds,
    /// because suggesting a filter that can only ever return nothing is
    /// worse than suggesting nothing. A machine is offered only when there
    /// is more than one to tell apart.
    var suggestedTokens: [LibraryToken] {
        LibrarySearchSyntax.suggestions(
            for: navigation.query.text, machines: searchableMachines,
            tags: library.tagCounts.map(\.name),
            applied: Set(navigation.query.tokens.map(\.id)))
    }

    /// Return over `is:video`, `tag:cat` or `on:hal9000` turns the text into
    /// its chip; over anything else it searches the words, as it always did.
    func commitTypedToken() {
        guard let token = LibrarySearchSyntax.committed(
            navigation.query.text, machines: searchableMachines,
            tags: library.tagCounts.map(\.name))
        else { return }
        navigation.query.tokens.append(token)
        navigation.query.text = ""
    }

    private var searchableMachines: [(id: MoldHost.ID, name: String)] {
        hosts.hosts.count > 1 ? hosts.hosts.map { ($0.id, $0.name) } : []
    }
}
