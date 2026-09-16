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

    /// Chips offered under the search field as you type.
    ///
    /// Only what this library actually holds: a machine you have, a tag
    /// somebody used, a kind of thing that exists. Suggesting a filter that
    /// can only ever return nothing is worse than suggesting nothing.
    var suggestedTokens: [LibraryToken] {
        let typed = navigation.query.text.trimmingCharacters(in: .whitespaces)
        guard !typed.isEmpty else { return [] }
        let folded = LibraryEntry.fold(typed)
        var found: [LibraryToken] = []

        if hosts.hosts.count > 1 {
            found += hosts.hosts
                .filter { LibraryEntry.fold($0.name).contains(folded) }
                .map { .machine(id: $0.id, name: $0.name) }
        }
        found += library.tagCounts
            .filter { LibraryEntry.fold($0.name).contains(folded) }
            .prefix(5)
            .map { .tag($0.name) }
        found += PrintKind.allCases
            .filter { LibraryEntry.fold($0.rawValue).hasPrefix(folded) }
            .map { .kind($0) }
        if LibraryEntry.fold("favourite").hasPrefix(folded)
            || LibraryEntry.fold("favorite").hasPrefix(folded) {
            found.append(.favorite)
        }
        // Already-applied chips would read as "add this twice".
        let applied = Set(navigation.query.tokens.map(\.id))
        return found.filter { !applied.contains($0.id) }
    }
}
