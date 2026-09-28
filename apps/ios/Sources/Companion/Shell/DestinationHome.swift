import SwiftUI

/// One destination's root: its own `NavigationStack`, so a tab keeps its place
/// while you visit another. Until a machine is added every destination says
/// what it will hold and how to get there, in DESIGN.md §7's words -- never a
/// blank screen and never a disappearing tab.
struct DestinationHome: View {
    let destination: Destination
    @Binding var selection: TabSelection
    @Binding var showsSettings: Bool

    var body: some View {
        NavigationStack {
            content
                .navigationTitle(destination.title)
                .toolbar {
                    if destination == .machines {
                        ToolbarItem(placement: .topBarLeading) {
                            Button { showsSettings = true } label: {
                                Label("Settings", systemImage: "gearshape")
                            }
                            .keyboardShortcut(",", modifiers: .command)
                        }
                    }
                }
        }
    }

    @ViewBuilder private var content: some View {
        switch destination {
        case .generate:
            EmptyState(title: String(localized: "Add a machine to start generating"),
                       symbol: destination.symbol,
                       message: String(localized: "Mold makes pictures on a computer you own.")) {
                Button("Add a Machine…") { selection = .go(.machines) }
                    .prominentAction()
            }
        case .library:
            EmptyState(title: String(localized: "No prints yet"), symbol: destination.symbol,
                       message: String(localized: "What you generate on any machine appears here."))
        case .queue:
            EmptyState(title: String(localized: "Nothing waiting"), symbol: destination.symbol,
                       message: String(localized: "Renders you start appear here."))
        case .models:
            EmptyState(title: String(localized: "No machine to show"), symbol: destination.symbol,
                       message: String(localized: "Models belong to a machine. Add one to see what it has installed."))
        case .machines:
            EmptyState(title: String(localized: "No machines yet"), symbol: destination.symbol,
                       message: String(localized: "Mold makes pictures on a computer you own. Add one to begin."))
        }
    }
}

/// The Search tab: Library search. Tokens (`is:video`, `tag:`, `on:`) arrive
/// with the Library in M4.
struct SearchHome: View {
    @State private var query = ""

    var body: some View {
        NavigationStack {
            EmptyState(title: String(localized: "Search your prints"), symbol: "magnifyingglass",
                       message: String(localized: "Add a machine, and its prints can be found here."))
                .navigationTitle("Search")
        }
        .searchable(text: $query, prompt: Text("Prints, tags and machines"))
    }
}
