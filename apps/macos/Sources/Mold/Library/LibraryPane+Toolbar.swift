import MoldClient
import SwiftUI

// The library's toolbar. Split from the pane purely for size.
//
// The shelf picker lives in the sidebar. The machine picker is a visible
// shortcut for the same search token that `on:machine` creates.
extension LibraryPane {

    @ToolbarContentBuilder var toolbar: some ToolbarContent {
        // Its own binding: `@Bindable` in `body` is local to `body`, and the
        // toolbar lives out here for size.
        @Bindable var navigation = navigation
        ToolbarItem {
            Menu {
                Button {
                    chooseMachine(nil)
                } label: {
                    if selectedMachines.isEmpty { Label("All Machines", systemImage: "checkmark") }
                    else { Text("All Machines") }
                }
                ForEach(hosts.hosts) { host in
                    Button {
                        chooseMachine(host)
                    } label: {
                        if selectedMachines.contains(host.id) {
                            Label(host.name, systemImage: "checkmark")
                        } else { Text(host.name) }
                    }
                }
            } label: {
                HStack(spacing: 5) {
                    Image(systemName: "server.rack")
                    Text(machineFilterTitle)
                        .lineLimit(1)
                        .frame(maxWidth: 110)
                }
            }
            .help("Show prints from one machine")
        }
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
        // No inspector switch here: it belongs over the column it opens,
        // and `trailingColumn` is what knows where that is.
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
            tags: library.tags.counts.map(\.name),
            applied: Set(navigation.query.tokens.map(\.id)))
    }

    /// Return over `is:video`, `tag:cat` or `on:hal9000` turns the text into
    /// its chip; over anything else it searches the words, as it always did.
    func commitTypedToken() {
        guard let token = LibrarySearchSyntax.committed(
            navigation.query.text, machines: searchableMachines,
            tags: library.tags.counts.map(\.name))
        else { return }
        if case let .machine(id, name) = token {
            chooseMachine(hosts.host(id))
            // A host can disappear during search completion. Keep the token's
            // own name if that happened; it still describes what was typed.
            if hosts.host(id) == nil { navigation.query.tokens.append(.machine(id: id, name: name)) }
        } else {
            navigation.query.tokens.append(token)
        }
        navigation.query.text = ""
    }

    var selectedMachines: Set<MoldHost.ID> {
        Set(navigation.query.tokens.compactMap { token -> MoldHost.ID? in
            if case let .machine(id, _) = token { return id }
            return nil
        })
    }

    var machineFilterTitle: String {
        switch selectedMachines.count {
        case 0: "All Machines"
        case 1: selectedMachines.first.flatMap(hosts.name(of:)) ?? "One Machine"
        default: "\(selectedMachines.count) Machines"
        }
    }

    func chooseMachine(_ host: MoldHost?) {
        navigation.query.tokens.removeAll {
            if case .machine = $0 { return true }
            return false
        }
        if let host { navigation.query.tokens.append(.machine(id: host.id, name: host.name)) }
        clearSelection()
    }

    private var searchableMachines: [(id: MoldHost.ID, name: String)] {
        hosts.hosts.count > 1 ? hosts.hosts.map { ($0.id, $0.name) } : []
    }
}
