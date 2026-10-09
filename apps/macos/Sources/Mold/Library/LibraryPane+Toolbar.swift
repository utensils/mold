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
                Image(systemName: "server.rack")
                    .accessibilityLabel("Machine: \(machineFilterTitle)")
            }
            .help("Show prints from one machine (\(machineFilterTitle))")
        }
        ToolbarItem {
            Picker("Media Type", selection: Binding(get: {
                LibraryMediaFilter.selected(in: navigation.query)
            }, set: { filter in
                guard let filter else { return }
                navigation.query = filter.applying(to: navigation.query)
                clearSelection()
            })) {
                ForEach(LibraryMediaFilter.allCases) { filter in Text(filter.title).tag(Optional(filter)) }
            }
            .help("Show all media, photos, videos or 3D in this shelf")
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
            .help("Choose the order of prints in this shelf")
        }
        ToolbarItem {
            Button {
                guard library.localSaveTask == nil else { return }
                library.localSaveTask = Task { await library.syncAllLocally() }
            } label: {
                Label("Sync All to This Mac", systemImage: "arrow.down.to.line.compact")
            }
            .help("Copy all remote Library prints and collections to This Mac, including clips and 3D prints")
            .disabled(library.localSaveTask != nil
                || !hosts.hosts.contains { $0.id != MoldEngine.localHostID })
        }
        ToolbarItem {
            Slider(value: $navigation.edge, in: 88...260) { Text("Thumbnail size") }
                .frame(width: 110)
                .help("Make the Library thumbnails larger or smaller")
                .onChange(of: navigation.edge) { _, _ in navigation.rememberEdge() }
        }
        if navigation.scope.isTrash {
            ToolbarItemGroup {
                Button("Put Back", systemImage: "arrow.uturn.backward") {
                    actions.restore(trashToolbarTargets)
                }
                .disabled(trashToolbarTargets.isEmpty || library.isBulkBusy)
                .help("Restore the selected copies on the displayed machines")
                Button("Delete Immediately…", systemImage: "trash.slash") {
                    actions.deleteForever(trashToolbarTargets)
                }
                .disabled(trashToolbarTargets.isEmpty || library.isBulkBusy)
                .help("Permanently delete the selected copies on the displayed machines")
                Button("Empty Trash…", systemImage: "trash") { actions.emptyTrash() }
                    .disabled(actions.trashEntries.isEmpty || library.isBulkBusy)
                    .help("Empty Trash on \(machineFilterTitle)")
            }
        }
        // No inspector switch here: it belongs over the column it opens,
        // and `trailingColumn` is what knows where that is.
    }

    var trashToolbarTargets: [LibraryEntry] {
        let visible = resolved.apply(to: pool)
        if let viewing, let entry = entry(viewing, in: visible) { return [entry] }
        return visible.filter { selection.items.contains($0.id) }
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
