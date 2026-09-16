import MoldClient
import SwiftUI

/// The merged library.
struct LibraryPane: View {
    @Environment(HostStore.self) private var hosts
    @Environment(LibraryStore.self) private var library

    @State private var scope: LibraryScope = .all
    @State private var sourceHost: MoldHost.ID?
    @State private var query = ""
    @State private var edge: CGFloat = 132
    @State private var selection = LibraryCursor.Selection.empty
    @State private var viewing: PrintID?

    private var actions: LibraryActions { LibraryActions(hosts: hosts, library: library) }

    var body: some View {
        Group {
            if let viewing, let entry = entry(viewing) {
                LibraryViewer(entry: entry, host: host(of: entry), actions: actions,
                              onClose: { self.viewing = nil },
                              onStep: step)
            } else if visible.isEmpty {
                empty
            } else {
                LibraryGrid(
                    sections: sections, hosts: hosts.hosts, edge: edge,
                    showsHostBadges: sourceHost == nil && hosts.hosts.count > 1,
                    scope: scope, actions: actions, entries: visible,
                    selection: $selection, onOpen: { viewing = $0 }
                )
            }
        }
        .navigationTitle("Library")
        .navigationSubtitle(subtitle)
        .searchable(text: $query, prompt: "Search prompts and models")
        .toolbar { toolbar }
        .inspector(isPresented: .constant(viewing == nil && !selected.isEmpty)) {
            LibraryInspector(entries: selected, host: selected.first.flatMap(host(of:)),
                             scope: scope, actions: actions)
                .inspectorColumnWidth(min: 260, ideal: 320, max: 420)
        }
        .task { await actions.reload() }
        .focusedSceneValue(\.refreshAction) { Task { await actions.reload() } }
        .onChange(of: scope) { _, _ in selection = .empty; viewing = nil }
    }

    // MARK: - Content

    private var pool: [LibraryEntry] {
        scope.isTrash ? library.trashed : library.items
    }

    private var visible: [LibraryEntry] {
        pool.filter { entry in
            if scope == .favorites, !entry.print.isFavorite { return false }
            guard sourceHost == nil || entry.hostID == sourceHost else { return false }
            return query.isEmpty || entry.matches(query)
        }
    }

    private var sections: [LibrarySection] { LibraryGrouping.byDay(visible) }

    private var selected: [LibraryEntry] {
        visible.filter { selection.items.contains($0.id) }
    }

    private func entry(_ id: PrintID) -> LibraryEntry? { visible.first { $0.id == id } }

    private func host(of entry: LibraryEntry) -> MoldHost? {
        hosts.hosts.first { $0.id == entry.hostID }
    }

    /// Walks the viewer through the list the grid is showing.
    private func step(_ delta: Int) {
        guard let viewing, let index = visible.firstIndex(where: { $0.id == viewing })
        else { return }
        let next = min(max(index + delta, 0), visible.count - 1)
        self.viewing = visible[next].id
    }

    private var subtitle: String {
        let shown = visible.count
        let total = pool.count
        let noun = scope.isTrash ? "in the trash" : "prints"
        return shown == total
            ? "\(total.formatted()) \(noun)"
            : "\(shown.formatted()) of \(total.formatted()) \(noun)"
    }

    @ViewBuilder private var empty: some View {
        if library.isLoading {
            ProgressView("Loading prints…")
        } else if let failure = library.failures.values.compactMap(\.self).first {
            ContentUnavailableView("Can't load the library", systemImage: "exclamationmark.triangle",
                                   description: Text(failure))
        } else if !query.isEmpty {
            ContentUnavailableView.search(text: query)
        } else {
            ContentUnavailableView(scope.title, systemImage: scope.symbol,
                                   description: Text(emptyMessage))
        }
    }

    private var emptyMessage: String {
        switch scope {
        case .all: "Prints from every machine appear here."
        case .favorites: "Stars you add show up here."
        case .trash: "Deleted prints wait here until their machine purges them."
        }
    }

    // MARK: - Toolbar

    @ToolbarContentBuilder private var toolbar: some ToolbarContent {
        ToolbarItem {
            Picker("Shelf", selection: $scope) {
                ForEach(LibraryScope.allCases) { shelf in
                    Label(shelf.title, systemImage: shelf.symbol).tag(shelf)
                }
            }
            .pickerStyle(.segmented)
            .labelStyle(.iconOnly)
            .help("All prints, favorites, or the trash")
        }
        ToolbarItem {
            Picker("Source", selection: $sourceHost) {
                Text("All machines").tag(MoldHost.ID?.none)
                ForEach(hosts.hosts) { host in
                    Text("\(host.name) (\(library.count(for: host.id)))")
                        .tag(MoldHost.ID?.some(host.id))
                }
            }
        }
        ToolbarItem {
            Slider(value: $edge, in: 88...260) { Text("Thumbnail size") }
                .frame(width: 110)
                .help("Thumbnail size")
        }
    }
}
