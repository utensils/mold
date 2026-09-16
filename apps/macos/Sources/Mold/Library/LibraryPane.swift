import MoldClient
import SwiftUI

/// The merged library.
struct LibraryPane: View {
    @Environment(HostStore.self) var hosts
    @Environment(LibraryStore.self) var library
    @Environment(GenerateController.self) var generate
    @Environment(ModelStore.self) var models
    @Binding var destination: Destination

    @State var scope: LibraryScope = .all
    @State var sourceHost: MoldHost.ID?
    @State var query = ""
    @State var edge: CGFloat = 132
    @State var selection = LibraryCursor.Selection.empty
    @State var viewing: PrintID?

    private var actions: LibraryActions {
        LibraryActions(hosts: hosts, library: library, reuse: reuse)
    }

    /// Seeds the Generate pane from a finished print and goes there.
    ///
    /// The model is adopted from the machine that MADE the print, because a
    /// model installed on one host is not available on another.
    private func reuse(_ entry: LibraryEntry) {
        generate.draft = RenderDraft(reusing: entry.print.metadata)
        if let name = entry.print.metadata.model,
           let model = models.model(named: name, on: entry.hostID) {
            generate.adopt(model: model, on: entry.hostID, keepingDraft: true)
        }
        destination = .generate
    }

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
                             scope: scope, actions: actions,
                             filterByTag: { query = $0 })
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
}
