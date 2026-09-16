import MoldClient
import SwiftUI

/// The merged library.
///
/// Which shelf is showing comes from the sidebar, so this pane is only ever
/// asked "draw what the query selects" -- the filtering itself is
/// `LibraryQuery`, which is pure and tested away from any view.
struct LibraryPane: View {
    @Environment(HostStore.self) var hosts
    @Environment(LibraryStore.self) var library
    @Environment(LibraryNavigation.self) var navigation
    @Environment(GenerateController.self) var generate
    @Environment(ModelStore.self) var models
    @Binding var destination: Destination

    @State var selection = LibraryCursor.Selection.empty
    @State var viewing: PrintID?
    @State var showsInspector = true
    @State private var pendingDestruction: LibraryActions.Destruction?

    var actions: LibraryActions {
        LibraryActions(hosts: hosts, library: library, reuse: reuse,
                       confirmDestruction: { pendingDestruction = $0 })
    }

    var body: some View {
        @Bindable var navigation = navigation

        Group {
            if let viewing, let entry = entry(viewing) {
                LibraryViewer(entry: entry, host: host(of: entry), actions: actions,
                              onClose: { self.viewing = nil },
                              onStep: step)
            } else if visible.isEmpty {
                empty
            } else {
                LibraryGrid(
                    sections: sections, hosts: hosts.hosts, edge: navigation.edge,
                    showsHostBadges: showsHostBadges,
                    scope: navigation.scope, actions: actions, entries: visible,
                    selection: $selection, onOpen: { viewing = $0 }
                )
            }
        }
        .navigationTitle(navigation.scope.title(in: library.shelves))
        .navigationSubtitle(retentionSentence.map { "\(subtitle) · \($0)" } ?? subtitle)
        .searchable(text: $navigation.query.text, tokens: $navigation.query.tokens,
                    suggestedTokens: .constant(suggestedTokens),
                    prompt: "Search prompts, models and tags") { token in
            Label(token.label, systemImage: token.symbol)
        }
        .toolbar { toolbar }
        .inspector(isPresented: $showsInspector) {
            LibraryInspector(entries: selected, host: selected.first.flatMap(host(of:)),
                             scope: navigation.scope, actions: actions,
                             filterByTag: { navigation.query.tokens.append(.tag($0)) })
                .inspectorColumnWidth(min: 260, ideal: 320, max: 420)
        }
        .task { await actions.reload() }
        .focusedSceneValue(\.refreshAction) { Task { await actions.reload() } }
        .focusedSceneValue(\.inspectorToggle, InspectorToggle(isShowing: showsInspector) {
            showsInspector.toggle()
        })
        .onChange(of: navigation.scope) { _, _ in selection = .empty; viewing = nil }
        .onChange(of: library.shelves) { _, shelves in navigation.reconcile(with: shelves) }
        // A plain confirm with a danger button. Never a typed phrase: making
        // somebody retype a word does not make them read the sentence.
        .confirmationDialog(
            pendingDestruction?.title ?? "",
            isPresented: Binding(get: { pendingDestruction != nil },
                                 set: { if !$0 { pendingDestruction = nil } }),
            presenting: pendingDestruction
        ) { destruction in
            Button(destruction.verb, role: .destructive, action: destruction.perform)
            Button("Cancel", role: .cancel) {}
        } message: { destruction in
            Text(destruction.message)
        }
    }

    // MARK: - Content

    /// The query the grid is actually drawing: what was typed, plus the
    /// narrowing the chosen shelf adds.
    private var resolved: LibraryQuery {
        var query = navigation.query
        query.hiddenCollectionIDs = library.hiddenCollectionIDs
        if let token = navigation.scope.token(in: library.shelves) {
            query.tokens.append(token)
        }
        return query
    }

    var visible: [LibraryEntry] { resolved.apply(to: pool) }

    private var sections: [LibrarySection] { LibraryGrouping.byDay(visible) }

    var selected: [LibraryEntry] {
        visible.filter { selection.items.contains($0.id) }
    }

    /// Only worth the ink when the grid can actually be showing two machines.
    private var showsHostBadges: Bool {
        guard hosts.hosts.count > 1 else { return false }
        return !navigation.query.tokens.contains { if case .machine = $0 { true } else { false } }
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
}
