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
    /// Which print the Generate pane's draft came from, and what its own
    /// machine still holds for it. Written by `reuse(_:)` in `+Wiring`.
    @Environment(ReuseStore.self) var reuseStore
    @Environment(PrintMaterializer.self) var materializer
    /// Make Bigger… and the clip jobs it starts.
    @Environment(UpscaleStore.self) var upscales
    /// The WINDOW's undo manager. The store registers against it so Edit ▸
    /// Undo, which SwiftUI wires to the responder chain, finds our edits --
    /// and so a focused text field still keeps ⌘Z for itself.
    @Environment(\.undoManager) private var undoManager
    @Binding var destination: Destination

    @State var selection = LibraryCursor.Selection.empty
    @State var viewing: PrintID?
    /// Backs Edit ▸ Find (design S6): `.searchFocused($isSearchFocused)`
    /// below, set from `body`'s own `findAction` focused value.
    /// Deliberately not `private`: what the menu is offered is assembled in an
    /// extension in another file, and `private` does not cross that boundary.
    @FocusState var isSearchFocused: Bool
    /// Published by the inspector's Title field and "Add a tag". The search
    /// field says so through `isSearchFocused` above, which this pane owns.
    @FocusedValue(\.editingText) var editingText: Bool?
    /// Persisted, and deliberately not `private`: the toolbar button that
    /// flips it lives in an extension in another file.
    @AppStorage("libraryShowsInspector", store: AppStorageSuite.defaults)
    var showsInspector = true
    @State private var pendingDestruction: LibraryActions.Destruction?
    /// The filtered, sorted and grouped library, kept between passes. A body
    /// pass happens on every arrow key and every character typed, and re-doing
    /// all of that per pass is work proportional to the whole library for a
    /// change that moved the cursor. See `LibraryShowingCache`.
    @State private var index = LibraryShowingCache()
    /// The shelf being renamed from the MENU BAR. The sidebar row has its own;
    /// both open the same sheet.
    @State var renamingShelf: CollectionShelf?
    /// The mesh export waiting on its controls. Both doors -- the tile's menu
    /// and the viewer's -- open this one sheet.
    @State private var meshExport: MeshExportPrompt?

    var actions: LibraryActions {
        LibraryActions(hosts: hosts, library: library, reuse: reuse,
                       confirmDestruction: { pendingDestruction = $0 },
                       materializer: materializer, upscales: upscales,
                       collectionAction: { performCollection($0) },
                       meshExport: { meshExport = $0 })
    }

    // Three stages rather than one chain: what is on screen, what dresses it,
    // and what plugs it in. `showing` is derived once per DATA or QUERY change
    // and threaded down, rather than each stage re-filtering the whole library
    // -- or this pass re-doing what the last one already worked out.
    var body: some View {
        let showing = index.showing(pool: pool, revision: library.rows.value,
                                    query: resolved, selection: selection.items)
        return watched(showing)
            .focusedSceneValue(\.refreshAction) { Task { await actions.reload() } }
            .focusedSceneValue(\.inspectorToggle, InspectorToggle(isShowing: showsInspector) {
                showsInspector.toggle()
            })
            .focusedSceneValue(\.librarySelection, menuSelection(showing))
            .focusedSceneValue(\.libraryImport, menuImport)
            .focusedSceneValue(\.libraryFile, menuFile(showing))
            .focusedSceneValue(\.findAction) { isSearchFocused = true }
            .focusedSceneValue(\.thumbnailScale, ThumbnailScaleAction(edge: navigation.edge) { delta in
                navigation.edge = ThumbnailStep.apply(navigation.edge, delta: delta)
                navigation.rememberEdge()
            })
            .destructionDialog($pendingDestruction)
            .sheet(item: $renamingShelf) { ShelfNameSheet(shelf: $0) }
            .sheet(item: $meshExport) { prompt in
                MeshExportSheet(prompt: prompt) { request in
                    actions.export(prompt.entry, request: request)
                }
            }
    }

    /// The pane, plugged in: what it does on appearing, and what it re-does
    /// when the machines or the shelves change under it.
    private func watched(_ showing: LibraryShowing) -> some View {
        chrome(showing)
            // Its own data, and nothing else: `HostStore` reconciles its own
            // event streams, and the library listens from the moment it is
            // built. What still belongs here is the first LISTING, because
            // the events are deltas and a client that has read nothing has
            // nothing to apply them to.
            .task { await actions.reload() }
            // The clip upscales already running on each machine. One listing
            // per machine, so a job survives a restart and a second Mac.
            .task { await upscales.recover() }
            .onAppear { library.undo.manager = undoManager }
            .onAppear { revealIfNeeded() }
            .onChange(of: undoManager) { _, manager in library.undo.manager = manager }
            .onChange(of: navigation.scope) { _, _ in clearSelection() }
            .onChange(of: library.shelves) { _, shelves in navigation.reconcile(with: shelves) }
            // A click on an already-open Library: `.onAppear` above only
            // fires when the pane is freshly mounted.
            .onChange(of: navigation.reveal) { _, _ in revealIfNeeded() }
    }

    /// The pane, dressed: title, search, toolbar, inspector.
    private func chrome(_ showing: LibraryShowing) -> some View {
        @Bindable var navigation = navigation

        return content(showing)
            .failureBanner(hosts)
            .mediaCacheNote(materializer)
            // The OPEN print while the viewer is showing one, and the grid's
            // selection otherwise: see `LibraryInspector+Pane`.
            .trailingColumn(isShowing: showsInspector) {
                inspector(showing.inspected(viewing: viewing))
            }
            .navigationTitle(navigation.scope.title(in: library.shelves))
            .navigationSubtitle(fullSubtitle(showing))
            .searchable(text: $navigation.query.text, tokens: $navigation.query.tokens,
                        suggestedTokens: .constant(suggestedTokens),
                        prompt: "Search prompts, models and tags") { token in
                Label(token.label, systemImage: token.symbol)
            }
            .searchFocused($isSearchFocused)
            .toolbar { toolbar }
    }

    /// What is actually on screen: a print, the grid, or an explanation.
    @ViewBuilder private func content(_ showing: LibraryShowing) -> some View {
        if let viewing, let entry = entry(viewing, in: showing.visible) {
            LibraryViewer(entry: entry, host: host(of: entry), actions: actions,
                          onClose: { close(viewing) },
                          onStep: { step($0, in: showing.visible) })
        } else if showing.visible.isEmpty {
            empty(showing)
        } else {
            LibraryGrid(
                sections: showing.sections, hosts: hosts.hosts, edge: navigation.edge,
                showsHostBadges: showsHostBadges,
                scope: navigation.scope, actions: actions, entries: showing.visible,
                shelves: library.shelves, enclosingShelf: enclosingShelf,
                trashCount: library.trashed.count,
                selection: $selection, onOpen: { viewing = $0 }
            )
        }
    }
}
