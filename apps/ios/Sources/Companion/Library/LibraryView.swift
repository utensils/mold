import MoldClient
import SwiftUI

/// The Library (DESIGN.md §5.2): every machine's prints as one day-sectioned
/// grid. The shelf comes from the title menu; search from the Search tab or a
/// pull-down; tile size from a pinch. All of it is this window's own state.
struct LibraryView: View {
    @Environment(LibraryStore.self) private var library
    @Environment(HostStore.self) private var hosts
    @Environment(AppRouter.self) private var router
    @SceneStorage("library.scope") private var storedScope = ""
    @SceneStorage("library.tile") private var tile = TileSize.medium
    @SceneStorage("library.sort") private var sort = LibrarySort.newest
    @State private var query = LibraryQuery()
    @State private var selecting = false
    @State private var managingCollections = false
    @State private var selection: Set<PrintID> = []
    @State private var showingCache = LibraryShowingCache()
    @State private var scrollPosition = LibraryScrollPosition()
    @State private var returnToPrint: PrintID?
    @Namespace private var zoom

    /// Search opens with a query already in it; the Library tab opens empty.
    var searchFocused = false

    /// Set by an iPad sidebar shelf: that shelf, with no title menu.
    var fixedScope: LibraryScope?

    private var scope: LibraryScope {
        fixedScope ?? (try? JSONDecoder().decode(LibraryScope.self, from: Data(storedScope.utf8))) ?? .all
    }

    private func setScope(_ new: LibraryScope) {
        storedScope = (try? JSONEncoder().encode(new)).flatMap { String(data: $0, encoding: .utf8) } ?? ""
        selection = []
        scrollPosition.reset()
    }

    var body: some View {
        let showing = showing()
        Group {
            if hosts.hosts.isEmpty {
                EmptyState(title: String(localized: "No prints yet"), symbol: Destination.library.symbol,
                           message: String(localized: "What you generate on any machine appears here.")) {
                    Button("Add a Machine…") { router.addMachine() }.prominentAction()
                }
            } else if showing.visible.isEmpty {
                empty
            } else {
                LibraryGrid(sections: showing.sections, tile: $tile, position: $scrollPosition,
                            returnToPrint: returnToPrint, selecting: selecting,
                            selection: $selection, trashed: scope.isTrash, zoom: zoom, visible: showing.visible)
                    // A different shelf or search is a new scroll context.
                    // Clearing the bound target alone leaves the old offset.
                    .id(LibraryGridContext(scope: scope, query: query))
            }
        }
        .navigationTitle(scope.title(in: library.shelves))
        .modifier(ShelfTitleMenu(enabled: fixedScope == nil && UIDevice.current.userInterfaceIdiom != .phone,
                                 scope: scope, choose: setScope))
        .safeAreaInset(edge: .top) {
            if fixedScope == nil && UIDevice.current.userInterfaceIdiom == .phone {
                Menu {
                    ShelfMenu(scope: scope, choose: setScope)
                } label: {
                    Label(scope.title(in: library.shelves), systemImage: scope.symbol)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .frame(minHeight: 44)
                        .contentShape(Rectangle())
                }
                .accessibilityIdentifier("library-collections")
                .padding(.horizontal, 16)
                .padding(.vertical, 8)
            }
        }
        .toolbar { toolbar(showing) }
        .sheet(isPresented: $managingCollections) {
            CollectionsSheet { scope in
                if fixedScope != nil { router.selection = .shelf(scope) } else { setScope(scope) }
            }
        }
        .modifier(LibrarySearch(enabled: searchFocused, query: $query, machines: machines, tags: tags))
        .onSubmit(of: .search) {
            // Return turns `is:video`, `tag:owls` or `on:workstation` into a
            // token when it names exactly one thing; a plain word stays text.
            if let token = LibrarySearchSyntax.committed(query.text, machines: machines, tags: tags) {
                query.tokens.append(token)
                query.text = ""
            }
        }
        .navigationDestination(for: PrintID.self) { id in
            PrintViewer(start: id, entries: showing.visible, trashed: scope.isTrash)
                .navigationTransition(.zoom(sourceID: id, in: zoom))
                .onAppear { returnToPrint = nil }
                .onDisappear { returnToPrint = id }
        }
        .safeAreaInset(edge: .bottom) {
            if selecting { SelectionBar(scope: scope, selected: showing.selected) { selection = [] } }
        }
        .refreshable { await library.reload() }
        // iPad: a picture dropped on the grid joins the Default machine's Library.
        .dropDestination(for: Data.self) { items, _ in
            guard !scope.isTrash, let host = hosts.preferredHost, !items.isEmpty else { return false }
            Task {
                for (index, data) in items.enumerated() {
                    await library.importPicture(data, stem: "dropped-\(Int(Date.now.timeIntervalSince1970))-\(index)", to: host)
                }
            }
            return true
        }
        .overlay(alignment: .top) {
            VStack(spacing: 4) {
                FailureBanner()
                OfflineNote()
            }
        }
        .onChange(of: sort) { query.sort = sort }
        .onChange(of: query) { scrollPosition.reset() }
        .onAppear { query.sort = sort }
    }

    private var machines: [(id: MoldHost.ID, name: String)] { hosts.hosts.map { ($0.id, $0.name) } }
    private var tags: [String] { library.knownTags }

    private func showing() -> LibraryShowing {
        let narrowed = scope.resolve(query, shelves: library.shelves,
                                     hiddenCollectionIDs: library.hiddenCollectionIDs)
        return showingCache.showing(pool: scope.isTrash ? library.trashPool : library.pool,
                                    revision: library.revision, query: narrowed, selection: selection)
    }

    @ViewBuilder private var empty: some View {
        if query.isNarrowed {
            EmptyState(title: String(localized: "No results"), symbol: "magnifyingglass",
                       message: String(localized: "Nothing here matches what you're looking for."))
        } else {
            switch scope {
            case .trash:
                EmptyState(title: String(localized: "Nothing deleted"), symbol: "trash",
                           message: String(localized: "Prints you delete stay here until their machine removes them for good."))
            case .favorites:
                EmptyState(title: String(localized: "No favourites yet"), symbol: "star",
                           message: String(localized: "Tap the star on a print to keep it here."))
            case .collection:
                EmptyState(title: String(localized: "Nothing filed here yet"), symbol: "rectangle.stack",
                           message: String(localized: "Choose Add to Collection on a print to file it here."))
            case .all:
                EmptyState(title: String(localized: "No prints yet"), symbol: Destination.library.symbol,
                           message: String(localized: "What you generate on any machine appears here."))
            }
        }
    }

    @ToolbarContentBuilder private func toolbar(_ showing: LibraryShowing) -> some ToolbarContent {
        // Hidden, not disabled, when there is nothing to select: a greyed
        // button failed the contrast audit and offers nothing anyway.
        if !showing.visible.isEmpty || selecting {
            ToolbarItem(placement: .topBarTrailing) {
                Button(selecting ? String(localized: "Done") : String(localized: "Select")) {
                    selecting.toggle()
                    if !selecting { selection = [] }
                }
            }
        }
        ToolbarItem(placement: .topBarTrailing) {
            Menu {
                if !library.shelves.isEmpty {
                    Button("Manage Collections…", systemImage: "rectangle.stack") { managingCollections = true }
                    Divider()
                }
                Picker("Sort By", selection: $sort) {
                    ForEach(LibrarySort.allCases, id: \.self) { Text($0.title).tag($0) }
                }
                Picker("Tile Size", selection: $tile) {
                    ForEach(TileSize.allCases) { Text($0.title).tag($0) }
                }
                Button("Larger Tiles") { tile = tile.stepped(bigger: true) }
                    .keyboardShortcut("+", modifiers: .command)
                    .disabled(tile == .large)
                Button("Smaller Tiles") { tile = tile.stepped(bigger: false) }
                    .keyboardShortcut("-", modifiers: .command)
                    .disabled(tile == .small)
                if scope.isTrash, !library.trashPool.isEmpty {
                    Divider()
                    EmptyTrashButton()
                }
            } label: {
                Label("View Options", systemImage: "ellipsis")
            }
        }
    }
}

private struct LibraryGridContext: Hashable {
    let scope: LibraryScope
    let query: LibraryQuery
}

/// Search lives on the Search tab (the iOS 26 search role), with the system's
/// own placement; the Library tab carries no second, collapsed search drawer.
private struct LibrarySearch: ViewModifier {
    let enabled: Bool
    @Binding var query: LibraryQuery
    let machines: [(id: MoldHost.ID, name: String)]
    let tags: [String]

    func body(content: Content) -> some View {
        if enabled {
            content
                .searchable(text: $query.text, tokens: $query.tokens) { token in
                    Label(token.label, systemImage: token.symbol)
                }
                .searchSuggestions { SearchSuggestions(query: $query, machines: machines, tags: tags) }
        } else {
            content
        }
    }
}

/// The shelf switcher in the title -- only where the sidebar does not
/// already list the shelves.
private struct ShelfTitleMenu: ViewModifier {
    let enabled: Bool
    let scope: LibraryScope
    let choose: (LibraryScope) -> Void

    func body(content: Content) -> some View {
        if enabled {
            content.toolbarTitleMenu { ShelfMenu(scope: scope, choose: choose) }
        } else {
            content
        }
    }
}
