import MoldClient
import SwiftUI

/// The Library (DESIGN.md §5.2): every machine's prints as one day-sectioned
/// grid. The shelf comes from the title menu; search from the Search tab or a
/// pull-down; tile size from a pinch. All of it is this window's own state.
struct LibraryView: View {
    @Environment(LibraryStore.self) private var library
    @Environment(HostStore.self) private var hosts
    @SceneStorage("library.scope") private var storedScope = ""
    @SceneStorage("library.tile") private var tile = TileSize.medium
    @SceneStorage("library.sort") private var sort = LibrarySort.newest
    @State private var query = LibraryQuery()
    @State private var selecting = false
    @State private var selection: Set<PrintID> = []
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
    }

    var body: some View {
        let showing = showing()
        Group {
            if hosts.hosts.isEmpty {
                EmptyState(title: String(localized: "No prints yet"), symbol: Destination.library.symbol,
                           message: String(localized: "What you generate on any machine appears here."))
            } else if showing.visible.isEmpty {
                empty
            } else {
                LibraryGrid(sections: showing.sections, tile: $tile, selecting: selecting,
                            selection: $selection, trashed: scope.isTrash, zoom: zoom, visible: showing.visible)
            }
        }
        .navigationTitle(scope.title(in: library.shelves))
        .modifier(ShelfTitleMenu(enabled: fixedScope == nil, scope: scope, choose: setScope))
        .toolbar { toolbar(showing) }
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
        .overlay(alignment: .top) { FailureBanner() }
        .onChange(of: sort) { query.sort = sort }
        .onAppear { query.sort = sort }
    }

    private var machines: [(id: MoldHost.ID, name: String)] { hosts.hosts.map { ($0.id, $0.name) } }
    private var tags: [String] { Array(Set(library.pool.flatMap(\.print.tagList))).sorted() }

    private func showing() -> LibraryShowing {
        var narrowed = query
        if let token = scope.token(in: library.shelves), !narrowed.tokens.contains(token) {
            narrowed.tokens.append(token)
        }
        return LibraryShowing(pool: scope.isTrash ? library.trashPool : library.pool,
                              query: narrowed, selection: selection)
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
                Picker("Sort By", selection: $sort) {
                    ForEach(LibrarySort.allCases, id: \.self) { Text($0.title).tag($0) }
                }
                Picker("Tile Size", selection: $tile) {
                    ForEach(TileSize.allCases) { Text($0.title).tag($0) }
                }
                Button("Larger Tiles") { tile = tile.step(1) }
                    .keyboardShortcut("+", modifiers: .command)
                    .disabled(tile == .large)
                Button("Smaller Tiles") { tile = tile.step(-1) }
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

/// The three tile sizes a pinch snaps between; minimum widths scale with text.
enum TileSize: String, CaseIterable, Identifiable {
    case small, medium, large
    var id: Self { self }

    var title: String {
        switch self {
        case .small: String(localized: "Small")
        case .medium: String(localized: "Medium")
        case .large: String(localized: "Large")
        }
    }

    /// One size up (1) or down (-1), stopping at the ends.
    func step(_ by: Int) -> TileSize {
        let all = Self.allCases
        let index = all.firstIndex(of: self)! + by
        return all.indices.contains(index) ? all[index] : self
    }

    /// Points at Large text; `@ScaledMetric` in the grid grows them with it.
    var basePoints: CGFloat {
        switch self {
        case .small: 96
        case .medium: 128
        case .large: 180
        }
    }

    func stepped(bigger: Bool) -> TileSize {
        let all = Self.allCases
        let index = all.firstIndex(of: self) ?? 1
        return all[max(0, min(all.count - 1, index + (bigger ? 1 : -1)))]
    }
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
