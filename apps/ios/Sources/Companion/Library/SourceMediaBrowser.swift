import MoldClient
import SwiftUI

/// This container observes library changes; scan progress and typing only redraw the content.
struct SourceMediaBrowser: View {
    @Environment(HostStore.self) private var hosts
    @Environment(LibraryStore.self) private var library
    @Environment(AppRouter.self) private var router
    @State private var query = LibraryQuery()
    @State private var scope = LibraryScope.all
    @State private var kind = SourceMediaFilter.all
    @State private var projection = SourceMediaProjection()
    @SceneStorage("sourceMedia.tile") private var tile = TileSize.medium
    @ScaledMetric(relativeTo: .body) private var scale: CGFloat = 1
    @State private var pinchStart: TileSize?
    private var edge: CGFloat { tile.basePoints * scale }
    @State private var browserID = UUID()
    var body: some View {
        let resolved: LibraryQuery = {
            var hostQuery = query
            if let id = library.machineID, let host = hosts.host(id) {
                hostQuery.tokens.removeAll { if case .machine = $0 { true } else { false } }
                hostQuery.tokens.append(.machine(id: id, name: host.name))
            }
            return scope.resolve(hostQuery, shelves: library.shelves, hiddenCollectionIDs: [:])
        }()
        let owners = projection.owners(in: library.pool, query: resolved, revision: library.sourceAccessRevision)
        let routes = Dictionary(uniqueKeysWithValues: hosts.hosts.map { ($0.id, $0) })
        let instances = Dictionary(uniqueKeysWithValues: hosts.hosts.map { ($0.id, hosts.instanceID(of: $0.id)) })
        let allowed = Dictionary(owners.map { ($0.id, SourceMediaAuthority(owner: $0, route: routes[$0.hostID], instance: instances[$0.hostID] ?? nil)) }, uniquingKeysWith: { first, _ in first })
        SourceMediaContent(owners: owners, routes: routes, instances: instances, allowed: allowed, catalog: library.sourceMediaCatalog,
                           openOwner: { router.openedPrint = .init(id: $0) }, search: query.text, kind: kind, queryKinds: Set(query.tokens.compactMap { if case let .kind(value) = $0 { value } else { nil } }), sort: query.sort, edge: edge)
            .onAppear { library.openSourceBrowsers.insert(browserID) }
            .onDisappear { library.openSourceBrowsers.remove(browserID) }
            .navigationTitle("Source Media")
            .searchable(text: $query.text, tokens: $query.tokens, prompt: "Search, or is:video · tag:name · on:machine") { token in
                Label(token.label, systemImage: token.symbol)
            }
            .searchSuggestions { SearchSuggestions(query: $query, machines: hosts.hosts.map { ($0.id, $0.name) }, tags: library.knownTags) }
            .onSubmit(of: .search) { commitSearch() }
            .onChange(of: query.tokens) { _, tokens in
                if tokens.contains(where: { if case .kind = $0 { true } else { false } }) { kind = .all }
            }
            .toolbar { filters }
            .gesture(PinchRecognizer(changed: { magnification in
                if pinchStart == nil { pinchStart = tile }
                tile = TileSize.pinched(from: pinchStart ?? tile, magnification: magnification, current: tile)
            }, ended: { pinchStart = nil }))
    }

    private var mediaChoice: Binding<SourceMediaFilter> {
        Binding(get: { kind }, set: { value in
            query.tokens.removeAll { if case .kind = $0 { true } else { false } }
            kind = value
        })
    }
    private func commitSearch() {
        guard let token = LibrarySearchSyntax.committed(query.text,
            machines: hosts.hosts.map { ($0.id, $0.name) }, tags: library.knownTags) else { return }
        query.tokens.append(token)
        query.text = ""
    }

    @ToolbarContentBuilder private var filters: some ToolbarContent {
        ToolbarItem(placement: .topBarTrailing) {
            Menu {
                Picker("Collection", selection: $scope) {
                    Text("All Collections").tag(LibraryScope.all)
                    Text("Favourites").tag(LibraryScope.favorites)
                    ForEach(library.shelves) { shelf in
                        Text(shelf.name + (shelf.hidden ? " (Hidden)" : ""))
                            .tag(LibraryScope.collection(slug: shelf.slug))
                    }
                }.pickerStyle(.inline)

                Divider()
                LibraryMachinePicker()
                Divider()
                Picker("Media Type", selection: mediaChoice) {
                    ForEach(SourceMediaFilter.allCases) { choice in Text(choice.title).tag(choice) }
                }.pickerStyle(.inline)
                Divider()

                Picker("Sort By", selection: $query.sort) {
                    ForEach(LibrarySort.allCases, id: \.self) { order in Text(order.title).tag(order) }
                }.pickerStyle(.inline)

                Picker("Tile Size", selection: $tile) {
                    ForEach(TileSize.allCases) { choice in Text(choice.title).tag(choice) }
                }
            } label: { Label("View Options", systemImage: "ellipsis") }
        }
    }
}

private struct SourceMediaContent: View {
    @Environment(HostStore.self) private var hosts
    let owners: [LibraryEntry]
    let routes: [UUID: MoldHost]
    let instances: [UUID: String?]
    let allowed: [PrintID: SourceMediaAuthority]
    let catalog: SourceMediaCatalog
    let openOwner: (PrintID) -> Void
    let search: String
    let kind: SourceMediaFilter
    let queryKinds: Set<PrintKind>
    let sort: LibrarySort
    let edge: CGFloat
    @State private var retry = 0
    @State private var refreshed = 0
    var body: some View {
        let visible = SourceMediaDisplay.filter(catalog.sources, allowed: allowed, kind: kind, search: search, queryKinds: queryKinds, sort: sort)
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 20) {
                HStack {
                    Text("\(visible.count) \(visible.count == 1 ? "source" : "sources") · \(catalog.completed) of \(owners.count) prints checked")
                        .font(.caption).foregroundStyle(.secondaryText)
                    if catalog.loading { ProgressView().controlSize(.small) }
                }
                if !catalog.failures.isEmpty {
                    DisclosureGroup("Sources unavailable for \(catalog.failures.count) prints") {
                        ForEach(owners.filter { catalog.failures[$0.id] != nil }.prefix(48)) { owner in
                            Text("\(owner.hostName) · \(owner.print.displayName): \(catalog.failures[owner.id] ?? "")")
                                .font(.caption).foregroundStyle(.secondaryText)
                        }
                        Button("Retry sources") { retry += 1 }
                    }
                }
                if visible.isEmpty && !catalog.loading {
                    ContentUnavailableView(search.isEmpty ? "No Source Media" : "No Matching Sources", systemImage: "photo.stack",
                        description: Text(search.isEmpty ? "Retained inputs appear here after a creation uses them." : "Try a different source, role or owner name."))
                }
                LazyVGrid(columns: [GridItem(.adaptive(minimum: edge, maximum: edge * 1.5), spacing: JustifiedLayout.gap)],
                          alignment: .leading, spacing: JustifiedLayout.gap) {
                    ForEach(visible) { source in
                        SourceMediaTile(source: source, openOwner: openOwner)
                    }
                }
                Text("Sources remain available while a library print uses them. Deleting the final owning print releases its sources.")
                    .font(.caption).foregroundStyle(.secondaryText)
            }
        }
        .toolbar { Button("Refresh", systemImage: "arrow.clockwise") { retry += 1 } }
        .task(id: SourceBrowserKey(owners: owners, routes: routes, instances: instances, retry: retry, revision: catalog.revision)) {
            let force = retry != refreshed
            refreshed = retry
            if force { await SourceMediaPreview.invalidate(authorities: Set(allowed.values)) }
            // The access monitor prunes by all authorized owners; filters must keep warm previews.
            await catalog.load(owners: owners, backend: { hosts.backend(for: $0) }, route: { hosts.host($0) },
                               instance: { hosts.instanceID(of: $0) }, refresh: force)
            while !Task.isCancelled {
                do { try await Task.sleep(for: .seconds(60)) } catch { return }
                await catalog.load(owners: owners, backend: { hosts.backend(for: $0) }, route: { hosts.host($0) },
                                   instance: { hosts.instanceID(of: $0) })
            }
        }
    }
}
private struct SourceBrowserKey: Hashable {
    let owners: [LibraryEntry]
    let routes: [UUID: MoldHost]
    let instances: [UUID: String?]
    let retry: Int
    let revision: Int
}
struct SourceMediaUsageCard: View {
    @Environment(HostStore.self) private var hosts
    let source: SourceMediaDisplay
    let openOwner: (PrintID) -> Void
    var gallery = true
    @State private var showingUses = false
    private var current: SourceMediaCatalog.Row {
        // Cards are created only for a non-empty filtered group.
        source.preferredUsage { hosts.host($0).map(hosts.isUp) ?? false }!
    }
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            SourceMediaCard(owner: current.owner, member: current.member, gallery: gallery, openSource: { showingUses = true })
                .id(SourcePreviewKey(authority: current.authority, member: current.member))
            Button { showingUses = true } label: {
                Label("\(source.usages.count) \(source.usages.count == 1 ? "use" : "uses") · Details and roles", systemImage: "link")
                    .font(.caption)
            }
            .buttonStyle(.plain).padding(.horizontal, 8)
        }
        .padding(gallery ? 12 : 0)
        .background(.quaternary.opacity(gallery ? 0.25 : 0), in: RoundedRectangle(cornerRadius: 12))
        .sheet(isPresented: $showingUses) {
            SourceMediaDetails(source: source, initial: current.id, gallery: gallery, openOwner: openOwner)
        }
    }
}
