import MoldClient
import SwiftUI

/// This container observes library changes; scan progress and typing only redraw the content.
struct SourceMediaBrowser: View {
    @Environment(HostStore.self) private var hosts
    @Environment(LibraryStore.self) private var library
    @Environment(AppRouter.self) private var router
    var body: some View {
        let query: LibraryQuery = {
        var query = LibraryQuery()
        query.tokens = library.machineIDs.map { .machine(id: $0, name: hosts.host($0)?.name ?? "") }
            return query
        }()
        let projected = LibraryScope.all.resolve(query, shelves: library.shelves, hiddenCollectionIDs: library.hiddenCollectionIDs).apply(to: library.pool)
        var seen = Set<PrintID>()
        let owners = projected.flatMap(\.everyCopy).filter { seen.insert($0.id).inserted }
        let routes = Dictionary(uniqueKeysWithValues: hosts.hosts.map { ($0.id, $0) })
        let instances = Dictionary(uniqueKeysWithValues: hosts.hosts.map { ($0.id, hosts.instanceID(of: $0.id)) })
        let allowed = Dictionary(owners.map { ($0.id, SourceMediaAuthority(owner: $0, route: routes[$0.hostID], instance: instances[$0.hostID] ?? nil)) }, uniquingKeysWith: { first, _ in first })
        SourceMediaContent(owners: owners, routes: routes, instances: instances, allowed: allowed, catalog: library.sourceMediaCatalog,
                           openOwner: { router.openedPrint = .init(id: $0) })
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
    @State private var search = ""
    @State private var retry = 0
    @State private var limit = 48
    @State private var refreshed = 0
    var body: some View {
        let term = search.folding(options: [.caseInsensitive, .diacriticInsensitive], locale: .current)
        let visible = catalog.sources.compactMap { source -> SourceMediaDisplay? in
            let usages = source.usages.filter { allowed[$0.owner.id] == $0.authority }
            guard !usages.isEmpty else { return nil }
            let key = usages.count == source.usages.count ? source.searchKey : usages.map { "\($0.member.sourceLabel) \($0.member.displayName) \($0.owner.print.displayName) \($0.owner.hostName)" }.joined(separator: " ").folding(options: [.caseInsensitive, .diacriticInsensitive], locale: .current)
            guard term.isEmpty || key.contains(term) else { return nil }
            return SourceMediaDisplay(id: source.id, usages: usages)
        }
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 20) {
                Text("Your original inputs, ready to use again. Identical sources appear once; every use stays linked to its print.")
                    .font(.callout).foregroundStyle(.secondaryText)
                HStack {
                    Text("\(visible.count) sources · \(catalog.completed) of \(owners.count) prints checked")
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
                LazyVGrid(columns: [GridItem(.adaptive(minimum: 230, maximum: 340), alignment: .top)], alignment: .leading, spacing: 18) {
                    ForEach(visible.prefix(limit)) { source in
                        SourceMediaUsageCard(source: source, openOwner: openOwner)
                    }
                }
                if visible.count > limit { Button("Show more sources") { limit += 48 }.buttonStyle(.bordered) }
                Text("Sources remain available while a library print uses them. Deleting the final owning print releases its sources.")
                    .font(.caption).foregroundStyle(.secondaryText)
            }.padding()
        }
        .navigationTitle("Source Media")
        .searchable(text: $search, prompt: "Search sources and owners")
        .onChange(of: search) { limit = 48 }
        .toolbar { Button("Refresh", systemImage: "arrow.clockwise") { retry += 1 } }
        .task(id: SourceBrowserKey(owners: owners, routes: routes, instances: instances, retry: retry, revision: catalog.revision)) {
            let force = retry != refreshed
            refreshed = retry
            if force { await SourceMediaPreview.invalidate(authorities: Set(allowed.values)) }
            await SourceMediaPreview.retain(authorities: Set(allowed.values))
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
struct SourceMediaDisplay: Identifiable {
    let id: String
    let usages: [SourceMediaCatalog.Row]
}
struct SourceMediaUsageCard: View {
    @Environment(HostStore.self) private var hosts
    let source: SourceMediaDisplay
    let openOwner: (PrintID) -> Void
    var gallery = true
    @State private var selection: String?
    @State private var showingUses = false
    @State private var usageLimit = 48
    private var current: SourceMediaCatalog.Row {
        source.usages.first { $0.id == selection }
            ?? source.usages.first { $0.member.contentKind == "image" && (hosts.host($0.owner.hostID).map(hosts.isUp) ?? false) }
            ?? source.usages.first { (hosts.host($0.owner.hostID).map(hosts.isUp) ?? false) }
            ?? source.usages.first { $0.member.contentKind == "image" }
            ?? source.usages[0]
    }
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            SourceMediaCard(owner: current.owner, member: current.member, gallery: gallery)
                .id(SourcePreviewKey(authority: current.authority, member: current.member))
            Button { showingUses = true } label: {
                Label("\(source.usages.count) \(source.usages.count == 1 ? "use" : "uses") · View prints and roles", systemImage: "link")
                    .font(.caption)
            }
            .buttonStyle(.plain).padding(.horizontal, 8)
        }
        .sheet(isPresented: $showingUses) {
            NavigationStack {
                List {
                    ForEach(source.usages.prefix(usageLimit)) { row in
                        VStack(alignment: .leading, spacing: 8) {
                            Text(row.member.sourceLabel).font(.headline)
                            Text("\(row.owner.print.displayName) · \(row.owner.hostName)").font(.caption)
                            HStack {
                                Button("Use this copy") { selection = row.id; showingUses = false }
                                Spacer()
                                Button("Open print") { showingUses = false; openOwner(row.owner.id) }
                            }.buttonStyle(.borderless)
                        }.padding(.vertical, 4)
                    }
                    if source.usages.count > usageLimit { Button("Show more uses") { usageLimit += 48 } }
                }
                .navigationTitle("Used in your library")
                .toolbar { ToolbarItem(placement: .confirmationAction) { Button("Done") { showingUses = false } } }
            }
            .presentationDetents([.medium, .large])
        }
    }
}
