import MoldClient
import SwiftUI

struct SourceMediaBrowser: View {
    @Environment(HostStore.self) private var hosts
    @Environment(LibraryStore.self) private var library
    @Environment(LibraryNavigation.self) private var navigation
    @Environment(\.sourceMediaOwner) private var openOwner
    @State private var catalog = SourceMediaCatalog()
    @State private var search = ""
    @State private var retry = 0

    private var owners: [LibraryEntry] {
        var query = LibraryQuery()
        query.tokens = navigation.query.machineIDs.map { .machine(id: $0, name: hosts.host($0)?.name ?? "") }
        let projected = LibraryScope.all.resolve(query, shelves: library.shelves, hiddenCollectionIDs: library.hiddenCollectionIDs).apply(to: library.items)
        var seen = Set<PrintID>()
        return projected.flatMap(\.everyCopy).filter { seen.insert($0.id).inserted }
    }
    private var visible: [SourceMediaCatalog.Row] {
        let allowed = Set(owners)
        return catalog.rows.filter { row in
            allowed.contains(row.owner) && row.route == hosts.host(row.owner.hostID) && row.instance == hosts.instanceIDs[row.owner.hostID] && (search.isEmpty ||
                "\(row.member.sourceLabel) \(row.member.displayName) \(row.owner.print.displayName) \(row.owner.hostName)".localizedCaseInsensitiveContains(search))
        }
    }
    var body: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 20) {
                Text("Original inputs retained by your library. Deleting a source’s final owning print permanently releases it.")
                    .font(.callout).foregroundStyle(.secondary)
                if catalog.loading {
                    ProgressView("Reading sources: \(catalog.completed) of \(catalog.total) prints")
                }
                if !catalog.failures.isEmpty {
                    Text("Sources unavailable for \(catalog.failures.count) prints. Connect their machines or retry.")
                        .foregroundStyle(.secondary)
                    DisclosureGroup("Show unavailable prints") {
                        ForEach(owners.filter { catalog.failures[$0.id] != nil }) { owner in
                            Text("\(owner.hostName) · \(owner.print.displayName): \(catalog.failures[owner.id] ?? "")")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                    }
                    Button("Retry sources") { retry += 1 }
                }
                if visible.isEmpty && !catalog.loading {
                    ContentUnavailableView(search.isEmpty ? "No Source Media" : "No Matching Sources", systemImage: "photo.stack",
                        description: Text(search.isEmpty ? "Retained inputs appear here after a creation uses them." : "Try a different source, role or owner name."))
                }
                let groups = Dictionary(grouping: visible, by: { $0.owner.id })
                ForEach(owners.filter { groups[$0.id] != nil }) { owner in
                    VStack(alignment: .leading, spacing: 12) {
                        Button { openOwner(owner.id) } label: {
                            VStack(alignment: .leading, spacing: 3) {
                                Label(owner.print.displayName, systemImage: "arrow.up.right.square")
                                    .font(.headline)
                                Text(owner.hostName).font(.caption).foregroundStyle(.secondary)
                            }
                        }
                        .buttonStyle(.plain)
                        .accessibilityLabel("Open owning print \(owner.print.displayName)")
                        LazyVGrid(columns: [GridItem(.adaptive(minimum: 230, maximum: 340), alignment: .top)], alignment: .leading, spacing: 14) {
                            ForEach(groups[owner.id] ?? []) { row in
                                SourceMediaCard(owner: row.owner, member: row.member, gallery: true)
                            }
                        }
                    }
                }
            }
            .padding()
        }
        .navigationTitle("Source Media")
        .searchable(text: $search, prompt: "Search sources and owners")
        .toolbar { Button("Refresh", systemImage: "arrow.clockwise") { retry += 1 } }
        .task(id: SourceBrowserKey(owners: owners, hosts: hosts.hosts, instances: hosts.hosts.map { hosts.instanceIDs[$0.id] }, retry: retry)) {
            await catalog.load(owners: owners, backend: { hosts.backend(for: $0) },
                               route: { hosts.host($0) }, instance: { hosts.instanceIDs[$0] })
        }
    }
}
private struct SourceBrowserKey: Hashable {
    let owners: [LibraryEntry]
    let hosts: [MoldHost]
    let instances: [String?]
    let retry: Int
}

extension EnvironmentValues {
    @Entry var sourceMediaOwner: (PrintID) -> Void = { _ in }
}
