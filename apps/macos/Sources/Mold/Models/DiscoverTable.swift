import MoldClient
import SwiftUI

/// The catalog itself: a family filter and a sort menu built from this
/// MACHINE's own advertised lists, never a client guess
/// (`capabilities.catalogFamilies`/`catalogSortOptions`), a table of results,
/// and a Load more row while there is more than this page holds.
struct DiscoverTable: View {
    let host: MoldHost
    /// The pane's own search field -- Discover reads the SAME text Installed
    /// filters by rather than a second search box (design S6).
    @Binding var searchText: String
    @Environment(HostStore.self) private var hosts
    @Environment(CatalogStore.self) private var catalog
    // Not `private`: `DiscoverTable+Cells.swift` reads both, and `private`
    // does not cross a file boundary even within one type.
    @Environment(DownloadStore.self) var downloads
    @State private var selection: CatalogEntry.ID?
    @State var detailEntry: CatalogEntry?

    private var capabilities: Capabilities? { hosts.capabilities[host.id] }
    private var entries: [CatalogEntry] { catalog.entries(on: host.id) }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            filters
            if let note = Self.providerNote(catalog.providerErrors(on: host.id)) {
                Text(note).font(.caption).foregroundStyle(.secondary)
                    .padding(.horizontal, 12).padding(.top, 4)
            }
            if entries.isEmpty {
                empty
            } else {
                table
                if catalog.hasMore(on: host.id) { loadMore }
            }
        }
        .task(id: host.id) {
            catalog.adopt(host.id, sortOptions: capabilities?.catalogSortOptions ?? [])
            if catalog.entries(on: host.id).isEmpty { catalog.search(on: host.id) }
            await catalog.loadCredentials(on: host.id)
        }
        .onChange(of: searchText) { _, text in catalog.setText(text, on: host.id) }
        .onChange(of: selection) { _, id in detailEntry = entries.first { $0.id == id } }
        .sheet(item: $detailEntry) { entry in CatalogDetailSheet(entry: entry, host: host) }
    }

    /// Explicit `rows:`, never the plain data-array initializer: a "Load
    /// more" affordance and a provider note are views AROUND this `Table`,
    /// never rows inside it, and `.alternatingRowBackgrounds(.disabled)` stops
    /// AppKit painting striped filler past the last real row as what a live
    /// capture read as two blank rows (design S6b).
    private var table: some View {
        Table(of: CatalogEntry.self, selection: $selection) {
            TableColumn("Name") { entry in nameCell(entry) }
            TableColumn("Family") { entry in Text(entry.family) }
            TableColumn("Kind") { entry in Text(entry.kind) }
            TableColumn("Size") { entry in Text(Self.sizeText(entry)).foregroundStyle(.secondary) }
            TableColumn("Downloads") { entry in Text(Self.downloadsText(entry)).foregroundStyle(.secondary) }
            TableColumn("State") { entry in stateCell(entry) }
        } rows: {
            ForEach(Self.rows(for: entries)) { entry in
                TableRow(entry).rowActionMenu(DiscoverRow.menuItems(for: entry)) {
                    perform($0, on: entry)
                }
            }
        }
        .alternatingRowBackgrounds(.disabled)
    }

    private var filters: some View {
        let query = catalog.query(on: host.id)
        return HStack(spacing: 8) {
            Picker("Family", selection: Binding(get: { query.family }, set: { catalog.setFamily($0, on: host.id) })) {
                Text("Any").tag(String?.none)
                ForEach(capabilities?.catalogFamilies ?? [], id: \.self) { Text($0).tag(String?.some($0)) }
            }
            .frame(width: 160)
            // Hidden, not drawn empty, on a host that advertises no sorts.
            if let sortOptions = capabilities?.catalogSortOptions, !sortOptions.isEmpty {
                Picker("Sort", selection: Binding(get: { query.sort }, set: { catalog.setSort($0, on: host.id) })) {
                    ForEach(sortOptions, id: \.self) { Text($0.capitalized).tag(String?.some($0)) }
                }
                .frame(width: 140)
            }
            if catalog.isSearching(on: host.id) { ProgressView().controlSize(.small) }
            Spacer()
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
    }

    private var loadMore: some View {
        HStack {
            Spacer()
            Button("Load more") { Task { await catalog.more(on: host.id) } }
                .buttonStyle(.bordered).controlSize(.small)
            Spacer()
        }
        .padding(.vertical, 8)
    }

    @ViewBuilder private var empty: some View {
        if catalog.isSearching(on: host.id) {
            ProgressView("Searching the catalog…").frame(maxWidth: .infinity, maxHeight: .infinity)
        } else {
            ContentUnavailableView("No results", systemImage: "magnifyingglass",
                                   description: Text("Nothing matches on this machine's catalog."))
        }
    }
}
