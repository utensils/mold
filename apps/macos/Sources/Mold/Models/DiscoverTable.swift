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
    @Environment(DownloadStore.self) private var downloads
    @State private var selection: CatalogEntry.ID?
    @State private var detailEntry: CatalogEntry?

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
            if catalog.entries(on: host.id).isEmpty { catalog.search(on: host.id) }
            await catalog.loadCredentials(on: host.id)
        }
        .onChange(of: searchText) { _, text in catalog.setText(text, on: host.id) }
        .onChange(of: selection) { _, id in detailEntry = entries.first { $0.id == id } }
        .sheet(item: $detailEntry) { entry in CatalogDetailSheet(entry: entry, host: host) }
    }

    private var table: some View {
        Table(entries, selection: $selection) {
            TableColumn("Name") { entry in nameCell(entry) }
            TableColumn("Family") { entry in Text(entry.family) }
            TableColumn("Kind") { entry in Text(entry.kind) }
            TableColumn("Size") { entry in Text(Self.sizeText(entry)).foregroundStyle(.secondary) }
            TableColumn("Downloads") { entry in Text(Self.downloadsText(entry)).foregroundStyle(.secondary) }
            TableColumn("State") { entry in stateCell(entry) }
        }
    }

    private var filters: some View {
        let query = catalog.query(on: host.id)
        return HStack(spacing: 8) {
            Picker("Family", selection: Binding(get: { query.family }, set: { catalog.setFamily($0, on: host.id) })) {
                Text("Any").tag(String?.none)
                ForEach(capabilities?.catalogFamilies ?? [], id: \.self) { Text($0).tag(String?.some($0)) }
            }
            .frame(width: 160)
            Picker("Sort", selection: Binding(get: { query.sort }, set: { catalog.setSort($0, on: host.id) })) {
                ForEach(capabilities?.catalogSortOptions ?? [], id: \.self) { Text($0.capitalized).tag(String?.some($0)) }
            }
            .frame(width: 140)
            if catalog.isSearching(on: host.id) { ProgressView().controlSize(.small) }
            Spacer()
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
    }

    private func nameCell(_ entry: CatalogEntry) -> some View {
        VStack(alignment: .leading, spacing: 1) {
            HStack(spacing: 4) {
                Text(entry.name).lineLimit(1)
                if let badge = Self.nsfwBadge(entry) {
                    Text(badge).font(.caption2.weight(.semibold)).foregroundStyle(.secondary)
                }
            }
            if let author = entry.author {
                Text(author).font(.caption).foregroundStyle(.secondary).lineLimit(1)
            }
        }
        .contentShape(Rectangle())
        .onTapGesture(count: 2) { detailEntry = entry }
    }

    @ViewBuilder private func stateCell(_ entry: CatalogEntry) -> some View {
        switch DiscoverRow.resolve(entry) {
        case .installed:
            Label("Installed", systemImage: "checkmark.circle.fill")
                .font(.caption).foregroundStyle(.secondary)
        case .install:
            Button("Install") { Task { await downloads.install(entry.id, on: host) } }
                .buttonStyle(.bordered).controlSize(.small)
        case let .unsupported(pageURL):
            HStack(spacing: 6) {
                Text("Not supported").font(.caption).foregroundStyle(.secondary)
                if let pageURL { Link("Open Page", destination: pageURL).font(.caption) }
            }
        }
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

    /// "Civitai didn't answer." -- one provider failing beside rows the
    /// other did return, never a banner (design S6 test 2).
    static func providerNote(_ errors: [CatalogProviderError]) -> String? {
        guard !errors.isEmpty else { return nil }
        let names = errors.map { $0.source.capitalized }.joined(separator: ", ")
        return "\(names) didn't answer."
    }

    static func sizeText(_ entry: CatalogEntry) -> String {
        guard let bytes = entry.sizeBytes else { return "—" }
        return Int64(bytes).formatted(.byteCount(style: .file))
    }

    static func downloadsText(_ entry: CatalogEntry) -> String {
        let downloads = entry.downloadCount.formatted(.number.notation(.compactName))
        guard let rating = entry.rating else { return downloads }
        return "\(downloads) · \(rating.formatted(.number.precision(.fractionLength(1))))★"
    }

    /// `false` draws nothing -- never an affirmative "Safe" claim for the
    /// ordinary case (design S6 test 7).
    static func nsfwBadge(_ entry: CatalogEntry) -> String? { entry.nsfw ? "NSFW" : nil }
}
