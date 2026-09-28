import MoldClient
import SwiftUI

/// Discover: the machine's catalog, searchable, with Family and Sort
/// filters it advertises. Each row offers Get, says Installed, or opens the
/// source page for one this machine cannot run.
struct DiscoverModels: View {
    @Environment(HostStore.self) private var hosts
    @Environment(CatalogStore.self) private var catalog
    @Environment(ModelStore.self) private var models
    let host: MoldHost
    @State private var text = ""

    var body: some View {
        let state = catalog.state(on: host.id)
        let capabilities = hosts.capabilities[host.id]
        Group {
            if capabilities?.canBrowseCatalog == false {
                EmptyState(title: String(localized: "No catalog here"), symbol: "magnifyingglass",
                           message: String(localized: "\(host.name) doesn't offer model discovery. Update it to browse and fetch models from here."))
            } else {
                List {
                    FailureBanner().listRowInsets(EdgeInsets()).listRowBackground(Color.clear)
                    ForEach(state.listing?.providerErrors ?? [], id: \.source) { problem in
                        Text("\(problem.source): \(problem.message)").foregroundStyle(.secondaryText)
                    }
                    ForEach(state.entries) { entry in
                        DiscoverRow(entry: entry, host: host)
                            .task { if entry.id == state.entries.last?.id { await catalog.more(on: host.id) } }
                    }
                    if state.isSearching {
                        ProgressView().frame(maxWidth: .infinity)
                    } else if state.listing != nil, state.entries.isEmpty {
                        Text("Nothing matches. Try other words, or another family.").foregroundStyle(.secondaryText)
                    }
                }
                .searchable(text: $text, prompt: Text("Search models"))
                .onChange(of: text) { catalog.setText(text, on: host.id) }
                .toolbar {
                    ToolbarItem(placement: .topBarTrailing) {
                        Menu {
                            Picker("Family", selection: Binding(get: { state.query.family }, set: { catalog.setFamily($0, on: host.id) })) {
                                Text("All Families").tag(String?.none)
                                ForEach(capabilities?.catalogFamilies ?? [], id: \.self) { Text($0).tag(String?.some($0)) }
                            }
                            if let sorts = capabilities?.catalogSortOptions, !sorts.isEmpty {
                                Picker("Sort By", selection: Binding(get: { state.query.sort }, set: { catalog.setSort($0, on: host.id) })) {
                                    ForEach(sorts, id: \.self) { Text($0.capitalized).tag(String?.some($0)) }
                                }
                            }
                        } label: {
                            Label("Filter", systemImage: "line.3.horizontal.decrease")
                        }
                    }
                }
            }
        }
        .task(id: host.id) { catalog.start(on: host.id); text = catalog.state(on: host.id).query.text ?? "" }
    }
}

private struct DiscoverRow: View {
    @Environment(ModelStore.self) private var models
    @Environment(HostStore.self) private var hosts
    @Environment(\.dynamicTypeSize) private var size
    let entry: CatalogEntry
    let host: MoldHost

    var body: some View {
        let stacked = RowAxis.for(size) == .vertical
        let layout = stacked ? AnyLayout(VStackLayout(alignment: .leading, spacing: 8))
                             : AnyLayout(HStackLayout(alignment: .center, spacing: 12))
        layout {
            VStack(alignment: .leading, spacing: 2) {
                Text(entry.name)
                if let author = entry.author { Text("by \(author)").font(.callout).foregroundStyle(.secondaryText) }
                Text(verbatim: detail).font(.caption.monospaced()).foregroundStyle(.secondaryText)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            action.frame(maxWidth: stacked ? .infinity : nil)
        }
        .padding(.vertical, 2)
    }

    private var detail: String {
        var parts = [entry.family]
        if let bytes = entry.sizeBytes { parts.append(ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)) }
        parts.append(String(localized: "\(entry.downloadCount.formatted(.number.notation(.compactName))) downloads"))
        return parts.joined(separator: " · ")
    }

    @ViewBuilder private var action: some View {
        if let (job, row) = models.progress(for: entry.id, on: host.id) {
            VStack(alignment: .trailing, spacing: 4) {
                if let fraction = row.fraction { ProgressView(value: fraction).frame(minWidth: 80) }
                Button("Cancel Download", role: .destructive) { Task { await models.cancel(job: job, on: host.id) } }
                    .buttonStyle(.bordered)
            }
        } else if entry.installed {
            Text("Installed").foregroundStyle(.secondaryText)
        } else if entry.supported {
            Button("Get") { Task { await models.install(entry.id, on: host.id) } }
                .buttonStyle(.bordered)
                .accessibilityLabel(String(localized: "Get \(entry.name)"))
        } else if let page = entry.pageUrl.flatMap(URL.init(string:)) {
            Link("Open Page", destination: page).buttonStyle(.bordered)
        }
    }
}
