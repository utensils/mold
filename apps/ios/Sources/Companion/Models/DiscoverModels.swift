import MoldClient
import SwiftUI

/// Search curated runnable checkpoints alongside the machine's live provider
/// catalog. Manifest rows remain available on hosts without catalog support.
struct DiscoverModels: View {
    @Environment(HostStore.self) private var hosts
    @Environment(CatalogStore.self) private var catalog
    let host: MoldHost
    @State private var text = ""

    private var state: CatalogStore.State { catalog.state(on: host.id) }
    private var capabilities: Capabilities? { hosts.capabilities[host.id] }

    var body: some View {
        List {
            FailureBanner().listRowInsets(EdgeInsets()).listRowBackground(Color.clear)
            curated
            community
        }
        .refreshable {
            await hosts.refresh(host)
            if capabilities?.canBrowseCatalog != false { catalog.search(on: host.id, debounce: false) }
        }
        .searchable(text: $text, prompt: Text("Search models"))
        .onChange(of: text) { catalog.setText(text, on: host.id) }
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) { filters }
        }
        .task(id: host.id) {
            await hosts.refresh(host)
            if capabilities?.canBrowseCatalog != false { catalog.start(on: host.id) }
            text = state.query.text ?? ""
        }
    }

    private var curated: some View {
        Section("Mold Models") {
            let matches = (hosts.models[host.id] ?? []).filter { $0.matchesDiscovery(state.query) }
            ForEach(matches) { model in CuratedModelRow(model: model, host: host) }
            if hosts.models[host.id] == nil {
                Text("The machine’s model list is unavailable. Pull to refresh.").foregroundStyle(.secondaryText)
            } else if matches.isEmpty {
                Text("No Mold models match these filters.").foregroundStyle(.secondaryText)
            }
        }
    }

    private var community: some View {
        Section("Community Models") {
            if capabilities?.canBrowseCatalog == false {
                Text("This machine does not offer community model discovery.").foregroundStyle(.secondaryText)
            } else {
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
        }
    }

    private var filters: some View {
        Menu {
            Picker("Source", selection: Binding(get: { state.query.source }, set: { catalog.setSource($0, on: host.id) })) {
                Text("All Sources").tag(String?.none)
                Text("Hugging Face").tag(String?.some("hf"))
                Text("Civitai").tag(String?.some("civitai"))
            }
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
