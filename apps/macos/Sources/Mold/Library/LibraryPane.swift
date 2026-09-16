import MoldClient
import SwiftUI

/// The merged library.
struct LibraryPane: View {
    @Environment(HostStore.self) private var hosts
    @Environment(LibraryStore.self) private var library

    @State private var selection: PrintID?
    @State private var edge: CGFloat = 132
    @State private var query = ""
    /// nil means every machine.
    @State private var sourceHost: MoldHost.ID?

    var body: some View {
        Group {
            if library.items.isEmpty {
                empty
            } else {
                LibraryGrid(
                    sections: sections,
                    hosts: hosts.hosts,
                    edge: edge,
                    showsHostBadges: sourceHost == nil && hosts.hosts.count > 1,
                    selection: $selection
                )
            }
        }
        .navigationTitle("Library")
        .navigationSubtitle(subtitle)
        .searchable(text: $query, prompt: "Search prompts and models")
        .toolbar { toolbar }
        .inspector(isPresented: .constant(selection != nil)) {
            LibraryInspector(item: selected, host: selectedHost)
                .inspectorColumnWidth(min: 260, ideal: 320, max: 420)
        }
        .task { await reload() }
        .focusedSceneValue(\.refreshAction) { Task { await reload() } }
    }

    // MARK: - Content

    private var visible: [LibraryEntry] {
        library.items.filter { item in
            guard sourceHost == nil || item.hostID == sourceHost else { return false }
            guard !query.isEmpty else { return true }
            return item.matches(query)
        }
    }

    private var sections: [LibrarySection] { LibraryGrouping.byDay(visible) }

    private var selected: LibraryEntry? {
        guard let selection else { return nil }
        return library.items.first { $0.id == selection }
    }

    private var selectedHost: MoldHost? {
        guard let selected else { return nil }
        return hosts.hosts.first { $0.id == selected.hostID }
    }

    private var subtitle: String {
        let shown = visible.count
        let total = library.items.count
        return shown == total
            ? "\(total.formatted()) prints"
            : "\(shown.formatted()) of \(total.formatted()) prints"
    }

    @ViewBuilder private var empty: some View {
        if library.isLoading {
            ProgressView("Loading prints…")
        } else if let failure = library.failures.values.first {
            ContentUnavailableView("Can't load the library", systemImage: "exclamationmark.triangle",
                                   description: Text(failure))
        } else {
            ContentUnavailableView("No prints yet", systemImage: "photo.on.rectangle.angled",
                                   description: Text("Prints from every machine appear here."))
        }
    }

    // MARK: - Toolbar

    @ToolbarContentBuilder private var toolbar: some ToolbarContent {
        ToolbarItem {
            Picker("Source", selection: $sourceHost) {
                Text("All machines").tag(MoldHost.ID?.none)
                ForEach(hosts.hosts) { host in
                    Text("\(host.name) (\(library.count(for: host.id)))")
                        .tag(MoldHost.ID?.some(host.id))
                }
            }
        }
        ToolbarItem {
            Slider(value: $edge, in: 88...260) { Text("Thumbnail size") }
                .frame(width: 110)
                .help("Thumbnail size")
        }
        ToolbarItem {
            Button { Task { await reload() } } label: {
                Label("Refresh", systemImage: "arrow.clockwise")
            }
            .disabled(library.isLoading)
        }
    }

    private func reload() async {
        await library.refresh(hosts: hosts.hosts) { hosts.backend(for: $0) }
    }
}
