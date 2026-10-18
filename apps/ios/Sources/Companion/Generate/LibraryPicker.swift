import MoldClient
import SwiftUI

/// "Choose from Library…": the Library's stills, and the chosen print's own
/// bytes fetched from the machine that holds it.
struct LibraryPicker: View {
    @Environment(LibraryStore.self) private var library
    @Environment(HostStore.self) private var hosts
    @Environment(\.dismiss) private var dismiss
    @ScaledMetric(relativeTo: .body) private var tile: CGFloat = 96
    let picked: (Data, String) -> Void
    @State private var search = ""
    @State private var machine: MoldHost.ID?
    @State private var choosing = false
    @State private var failure: String?
    @State private var selectionTask: Task<Void, Never>?

    private var stills: [LibraryEntry] {
        var query = LibraryQuery()
        query.text = search
        query.hiddenCollectionIDs = library.hiddenCollectionIDs
        query.privacyEvidence = library.trashPool
        if let machine, let host = hosts.host(machine) { query.tokens = [.machine(id: machine, name: host.name)] }
        return query.apply(to: library.pool).filter { $0.print.kind == .picture }
    }

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                filters
                Divider()
                pictures
            }
            .searchable(text: $search, prompt: "Search pictures")
            .refreshable { await library.reload() }
            .task { await library.reload() }
            .navigationTitle("Choose from Library")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar { ToolbarItem(placement: .cancellationAction) { Button("Cancel") { dismiss() } } }
        }
        .onDisappear { selectionTask?.cancel(); selectionTask = nil }
    }

    /// Ordinary content owns the filter's height. An opaque top safe-area
    /// inset painted an intrinsic-width rectangle through the navigation bar.
    private var filters: some View {
        VStack(alignment: .leading, spacing: 8) {
            Picker("Machine", selection: $machine) {
                Text("All Machines").tag(MoldHost.ID?.none)
                ForEach(hosts.hosts) { host in Text(host.name).tag(MoldHost.ID?.some(host.id)) }
            }
            if let failure { Text(failure).foregroundStyle(.secondaryText) }
            if choosing { ProgressView("Fetching picture…") }
        }
        .padding(.horizontal, 16).padding(.vertical, 8)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(uiColor: .systemBackground), ignoresSafeAreaEdges: [])
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("library-picker-filters")
    }

    @ViewBuilder private var pictures: some View {
        if stills.isEmpty {
            EmptyState(title: library.isLoading ? String(localized: "Loading pictures…") : String(localized: "No matching pictures"), symbol: "photo",
                       message: String(localized: "Pictures you generate appear here to start from."))
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        } else {
            ScrollView {
                LazyVGrid(columns: [GridItem(.adaptive(minimum: tile), spacing: 3)], spacing: 3) {
                    ForEach(stills) { entry in
                        VStack {
                        Button { choose(entry) } label: {
                            Color.clear.aspectRatio(1, contentMode: .fit)
                                .overlay { PrintThumbnail(entry: entry, points: tile * 1.5) }
                                .clipShape(.rect(cornerRadius: 5))
                        }
                        .buttonStyle(.plain)
                        .disabled(choosing)
                        .accessibilityLabel(entry.spokenDescription(showsHost: true))
                        let source = entry.presented(onAnyOf: Set(hosts.upHosts.map(\.id))) ?? entry
                        let authority = hosts.host(source.hostID).map { InputMediaPreviewAuthority(host: $0, instance: hosts.instanceID(of: source.hostID)) }
                        InputMediaViewButton(title: source.print.displayName, identity: "\(source.id)|\(String(describing: authority))", load: {
                            guard let authority, authority.isCurrent(host: hosts.host(source.hostID), instance: hosts.instanceID(of: source.hostID)) else { throw CancellationError() }
                            let route = authority.host
                            let file = try await hosts.backend(for: route).mediaPreviewFile(source.print.filename, trashed: false)
                            guard authority.isCurrent(host: hosts.host(source.hostID), instance: hosts.instanceID(of: source.hostID)),
                                  stills.contains(where: { $0.id == entry.id }) else { throw CancellationError() }
                            return file
                        }, isCurrent: { stills.contains(where: { $0.id == entry.id }) && authority?.isCurrent(host: hosts.host(source.hostID), instance: hosts.instanceID(of: source.hostID)) == true })
                        }
                    }
                }
            }
        }
    }

    private func choose(_ entry: LibraryEntry) {
        // A merged tile can lead with a saved copy on an offline machine.
        // Prefer the same print on a machine that is currently answering.
        let source = entry.presented(onAnyOf: Set(hosts.upHosts.map(\.id))) ?? entry
        guard let host = hosts.host(source.hostID) else { return }
        choosing = true
        failure = nil
        selectionTask?.cancel()
        selectionTask = Task {
            defer { choosing = false }
            do {
                let data = try await hosts.backend(for: host).media(source.print.filename, trashed: false)
                guard !Task.isCancelled else { return }
                picked(data, source.print.filename)
                dismiss()
            } catch {
                guard !Task.isCancelled else { return }
                failure = "Couldn’t fetch that picture. Try again or choose another machine."
                hosts.report(host, doing: String(localized: "fetch that picture"), error)
            }
        }
    }
}
