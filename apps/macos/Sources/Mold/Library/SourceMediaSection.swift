import MoldClient
import SwiftUI

/// Details shares fresh catalog answers and groups bytes without dropping roles.
struct SourceMediaSection: View {
    @Environment(HostStore.self) private var hosts
    @Environment(LibraryStore.self) private var library
    @Environment(\.sourceMediaOwner) private var openOwner
    let entry: LibraryEntry
    @State private var inventory: RetainedSourceMedia.Inventory?
    @State private var detailID = UUID()
    @State private var loadedRevision = -1
    @State private var loadedAuthority: SourceMediaAuthority?
    @State private var failure: String?
    @State private var retry = 0
    @State private var limit = 48
    private var authority: SourceMediaAuthority {
        .init(owner: entry, route: hosts.host(entry.hostID), instance: hosts.instanceIDs[entry.hostID])
    }
    private var state: SourceMediaState {
        SourceMediaState(inventory: loadedAuthority == authority && loadedRevision == library.sourceMediaCatalog.contentRevision(for: entry.id) ? inventory : nil,
                         failed: failure != nil, metadata: entry.print.metadata)
    }
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            if state.showsSection(hasConditioning: RetainedSourceMedia.disclosable(entry.print.metadata)) {
            Text("Source Media").font(.headline)
            if let inventory, loadedAuthority == authority, loadedRevision == library.sourceMediaCatalog.contentRevision(for: entry.id), library.allowsSource(authority, detail: true) {
                if inventory.availability == .available {
                    let groups = SourceMediaCatalog.group(inventory.members.map {
                        SourceMediaCatalog.Row(owner: entry, member: $0, route: hosts.host(entry.hostID), instance: hosts.instanceIDs[entry.hostID])
                    })
                    ForEach(groups.prefix(limit)) { group in
                        SourceMediaUsageCard(source: .init(id: group.id, usages: group.usages), openOwner: { openOwner($0) }, gallery: false)
                        Divider()
                    }
                    if groups.count > limit { Button("Show more sources") { limit += 48 } }
                    if groups.isEmpty { Text("No source media.").foregroundStyle(.secondary) }
                } else if SourceMediaState(inventory: inventory, failed: false, metadata: entry.print.metadata) == .none {
                    Text("No source media.").foregroundStyle(.secondary)
                } else {
                    Text(RetainedSourceMedia.disclosure(inventory.availability) ?? "Source media unavailable.").foregroundStyle(.secondary)
                }
            } else if failure == nil { ProgressView("Loading sources…") }
            if let failure { Text(failure).foregroundStyle(.secondary) }
            if SourceMediaState(inventory: inventory, failed: failure != nil, metadata: entry.print.metadata).canRetry {
                Button("Retry sources") { retry += 1 }
            }
            }
        }
        .onDisappear { library.openSourceDetails[detailID] = nil }
        .task(id: SourceMediaInventoryKey(authority: authority, retry: retry, revision: library.sourceMediaCatalog.contentRevision(for: entry.id))) {
            library.openSourceDetails[detailID] = authority
            await load(force: retry > 0)
            if state == .none { library.openSourceDetails[detailID] = nil }
        }
    }
    private func load(force: Bool) async {
        if loadedAuthority != authority || loadedRevision != library.sourceMediaCatalog.contentRevision(for: entry.id) {
            inventory = nil; loadedAuthority = nil
        }
        failure = nil
        guard library.allowsSource(authority, detail: true) else { failure = "This print is no longer available in the library."; return }
        guard let host = hosts.host(entry.hostID) else { failure = "Connect this machine to read its sources."; return }
        let captured = authority
        let revision = library.sourceMediaCatalog.contentRevision(for: entry.id)
        do {
            guard let result = try await SourceMediaLoading.inventory(owner: entry, authority: captured,
                catalog: library.sourceMediaCatalog, backend: hosts.backend(for: host), refresh: force, isCurrent: {
                    captured == authority && library.allowsSource(captured, detail: true)
                        && revision == library.sourceMediaCatalog.contentRevision(for: entry.id)
                }) else { return }
            inventory = result; loadedAuthority = captured; loadedRevision = revision
        } catch { if !Task.isCancelled { failure = error.localizedDescription } }
    }
}
