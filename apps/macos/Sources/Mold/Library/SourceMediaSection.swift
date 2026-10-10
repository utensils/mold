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
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Source Media").font(.headline)
            if let inventory, loadedAuthority == authority, loadedRevision == library.sourceMediaCatalog.revision, library.allowsSource(authority, detail: true) {
                if inventory.availability == .available {
                    let groups = SourceMediaCatalog.group(inventory.members.map {
                        SourceMediaCatalog.Row(owner: entry, member: $0, route: hosts.host(entry.hostID), instance: hosts.instanceIDs[entry.hostID])
                    })
                    ForEach(groups.prefix(limit)) { group in
                        SourceMediaUsageCard(source: .init(id: group.id, usages: group.usages), openOwner: { openOwner($0) }, gallery: false)
                        Divider()
                    }
                    if groups.count > limit { Button("Show more sources") { limit += 48 } }
                    if groups.isEmpty { Text("No retained source media.").foregroundStyle(.secondary) }
                } else if inventory.availability != .unavailableLegacy || RetainedSourceMedia.disclosable(entry.print.metadata) {
                    Text(RetainedSourceMedia.disclosure(inventory.availability) ?? "Source media unavailable.").foregroundStyle(.secondary)
                } else { Text("No retained source media.").foregroundStyle(.secondary) }
            } else if failure == nil { ProgressView("Loading sources…") }
            if let failure { Text(failure).foregroundStyle(.secondary) }
            if failure != nil || inventory?.availability != .available {
                Button("Retry sources") { retry += 1 }
            }
        }
        .onDisappear { library.openSourceDetails[detailID] = nil }
        .task(id: SourceLoadKey(authority: authority, retry: retry, revision: library.sourceMediaCatalog.revision)) {
            library.openSourceDetails[detailID] = authority
            await load(force: retry > 0)
            while !Task.isCancelled {
                do { try await Task.sleep(for: .seconds(60)) } catch { return }
                await load(force: true)
            }
        }
    }
    private func load(force: Bool) async {
        inventory = nil; loadedAuthority = nil; failure = nil
        guard library.allowsSource(authority, detail: true) else { failure = "This print is no longer available in the library."; return }
        guard let host = hosts.host(entry.hostID) else { failure = "Connect this machine to read its sources."; return }
        let captured = authority
        let revision = library.sourceMediaCatalog.revision
        do {
            if force { await SourceMediaPreview.invalidate(host: entry.hostID, filename: entry.print.filename) }
            let result = try await library.sourceMediaCatalog.inventory(for: entry, backend: hosts.backend(for: host), route: host,
                                                                         instance: hosts.instanceIDs[entry.hostID], refresh: force)
            guard !Task.isCancelled, captured == authority, library.allowsSource(captured, detail: true), revision == library.sourceMediaCatalog.revision else { return }
            inventory = result; loadedAuthority = captured; loadedRevision = revision
        } catch { if !Task.isCancelled { failure = error.localizedDescription } }
    }
}
private struct SourceLoadKey: Hashable {
    let authority: SourceMediaAuthority
    let retry: Int
    let revision: Int
}
