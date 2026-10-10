import MoldClient
import SwiftUI

/// Details always asks, including outputs whose embedded metadata omits inputs.
struct SourceMediaSection: View {
    @Environment(HostStore.self) private var hosts
    let entry: LibraryEntry
    @State private var inventory: RetainedSourceMedia.Inventory?
    @State private var failure: String?
    @State private var retry = 0
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Source Media").font(.headline)
            if let inventory {
                if inventory.availability == .available {
                    ForEach(inventory.members) { member in
                        SourceMediaCard(owner: entry, member: member)
                        Divider()
                    }
                    if inventory.members.isEmpty { Text("No retained source media.").foregroundStyle(.secondaryText) }
                } else if inventory.availability != .unavailableLegacy || RetainedSourceMedia.disclosable(entry.print.metadata) {
                    Text(RetainedSourceMedia.disclosure(inventory.availability) ?? "Source media unavailable.").foregroundStyle(.secondaryText)
                } else { Text("No retained source media.").foregroundStyle(.secondaryText) }
            } else if failure == nil { ProgressView("Loading sources…") }
            if let failure { Text(failure).foregroundStyle(.secondaryText) }
            if failure != nil || inventory?.availability != .available {
                Button("Retry sources") { retry += 1 }
            }
        }
        .task(id: SourceLoadKey(entry: entry, host: hosts.host(entry.hostID), instance: hosts.instanceID(of: entry.hostID), retry: retry)) {
            inventory = nil; failure = nil
            guard let host = hosts.host(entry.hostID) else { failure = "Connect this machine to read its sources."; return }
            let instance = hosts.instanceID(of: entry.hostID)
            do {
                let result = try await hosts.backend(for: host).retainedSourceMedia(for: entry.print.filename)
                guard !Task.isCancelled, hosts.host(entry.hostID) == host, hosts.instanceID(of: entry.hostID) == instance else { return }
                inventory = result
            } catch { if !Task.isCancelled { failure = error.localizedDescription } }
        }
    }
}
private struct SourceLoadKey: Hashable {
    let entry: LibraryEntry
    let host: MoldHost?
    let instance: String?
    let retry: Int
}
