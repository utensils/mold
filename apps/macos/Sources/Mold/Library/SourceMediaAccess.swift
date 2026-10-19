import MoldClient
import SwiftUI

extension LibraryStore {
    /// Includes hide-only receipt links, which may change without row edits.
    struct SourcePrivacyRevision: Equatable {
        let rows: Int
        let receipts: Int?
    }
    var sourcePrivacyRevision: SourcePrivacyRevision {
        .init(rows: sourceAccessRevision, receipts: hiddenCollectionIDs.isEmpty ? nil : syncRecords.privacyRevision)
    }

    func sourceMediaChanged(_ event: MoldEvent, host: UUID) {
        let filename: String?
        let preservingDisplay: Bool
        switch event {
        case .resyncRequired:
            filename = nil; preservingDisplay = true
        case .gallery(let change):
            switch change {
            case .updated(let name, let updated):
                let previous = entry(.init(host: host, filename: name))?.print
                if let previous, let updated,
                   SourceMediaAuthority.isDisplayOnlyUpdate(previous: previous, updated: updated) { return }
                filename = name; preservingDisplay = true
            case .added(let name, _, _), .restored(let name, _):
                filename = name; preservingDisplay = true
            case .removed(let name), .trashed(let name):
                filename = name; preservingDisplay = false
            case .collectionsChanged: return
            }
        default: return
        }
        let route = hosts.host(host).map { host in var route = host; route.name = ""; return route }
        let instance = hosts.instanceIDs[host]
        Task { @MainActor [weak self] in
            // Publish the tile's next epoch only after cancellation of OLD
            // pixel work completes. Reversing these steps cancels the new
            // tile load and can leave it blank until it is scrolled back in.
            await SourceMediaPreview.invalidate(host: host, filename: filename)
            guard let self else { return }
            let current = self.hosts.host(host).map { host in var route = host; route.name = ""; return route }
            guard current == route, self.hosts.instanceIDs[host] == instance else { return }
            self.sourceMediaCatalog.invalidate(host: host, filename: filename,
                                               preservingDisplay: preservingDisplay)
        }
    }
    var sourceAccess: SourceMediaAccessIndex.Snapshot {
        sourceAccessIndex.snapshot(key: .init(revision: sourceAccessRevision,
            visibility: collectionVisibility.generation, hosts: hosts.hosts, instances: hosts.instanceIDs,
            privacyRevision: nil)) {
            let visible = items
            return .init(visible: visible.flatMap(\.everyCopy), details: (items + trashed).flatMap(\.everyCopy),
                         hosts: hosts.hosts, instances: hosts.instanceIDs)
        }
    }
    /// Source Media explicitly includes every live collection, including hidden ones.
    var sourceMediaOwners: [LibraryEntry] { sourceAccess.owners }
    func allowsSource(_ authority: SourceMediaAuthority, detail: Bool = false) -> Bool {
        sourceAccess.allows(authority, detail: detail)
    }
}

/// Runs even when Sources is closed, including while an inspector remains open.
struct SourceMediaAccessMonitor: ViewModifier {
    @Environment(HostStore.self) private var hosts
    @Environment(LibraryStore.self) private var library
    func body(content: Content) -> some View {
        let active = !library.openSourceBrowsers.isEmpty || !library.openSourceDetails.isEmpty
        let access = active ? library.sourceAccess : nil
        let authorities = access?.authorities ?? []
        let detailAuthorities = Set(library.openSourceDetails.values.filter { access?.allows($0, detail: true) == true })
        let previewAuthorities = authorities.union(detailAuthorities)
        content.task(id: previewAuthorities) {
            guard active else { return }
            library.sourceMediaCatalog.retain(authorities: authorities)
            await SourceMediaPreview.retain(authorities: previewAuthorities)
        }
    }
}
