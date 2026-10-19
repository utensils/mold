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
        switch event {
        case .resyncRequired: filename = nil
        case .gallery(let change):
            switch change {
            case .added(let name, _, _), .updated(let name, _), .restored(let name, _),
                 .removed(let name), .trashed(let name): filename = name
            case .collectionsChanged: return
            }
        default: return
        }
        sourceMediaCatalog.invalidate(host: host, filename: filename)
        Task { await SourceMediaPreview.invalidate(host: host, filename: filename) }
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
