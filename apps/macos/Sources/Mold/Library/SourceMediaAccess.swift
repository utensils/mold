import MoldClient
import SwiftUI

extension LibraryStore {
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
            visibility: collectionVisibility.generation, hosts: hosts.hosts, instances: hosts.instanceIDs)) {
            let currentShelves = shelves
            let visible = LibraryScope.all.resolve(LibraryQuery(), shelves: currentShelves,
                hiddenCollectionIDs: CollectionShelf.hiddenIDs(in: currentShelves), privacyEvidence: trashed, privacyLinks: syncLinks()).apply(to: items)
            return .init(visible: visible.flatMap(\.everyCopy), details: (items + trashed).flatMap(\.everyCopy),
                         hosts: hosts.hosts, instances: hosts.instanceIDs)
        }
    }
    /// Unfiltered visible publications: source privacy is independent of the open pane.
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
        let access = library.sourceAccess
        let authorities = access.authorities
        let detailAuthorities = Set(library.openSourceDetails.values.filter { access.allows($0, detail: true) })
        let previewAuthorities = authorities.union(detailAuthorities)
        content.task(id: previewAuthorities) {
            library.sourceMediaCatalog.retain(authorities: authorities)
            await SourceMediaPreview.retain(authorities: previewAuthorities)
        }
    }
}
