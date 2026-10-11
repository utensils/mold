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
        let instances = Dictionary(uniqueKeysWithValues: hosts.hosts.compactMap { host in
            hosts.instanceID(of: host.id).map { (host.id, $0) }
        })
        return sourceAccessIndex.snapshot(key: .init(revision: sourceAccessRevision,
            visibility: collectionVisibility.generation, hosts: hosts.hosts, instances: instances)) {
            let visible = LibraryScope.all.resolve(LibraryQuery(), shelves: shelves,
                hiddenCollectionIDs: hiddenCollectionIDs).apply(to: pool)
            return .init(visible: visible.flatMap(\.everyCopy), details: (pool + trashPool).flatMap(\.everyCopy),
                         hosts: hosts.hosts, instances: instances)
        }
    }
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
