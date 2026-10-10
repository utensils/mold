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
    /// Unfiltered visible publications: source privacy is independent of the open pane.
    var sourceMediaOwners: [LibraryEntry] {
        let projected = LibraryScope.all.resolve(LibraryQuery(), shelves: shelves,
            hiddenCollectionIDs: hiddenCollectionIDs).apply(to: pool)
        return projected.flatMap(\.everyCopy)
    }
    func allowsSource(_ authority: SourceMediaAuthority, detail: Bool = false) -> Bool {
        let owners = detail ? (pool + trashPool).flatMap(\.everyCopy) : sourceMediaOwners
        return owners.contains { owner in
            owner.id == authority.owner && SourceMediaAuthority(owner: owner,
                route: hosts.host(owner.hostID), instance: hosts.instanceID(of: owner.hostID)) == authority
        }
    }
}

/// Runs even when Sources is closed, including while an inspector remains open.
struct SourceMediaAccessMonitor: ViewModifier {
    @Environment(HostStore.self) private var hosts
    @Environment(LibraryStore.self) private var library
    func body(content: Content) -> some View {
        let authorities = Set(library.sourceMediaOwners.map { owner in
            SourceMediaAuthority(owner: owner, route: hosts.host(owner.hostID),
                instance: hosts.instanceID(of: owner.hostID))
        })
        let detailAuthorities = Set(library.openSourceDetails.values.filter { library.allowsSource($0, detail: true) })
        let previewAuthorities = authorities.union(detailAuthorities)
        content.task(id: previewAuthorities) {
            library.sourceMediaCatalog.retain(authorities: authorities)
            await SourceMediaPreview.retain(authorities: previewAuthorities)
        }
    }
}
