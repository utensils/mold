import MoldClient
import SwiftUI

extension LibraryStore {
    func sourceMediaChanged(_ event: MoldEvent, host: UUID) {
        let filename: String?
        let preservingDisplay: Bool
        switch event {
        case .resyncRequired:
            filename = nil; preservingDisplay = true
        case .gallery(let change):
            switch change {
            case .updated(let name, let updated):
                let previous = (pendingLive.isEmpty ? liveTimeline.copy(.init(host: host, filename: name))?.print : nil)
                    ?? live[host]?.first { $0.filename == name }
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
        let instance = hosts.instanceID(of: host)
        Task { @MainActor [weak self] in
            // Publish the tile's next epoch only after cancellation of OLD
            // pixel work completes. Reversing these steps cancels the new
            // tile load and can leave it blank until it is scrolled back in.
            await SourceMediaPreview.invalidate(host: host, filename: filename)
            guard let self else { return }
            let current = self.hosts.host(host).map { host in var route = host; route.name = ""; return route }
            guard current == route, self.hosts.instanceID(of: host) == instance else { return }
            self.sourceMediaCatalog.invalidate(host: host, filename: filename,
                                               preservingDisplay: preservingDisplay)
        }
    }
    var sourceAccess: SourceMediaAccessIndex.Snapshot {
        let instances = Dictionary(uniqueKeysWithValues: hosts.hosts.compactMap { host in
            hosts.instanceID(of: host.id).map { (host.id, $0) }
        })
        return sourceAccessIndex.snapshot(key: .init(revision: sourceAccessRevision,
            visibility: collectionVisibility.generation, hosts: hosts.hosts, instances: instances)) {
            let visible = pool
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
