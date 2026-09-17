import Foundation
import MoldClient

// The merged timeline itself: what is in it, and saying when it changed.
// Split from the store for size.
@MainActor
extension LibraryStore {

    /// Drops machines that are no longer in the list, so their prints don't
    /// linger. Called from `refresh`, because removing a machine is exactly
    /// when nobody thinks to reload the library.
    func prune(to hostList: [MoldHost]) {
        let live = Set(hostList.map(\.id))
        guard perHost.contains(where: { !live.contains($0.key) })
            || trashPerHost.contains(where: { !live.contains($0.key) })
        else { return }
        perHost = perHost.filter { live.contains($0.key) }
        trashPerHost = trashPerHost.filter { live.contains($0.key) }
        etags = etags.filter { live.contains($0.key) }
        trashEtags = trashEtags.filter { live.contains($0.key) }
        collectionsPerHost = collectionsPerHost.filter { live.contains($0.key) }
        tagsPerHost = tagsPerHost.filter { live.contains($0.key) }
        rebuild()
        trashed = trashPerHost.values.flatMap(\.self)
            .sorted { ($0.print.trashedAt ?? 0) > ($1.print.trashedAt ?? 0) }
    }

    func rebuild() {
        items = perHost.values.flatMap(\.self)
            .filter { $0.print.trashedAt == nil }
            .sorted { $0.print.timestamp > $1.print.timestamp }
        rows.bump()
    }

    func count(for host: MoldHost.ID) -> Int { perHost[host]?.count ?? 0 }
}
