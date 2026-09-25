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
            || collectionsPerHost.contains(where: { !live.contains($0.key) })
        else { return }
        perHost = perHost.filter { live.contains($0.key) }
        trashPerHost = trashPerHost.filter { live.contains($0.key) }
        etags = etags.filter { live.contains($0.key) }
        trashEtags = trashEtags.filter { live.contains($0.key) }
        collectionsPerHost = collectionsPerHost.filter { live.contains($0.key) }
        tags.prune(to: live)
        rebuild()
        rebuildTrash()
    }

    /// One tile per PRINT, not per copy -- see `LibraryMerge`. Machines are
    /// walked in the machine list's order so which copy leads never depends
    /// on dictionary order.
    func rebuild() {
        let live = inMachineOrder(perHost).filter { $0.print.trashedAt == nil }
        items = LibraryMerge.merge(live, localHost: MoldEngine.localHostID, links: syncLinks())
            .sorted { $0.print.timestamp > $1.print.timestamp }
        rows.bump()
    }

    /// The trash merges the same way, so a print and its copy are put back
    /// or purged as one -- the desktop app's rule for its trash too.
    func rebuildTrash() {
        trashed = LibraryMerge.merge(inMachineOrder(trashPerHost), localHost: MoldEngine.localHostID,
                                     links: syncLinks())
            .sorted { ($0.print.trashedAt ?? 0) > ($1.print.trashedAt ?? 0) }
    }

    /// Any print by its own machine's id -- a lead OR a copy merged under
    /// another machine's tile. Lookups by id go through here, because a
    /// copy's id is no longer a row of `items`.
    func entry(_ id: PrintID) -> LibraryEntry? {
        (perHost[id.host] ?? []).first { $0.print.filename == id.filename }
            ?? (trashPerHost[id.host] ?? []).first { $0.print.filename == id.filename }
            ?? (items + trashed).lazy.flatMap(\.everyCopy).first { $0.id == id }
    }

    /// The tile a print is shown under, whichever copy `id` names.
    func tile(containing id: PrintID) -> LibraryEntry? {
        (items + trashed).first { tile in tile.everyCopy.contains { $0.id == id } }
    }

    private func inMachineOrder(_ rows: [MoldHost.ID: [LibraryEntry]]) -> [LibraryEntry] {
        let order = hosts.hosts.map(\.id)
        let known = order.flatMap { rows[$0] ?? [] }
        let rest = rows.filter { !order.contains($0.key) }.values.flatMap(\.self)
        return known + rest
    }

    func count(for host: MoldHost.ID) -> Int { perHost[host]?.count ?? 0 }
}
