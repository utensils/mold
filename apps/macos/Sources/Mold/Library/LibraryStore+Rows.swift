import Foundation
import MoldClient

/// Ids whose rows changed since the timelines last settled.
struct PendingRows {
    var full = false
    var upserted: Set<PrintID> = []
    var removed: Set<PrintID> = []
    var isEmpty: Bool { !full && upserted.isEmpty && removed.isEmpty }

    mutating func note(_ changed: [PrintID], removed gone: [PrintID] = []) {
        upserted.formUnion(changed)
        upserted.subtract(gone)
        removed.formUnion(gone)
        removed.subtract(changed)
    }
}

// The merged timeline itself: what is in it, and saying when it changed.
// Split from the store for size.
@MainActor
extension LibraryStore {

    /// Drops machines that are no longer in the list, so their prints don't
    /// linger. Called from `refresh`, because removing a machine is exactly
    /// when nobody thinks to reload the library.
    func prune(to hostList: [MoldHost]) {
        let live = Set(hostList.map(\.id))
        discardReplicas(keeping: live)
        guard perHost.contains(where: { !live.contains($0.key) })
            || trashPerHost.contains(where: { !live.contains($0.key) })
            || collectionsPerHost.contains(where: { !live.contains($0.key) })
        else { return }
        perHost = perHost.filter { live.contains($0.key) }
        trashPerHost = trashPerHost.filter { live.contains($0.key) }
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
        liveTimeline.rebuild(live, localHost: MoldEngine.localHostID, links: syncLinks())
        pendingLive = PendingRows()
        items = liveTimeline.tiles
        refreshUnreadMedia()
        rows.bump()
    }

    /// The trash merges the same way, so a print and its copy are put back
    /// or purged as one -- the desktop app's rule for its trash too.
    func rebuildTrash() {
        trashTimeline.rebuild(inMachineOrder(trashPerHost), localHost: MoldEngine.localHostID,
                              links: syncLinks())
        pendingTrash = PendingRows()
        trashed = trashTimeline.tiles
        rows.bump()
    }

    /// Replaces one machine's live rows from a path that notes what it
    /// changed (`changedLive`), so the timeline can be patched rather than
    /// rebuilt.
    func writeLive(_ host: MoldHost.ID, _ list: [LibraryEntry]) {
        writingRows = true
        perHost[host] = list
        writingRows = false
    }

    /// Notes rows this store just changed in place and settles them, unless
    /// a caller is batching (`absorb` replaying the outbox).
    func changedLive(_ ids: [PrintID], removed: [PrintID] = []) {
        pendingLive.note(ids, removed: removed)
        settleTimelines()
    }

    /// Brings both timelines up to the rows: in place where a merge could not
    /// differ, by a full merge otherwise, and not at all -- no re-sort, no
    /// `rows.bump()` -- when nothing changed.
    func settleTimelines() {
        guard deferredSettles == 0 else { return }
        let links = pendingLive.isEmpty && pendingTrash.isEmpty ? [:] : syncLinks()
        if pendingLive.full {
            rebuild()
        } else if !pendingLive.isEmpty {
            let (changed, gone) = current(pendingLive, in: perHost, live: true)
            pendingLive = PendingRows()
            if liveTimeline.patch(upserted: changed, removed: gone, links: links) {
                items = liveTimeline.tiles
                refreshUnreadMedia()
                rows.bump()
            } else {
                rebuild()
            }
        }
        if pendingTrash.full {
            rebuildTrash()
        } else if !pendingTrash.isEmpty {
            let (changed, gone) = current(pendingTrash, in: trashPerHost, live: false)
            pendingTrash = PendingRows()
            if trashTimeline.patch(upserted: changed, removed: gone, links: links) {
                trashed = trashTimeline.tiles
                rows.bump()
            } else {
                rebuildTrash()
            }
        }
    }

    /// The rows as they are NOW for a pending set: an id noted as changed
    /// whose row has since gone (or left this partition) is a removal.
    private func current(_ pending: PendingRows, in rows: [MoldHost.ID: [LibraryEntry]],
                         live: Bool) -> ([LibraryEntry], [PrintID]) {
        var lookup: [MoldHost.ID: [String: LibraryEntry]] = [:]
        for host in Set(pending.upserted.map(\.host)) {
            lookup[host] = Dictionary((rows[host] ?? []).map { ($0.print.filename, $0) },
                                      uniquingKeysWith: { $1 })
        }
        var changed: [LibraryEntry] = []
        var gone = Array(pending.removed)
        for id in pending.upserted {
            // The live timeline holds only unstamped rows (`rebuild`'s own
            // filter); the trash holds whatever its listing put there.
            if let entry = lookup[id.host]?[id.filename], !live || entry.print.trashedAt == nil {
                changed.append(entry)
            } else {
                gone.append(id)
            }
        }
        return (changed, gone)
    }

    /// Any print by its own machine's id -- a lead OR a copy merged under
    /// another machine's tile. Indexed, so a lookup per selected row or menu
    /// item never walks the whole library.
    func entry(_ id: PrintID) -> LibraryEntry? {
        if pendingLive.isEmpty, pendingTrash.isEmpty,
           let copy = liveTimeline.copy(id) ?? trashTimeline.copy(id) {
            return copy
        }
        return (perHost[id.host] ?? []).first { $0.print.filename == id.filename }
            ?? (trashPerHost[id.host] ?? []).first { $0.print.filename == id.filename }
            ?? (items + trashed).lazy.flatMap(\.everyCopy).first { $0.id == id }
    }

    /// The tile a print is shown under, whichever copy `id` names.
    func tile(containing id: PrintID) -> LibraryEntry? {
        liveTimeline.tile(containing: id) ?? trashTimeline.tile(containing: id)
    }

    private func inMachineOrder(_ rows: [MoldHost.ID: [LibraryEntry]]) -> [LibraryEntry] {
        let order = hosts.hosts.map(\.id)
        let known = order.flatMap { rows[$0] ?? [] }
        let rest = rows.filter { !order.contains($0.key) }.values.flatMap(\.self)
        return known + rest
    }

    func count(for host: MoldHost.ID) -> Int { perHost[host]?.count ?? 0 }
}
