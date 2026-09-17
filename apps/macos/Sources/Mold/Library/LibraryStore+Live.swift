import Foundation
import MoldClient

// Reacting to what the machines say, instead of asking them on a timer.
@MainActor
extension LibraryStore {

    /// One machine's frame. Registered from `init`, so it is `internal`
    /// rather than `private`: `private` does not cross a file boundary, even
    /// within one type.
    func apply(_ event: MoldEvent, from host: MoldHost.ID) {
        switch event {
        case .authority:
            break
        // A device's lifecycle or runtime state is not a gallery concern --
        // `MachineStore` is what reads `/api/devices` again.
        case .deviceStateChanged:
            break
        // Queue lifecycle is `QueueStore`'s concern (M6), not the gallery's.
        case .job, .queue:
            break
        case .resyncRequired:
            // The stream admitted it dropped deltas, so nothing on screen for
            // this machine can be trusted. Reading the listing again is the
            // only honest repair -- the events that would have told us what
            // changed are the ones that went missing.
            // Through the gate: a burst of markers -- and every reconnect
            // emits one -- is one read and at most one more behind it, rather
            // than K concurrent reads of the same index racing to assign it.
            Task { await relists.run(host) { await relist(host) } }
        case let .gallery(change):
            apply(change, from: host)
        }
    }

    private func apply(_ change: MoldEvent.Gallery, from host: MoldHost.ID) {
        // An edit this app made is already on screen, and the machine is
        // echoing it back. Applying it again is a wasted re-list -- but that
        // is only true of the ROW the edit names, and this guard used to be
        // the whole chain: a `gallery_added` for a render landing while a star
        // was in flight was discarded, and nothing re-listed afterwards. See
        // `GalleryEcho`, which also remembers the machine so `drain` repairs
        // the one case that remains.
        guard !echo.isEcho(change, on: host, pending: outbox.chain(for: host)) else { return }

        switch change {
        case let .updated(filename, row), let .restored(filename, row):
            if let row { replace(filename, with: row, on: host) } else {
                Task { await relist(host) }
            }
        case let .added(_, row):
            if let row { insert(row, on: host) } else {
                Task { await relist(host) }
            }
        case let .removed(filename), let .trashed(filename):
            // Both take the print out of the live listing. The trash is its
            // own scope with its own ETag, so it re-reads when it is opened.
            drop(filename, on: host)
            trashEtags.removeAll()
        case .collectionsChanged:
            Task { await reloadCollections() }
        }
    }

    private func replace(_ filename: String, with print: GalleryPrint, on host: MoldHost.ID) {
        perHost[host] = (perHost[host] ?? []).map {
            $0.print.filename == filename ? $0.replacingPrint(print) : $0
        }
        rebuild()
    }

    /// Adds a print this machine just reported.
    ///
    /// Reads the machine from `HostStore` rather than from an existing row on
    /// this host -- a host with nothing in `perHost` yet has no row to borrow
    /// a name from, and used to silently drop its first print.
    private func insert(_ print: GalleryPrint, on host: MoldHost.ID) {
        guard let machine = hosts.host(host) else { return }
        var list = perHost[host] ?? []
        guard !list.contains(where: { $0.print.filename == print.filename }) else { return }
        list.append(LibraryEntry(host: machine, print: print))
        perHost[host] = list
        rebuild()
    }

    private func drop(_ filename: String, on host: MoldHost.ID) {
        perHost[host] = (perHost[host] ?? []).filter { $0.print.filename != filename }
        rebuild()
    }

    /// Reads one machine's listing again and re-applies whatever is still
    /// queued for it. Its ETag goes first, or the machine answers 304 and the
    /// repair repairs nothing.
    ///
    /// Always replays the outbox, even on the plain re-list a `.resyncRequired`
    /// event asks for -- re-listing without replaying could drop an edit still
    /// in flight, the same defect the outbox's own repair exists to avoid.
    func relist(_ id: MoldHost.ID) async {
        guard let machine = hosts.host(id), let client = hosts.backend(for: id) else { return }
        etags[id] = nil
        // What this machine had when we ASKED, which is what tells its answer
        // apart from what happened while the answer was in flight. See
        // `RelistMerge`.
        let asked = Set((perHost[id] ?? []).map(\.print.filename))
        let fetched: Fetched<[GalleryPrint]>
        do {
            fetched = try await client.gallery(etag: nil)
        } catch {
            // The repair itself failing is the one thing the person has to be
            // told about: this machine's rows are known-stale and nothing
            // else is going to correct them. Swallowing it was how a resync
            // that never resynced looked exactly like one that worked.
            hosts.report(error, on: id, doing: "list its prints")
            return
        }
        guard case let .fresh(prints, etag) = fetched else { return }
        if let etag { etags[id] = etag }
        perHost[id] = RelistMerge.merged(
            answer: prints.map { LibraryEntry(host: machine, print: $0) },
            onScreen: perHost[id] ?? [], asked: asked)
        hosts.succeeded(on: id, doing: "list its prints")
        replayPending(on: id)
        rebuild()
    }
}
