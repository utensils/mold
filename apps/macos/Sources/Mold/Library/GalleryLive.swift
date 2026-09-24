import Foundation
import MoldClient

/// One machine's live frames, turned into changes to the rows on screen --
/// or, when a frame cannot say enough on its own, into a re-list.
///
/// Its own object rather than more of `LibraryStore`: the store is the merged
/// timeline and what is showing in it, and this is the RECONCILER. It decides
/// whether a frame is this app's own edit coming back, whether a delta can be
/// applied in place or the listing has to be read again, and it holds the two
/// pieces of memory only that question needs -- `GalleryEcho` and the
/// `RelistGate` that keeps a burst of resync markers down to one read and at
/// most one behind it. It takes the store as a parameter, the arrangement
/// `LibraryMutations` already has with it.
@MainActor
final class GalleryLive {
    /// Which live frames are this app's own edit coming back, and which
    /// machines are owed a re-list because one was skipped. See `GalleryEcho`.
    var echo = GalleryEcho()
    /// One resync-driven re-list per machine at a time. See `RelistGate`.
    let relists = RelistGate()

    /// One machine's frame.
    func apply(_ event: MoldEvent, from host: MoldHost.ID, in store: LibraryStore) {
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
            Task { await relists.run(host) { await relist(host, in: store) } }
        case let .gallery(change):
            apply(change, from: host, in: store)
        }
    }

    private func apply(_ change: MoldEvent.Gallery, from host: MoldHost.ID,
                       in store: LibraryStore) {
        // An edit this app made is already on screen, and the machine is
        // echoing it back. Applying it again is a wasted re-list -- but that
        // is only true of the ROW the edit names, and this guard used to be
        // the whole chain: a `gallery_added` for a render landing while a star
        // was in flight was discarded, and nothing re-listed afterwards. See
        // `GalleryEcho`, which also remembers the machine so `drain` repairs
        // the one case that remains.
        guard !echo.isEcho(change, on: host,
                           pending: store.mutations.outbox.chain(for: host)) else { return }

        switch change {
        case let .updated(filename, row), let .restored(filename, row):
            if let row { replace(filename, with: row, on: host, in: store) } else {
                Task { await relists.run(host) { await relist(host, in: store) } }
            }
        case let .added(_, row, imported):
            // The bulk-save task performs one final local refresh. Redrawing
            // the full grid for every imported picture makes a large save
            // quadratic, even though the event still reaches other clients.
            if imported && host == MoldEngine.localHostID && store.localSaveProgress != nil { return }
            if let row { insert(row, on: host, in: store) } else {
                Task { await relists.run(host) { await relist(host, in: store) } }
            }
        case let .removed(filename), let .trashed(filename):
            // Both take the print out of the live listing. The trash is its
            // own scope with its own ETag, so it re-reads when it is opened.
            drop(filename, on: host, in: store)
            store.trashEtags.removeAll()
        case .collectionsChanged:
            Task { await store.reloadCollections() }
        }
    }

    private func replace(_ filename: String, with print: GalleryPrint,
                         on host: MoldHost.ID, in store: LibraryStore) {
        store.perHost[host] = (store.perHost[host] ?? []).map {
            $0.print.filename == filename ? $0.replacingPrint(print) : $0
        }
        store.rebuild()
    }

    /// Adds a print this machine just reported.
    ///
    /// Reads the machine from `HostStore` rather than from an existing row on
    /// this host -- a host with nothing in `perHost` yet has no row to borrow
    /// a name from, and used to silently drop its first print.
    private func insert(_ print: GalleryPrint, on host: MoldHost.ID, in store: LibraryStore) {
        guard let machine = store.hosts.host(host) else { return }
        var list = store.perHost[host] ?? []
        guard !list.contains(where: { $0.print.filename == print.filename }) else { return }
        list.append(LibraryEntry(host: machine, print: print))
        store.perHost[host] = list
        store.rebuild()
    }

    private func drop(_ filename: String, on host: MoldHost.ID, in store: LibraryStore) {
        store.perHost[host] = (store.perHost[host] ?? []).filter { $0.print.filename != filename }
        store.rebuild()
    }

    /// Reads one machine's listing again and re-applies whatever is still
    /// queued for it. Its ETag goes first, or the machine answers 304 and the
    /// repair repairs nothing.
    ///
    /// Always replays the outbox, even on the plain re-list a `.resyncRequired`
    /// event asks for -- re-listing without replaying could drop an edit still
    /// in flight, the same defect the outbox's own repair exists to avoid.
    func relist(_ id: MoldHost.ID, in store: LibraryStore) async {
        guard let machine = store.hosts.host(id),
              let client = store.hosts.backend(for: id) else { return }
        store.etags[id] = nil
        // What this machine had when we ASKED, which is what tells its answer
        // apart from what happened while the answer was in flight. See
        // `RelistMerge`.
        let asked = Set((store.perHost[id] ?? []).map(\.print.filename))
        let fetched: Fetched<[GalleryPrint]>
        do {
            fetched = try await client.gallery(etag: nil)
        } catch {
            // The repair itself failing is the one thing the person has to be
            // told about: this machine's rows are known-stale and nothing
            // else is going to correct them. Swallowing it was how a resync
            // that never resynced looked exactly like one that worked.
            store.hosts.report(error, on: id, doing: "list its prints")
            return
        }
        guard case let .fresh(prints, etag) = fetched else { return }
        if let etag { store.etags[id] = etag }
        store.perHost[id] = RelistMerge.merged(
            answer: prints.map { LibraryEntry(host: machine, print: $0) },
            onScreen: store.perHost[id] ?? [], asked: asked)
        store.hosts.succeeded(on: id, doing: "list its prints")
        store.mutations.replayPending(on: id, in: store)
        store.rebuild()
    }
}
