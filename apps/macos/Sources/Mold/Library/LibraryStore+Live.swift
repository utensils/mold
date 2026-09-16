import Foundation
import MoldClient

// Reacting to what the machines say, instead of asking them on a timer.
@MainActor
extension LibraryStore {

    /// Starts applying live gallery events.
    ///
    /// The app still lists on launch and on ⌘R: these are DELTAS, and a client
    /// that has never read the listings has nothing to apply them to. What
    /// they replace is the polling in between.
    func listen() {
        guard listening == nil else { return }
        listening = hosts.onEvent { [weak self] host, event in
            self?.apply(event, from: host)
        }
    }

    private func apply(_ event: MoldEvent, from host: MoldHost.ID) {
        switch event {
        case .authority:
            break
        case .resyncRequired:
            // The stream admitted it dropped deltas, so nothing on screen for
            // this machine can be trusted. Reading the listing again is the
            // only honest repair -- the events that would have told us what
            // changed are the ones that went missing.
            Task { await relist(host) }
        case let .gallery(change):
            apply(change, from: host)
        }
    }

    private func apply(_ change: MoldEvent.Gallery, from host: MoldHost.ID) {
        // An edit this app made is already on screen, and the machine is
        // echoing it back. Applying it again is harmless for a row that rides
        // along and a wasted re-list for one that does not, so anything still
        // in flight for this machine means the echo is ours: skip it.
        guard outbox.chain(for: host).isEmpty else { return }

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
        guard case let .fresh(prints, etag) = try? await client.gallery(etag: nil) else { return }
        if let etag { etags[id] = etag }
        perHost[id] = prints.map { LibraryEntry(host: machine, print: $0) }
        replayPending(on: id)
        rebuild()
    }
}
