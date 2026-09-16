import Foundation
import MoldClient

// Draining the outbox. One chain per machine, each drained by one task.
//
// The optimistic change is already on screen by the time anything here runs,
// so nothing below is about making an edit happen -- it is about the screen
// eventually agreeing with the machine, or saying so.
@MainActor
extension LibraryStore {

    /// How many times one entry is sent before we stop and repair.
    ///
    /// Four, with a widening wait: a Tailscale link that drops usually comes
    /// back inside a few seconds, and a machine that is still gone after ~15
    /// is gone in a way waiting will not fix. The person is told then, rather
    /// than watching an edit hang indefinitely on a machine they turned off.
    private static let maxAttempts = 4

    func send(_ edit: PrintEdit) {
        outbox.enqueue(edit)
        for host in outbox.waiting { drain(host) }
    }

    private func drain(_ host: MoldHost.ID) {
        // One task per machine, so its chain stays a chain. A second edit
        // arriving mid-drain joins the queue the running task is already
        // walking.
        guard draining.insert(host).inserted else { return }
        Task {
            defer { draining.remove(host) }
            var touchedCollections = false
            while let entry = outbox.head(for: host) {
                if case .collection = entry.change { touchedCollections = true }
                guard let client = hosts.backend(for: host) else {
                    // The machine was removed. Its rows went with it.
                    outbox.failed(entry.id)
                    continue
                }
                do {
                    try await send(entry, to: client)
                    outbox.succeeded(entry.id)
                    hosts.succeeded(on: host)
                } catch {
                    let transient = (error as? MoldClientError)?.isTransient ?? false
                    if transient, entry.attempts < Self.maxAttempts {
                        outbox.retry(entry.id)
                        try? await Task.sleep(for: .seconds(pow(2.0, Double(entry.attempts - 1))))
                    } else {
                        hosts.report(error, on: host, doing: entry.change.verb)
                        outbox.failed(entry.id)
                        await relist(host)
                    }
                }
            }
            // Membership moved, so every machine's collection counts are stale.
            if touchedCollections { await reloadCollections() }
        }
    }

    /// Sends one queued entry.
    ///
    /// A title is the one change that is not a bulk mutation: it is a PATCH on
    /// a single print, and so it carries no operation id and no fence. That is
    /// safe precisely because it is idempotent -- setting a title twice is
    /// setting a title -- where adding a tag twice would not be.
    private func send(_ entry: MutationOutbox.Entry, to client: any MoldBackend) async throws {
        if case let .title(_, to) = entry.change {
            for filename in entry.filenames {
                try await client.patch(filename, with: GalleryPatch(title: to))
            }
            return
        }
        try await client.mutate(mutation(for: entry))
    }

    /// The wire form of one queued entry.
    private func mutation(for entry: MutationOutbox.Entry) -> GalleryBulkMutation {
        // The entry's own id, every attempt: the host applies a given
        // operation once, so reusing it is what makes a retry safe.
        switch entry.change {
        case let .favorite(on):
            GalleryBulkMutation(filenames: entry.filenames, favorite: on,
                                operationId: entry.id)
        case let .tag(name, adding):
            GalleryBulkMutation(filenames: entry.filenames,
                                addTags: adding ? [name] : [],
                                removeTags: adding ? [] : [name],
                                operationId: entry.id)
        case let .collection(name, slug, filing):
            GalleryBulkMutation(filenames: entry.filenames,
                                addToCollection: filing ? .named(name) : nil,
                                removeFromCollectionSlug: filing ? nil : slug,
                                operationId: entry.id)
        case .title:
            // Unreachable: `send` takes titles down the PATCH route above.
            GalleryBulkMutation(filenames: entry.filenames, operationId: entry.id)
        }
    }

    /// Re-applies every edit still queued for a machine, in order, onto rows
    /// just read from it. Applied locally with `mutate`, never `apply`: these
    /// edits are already on their way to the machine, and re-enqueuing them
    /// would send them twice.
    func replayPending(on host: MoldHost.ID) {
        for pending in outbox.chain(for: host) {
            mutate(PrintEdit(change: pending.change, targets: [host: pending.filenames]))
        }
    }
}
