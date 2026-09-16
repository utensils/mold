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

    func send(_ edit: PrintEdit, backend: @escaping (MoldHost.ID) -> (any MoldBackend)?) {
        outbox.enqueue(edit)
        for host in outbox.waiting { drain(host, backend) }
    }

    private func drain(_ host: MoldHost.ID, _ backend: @escaping (MoldHost.ID) -> (any MoldBackend)?) {
        // One task per machine, so its chain stays a chain. A second edit
        // arriving mid-drain joins the queue the running task is already
        // walking.
        guard draining.insert(host).inserted else { return }
        Task {
            defer { draining.remove(host) }
            var touchedCollections = false
            while let entry = outbox.head(for: host) {
                if case .collection = entry.change { touchedCollections = true }
                guard let client = backend(host) else {
                    // The machine was removed. Its rows went with it.
                    outbox.failed(entry.id)
                    continue
                }
                do {
                    try await send(entry, to: client)
                    outbox.succeeded(entry.id)
                    failures[host] = nil
                } catch {
                    let transient = (error as? MoldClientError)?.isTransient ?? false
                    if transient, entry.attempts < Self.maxAttempts {
                        outbox.retry(entry.id)
                        try? await Task.sleep(for: .seconds(pow(2.0, Double(entry.attempts - 1))))
                    } else {
                        failures[host] = entry.change.failureSentence
                        await repair(outbox.failed(entry.id), on: host, backend)
                    }
                }
            }
            // Membership moved, so every machine's collection counts are stale.
            if touchedCollections { await reloadCollections(backend) }
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

    /// Puts the named rows back to what the machine says they are.
    ///
    /// Re-lists the machine rather than patching the rows by hand, because the
    /// server is the authority and this app has just proved it does not know
    /// what happened. Everything still queued for that machine is then applied
    /// again on top, so a row with a newer edit in flight keeps showing that
    /// newer intent -- which is the same rule `failed` uses to decide what to
    /// repair at all.
    private func repair(_ filenames: [String], on host: MoldHost.ID,
                        _ backend: @escaping (MoldHost.ID) -> (any MoldBackend)?) async {
        guard !filenames.isEmpty, let client = backend(host) else { return }
        guard let name = hostName(host) else { return }
        etags[host] = nil
        guard case let .fresh(prints, etag) = try? await client.gallery(etag: nil) else { return }
        if let etag { etags[host] = etag }
        perHost[host] = prints.map { LibraryEntry(hostID: host, hostName: name, print: $0) }
        replayPending(on: host)
        rebuild()
    }

    /// Re-applies every edit still queued for a machine, in order.
    private func replayPending(on host: MoldHost.ID) {
        for pending in outbox.chain(for: host) {
            mutate(PrintEdit(change: pending.change, targets: [host: pending.filenames]))
        }
    }

    private func hostName(_ host: MoldHost.ID) -> String? {
        perHost[host]?.first?.hostName ?? trashPerHost[host]?.first?.hostName
    }
}
