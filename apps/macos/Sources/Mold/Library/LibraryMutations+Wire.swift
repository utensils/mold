import Foundation
import MoldClient

// One queued edit's round trip: sending it, and settling it with the outbox
// afterwards. Split from the drain itself for size; `internal` rather than
// `private` because `private` does not cross a file boundary even within one
// type.
@MainActor
extension LibraryMutations {
    /// Sends one entry and settles it with the outbox: gone on success, kept
    /// for another round on a transient failure, gone and reported on
    /// anything else. Returns the error, if any, so a later give-up still
    /// has something to tell the person.
    @discardableResult
    func attempt(
        _ entry: MutationOutbox.Entry, on host: MoldHost.ID, in store: LibraryStore
    ) async -> Error? {
        guard let destination = destination(for: entry),
              store.hosts.host(host) == destination.host else {
            await rejectStale(entry, in: store)
            return nil
        }
        do {
            try await send(entry, to: destination.client)
            guard store.hosts.host(host) == destination.host else {
                await rejectStale(entry, in: store)
                return nil
            }
            outbox.succeeded(entry.id)
            settleProgress(entry)
            store.undo.settled(entry: entry.id)
            store.hosts.succeeded(on: host)
            return nil
        } catch {
            guard store.hosts.host(host) == destination.host else {
                await rejectStale(entry, in: store)
                return nil
            }
            if (error as? MoldClientError)?.isTransient == true {
                outbox.retry(entry.id)
            } else {
                store.hosts.report(error, on: host, doing: entry.change.verb)
                outbox.failed(entry.id)
                settleProgress(entry)
                // The inverse was registered synchronously, before anything
                // was sent -- it has to be, or `UndoManager` files it on the
                // undo stack instead of the redo one. The machine has now
                // refused, and `relist` is about to put the row back, so "Undo
                // Favorite" would offer to reverse a favourite that never
                // happened: a local no-op and a redundant mutation, and worse
                // the day a change is not idempotent. THIS entry's inverse
                // goes, and only it -- a favourite that succeeded a moment ago
                // is still undoable.
                store.undo.forget(entry: entry.id)
                await store.live.relist(host, in: store)
            }
            return error
        }
    }

    /// Sends one queued entry in its wire form.
    private func send(_ entry: MutationOutbox.Entry, to client: any MoldBackend) async throws {
        switch entry.wire {
        case let .patch(patch, filenames):
            for filename in filenames {
                try await client.patch(filename, with: patch)
            }
        case let .mutate(mutation):
            try await client.mutate(mutation)
        }
    }
}
