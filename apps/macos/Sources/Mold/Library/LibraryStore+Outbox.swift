import Foundation
import MoldClient

// Draining the outbox. One chain per machine, each drained by one task.
//
// The optimistic change is already on screen by the time anything here runs,
// so nothing below is about making an edit happen -- it is about the screen
// eventually agreeing with the machine, or saying so. WHEN to send, how long
// to wait, and when to give up is `MutationOutbox`'s own policy now -- see
// `MutationOutbox+Policy` -- so this file keeps only what needs the store:
// the wire call itself, the local report, and `relist`.
@MainActor
extension LibraryStore {

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
            var lastError: Error?
            loop: while true {
                switch outbox.next(for: host) {
                case .idle:
                    break loop
                case let .send(entry):
                    if case .collection = entry.change { touchedCollections = true }
                    lastError = await attempt(entry, on: host)
                case let .wait(duration, then: entry):
                    if case .collection = entry.change { touchedCollections = true }
                    try? await Task.sleep(for: duration)
                    lastError = await attempt(entry, on: host)
                case let .giveUp(entry, orphaned: _):
                    // `relist` re-reads the machine and replays what is
                    // still queued, which is how these rows get repaired --
                    // the same thing `attempt`'s own give-up below does.
                    hosts.report(lastError ?? MoldClientError.malformedResponse,
                                 on: host, doing: entry.change.verb)
                    await relist(host)
                }
            }
            // A frame from this machine was skipped as our own echo while the
            // chain was running. Ours is now settled, so what that frame might
            // ALSO have been saying -- another client editing the same row --
            // is the only thing left, and reading the listing again is how it
            // is recovered.
            if echo.takeStale(host) { await relist(host) }
            // Membership moved, so every machine's collection counts are stale.
            if touchedCollections { await reloadCollections() }
        }
    }

    /// Sends one entry and settles it with the outbox: gone on success, kept
    /// for another round on a transient failure, gone and reported on
    /// anything else. Returns the error, if any, so a later give-up still
    /// has something to tell the person.
    @discardableResult
    private func attempt(_ entry: MutationOutbox.Entry, on host: MoldHost.ID) async -> Error? {
        guard let client = hosts.backend(for: host) else {
            // The machine was removed. Its rows went with it.
            outbox.failed(entry.id)
            return nil
        }
        do {
            try await send(entry, to: client)
            outbox.succeeded(entry.id)
            hosts.succeeded(on: host)
            return nil
        } catch {
            if (error as? MoldClientError)?.isTransient == true {
                outbox.retry(entry.id)
            } else {
                hosts.report(error, on: host, doing: entry.change.verb)
                outbox.failed(entry.id)
                // The inverse was registered synchronously, before anything
                // was sent -- it has to be, or `UndoManager` files it on the
                // undo stack instead of the redo one. The machine has now
                // refused, and `relist` is about to put the row back, so "Undo
                // Favorite" would offer to reverse a favourite that never
                // happened: a local no-op and a redundant mutation, and worse
                // the day a change is not idempotent. Only THIS store's
                // entries go; a field editor's are its own.
                undo.forget()
                await relist(host)
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
