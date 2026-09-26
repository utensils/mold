import Foundation
import MoldClient

/// Organization edits on their way to the machines. One chain per machine,
/// each drained by one task.
///
/// Its own object rather than more of `LibraryStore`: the store is the
/// merged timeline and what is showing in it, and this is a queue with a
/// retry policy, which is a different thing that happens to need the store
/// to report and repair. The optimistic change is already on screen by the
/// time anything here runs, so nothing below is about making an edit happen
/// -- it is about the screen eventually agreeing with the machine, or saying
/// so. WHEN to send, how long to wait and when to give up is
/// `MutationOutbox`'s own policy (`MutationOutbox+Policy`), so this keeps
/// only what needs the store: the wire call itself, the local report, and
/// the re-list.
@MainActor
@Observable
final class LibraryMutations {
    struct Destination {
        let host: MoldHost
        let client: any MoldBackend
    }

    /// The queued edits themselves. Read by the store's echo test, which
    /// asks what is in flight before trusting a frame.
    var outbox = MutationOutbox()
    /// The machines whose chain a task is already walking.
    private var draining: Set<MoldHost.ID> = []
    /// The exact address/key identity an edit was queued for. A retry can run
    /// seconds later; resolving by UUID then could send the old filenames to
    /// a completely different machine the person edited into that row.
    @ObservationIgnored private var destinations: [String: Destination] = [:]
    /// One line in the Library's activity bar while optimistic edits are still
    /// making their way to the machines. The count is print-updates rather
    /// than requests: one bulk request carrying 500 filenames should say that
    /// it has 500 things left to settle.
    private(set) var progress: String?
    private var targetCount = 0
    private var settledCount = 0

    /// Queues an edit for every machine it names, and answers with the ids of
    /// the entries carrying it -- what `undo` ties its registration to.
    @discardableResult
    func send(_ edit: PrintEdit, in store: LibraryStore) -> [String] {
        let queued = outbox.enqueue(edit)
        for entry in queued {
            guard let host = store.hosts.host(entry.host),
                  let client = store.hosts.backend(for: entry.host) else { continue }
            destinations[entry.id] = Destination(host: host, client: client)
        }
        targetCount += queued.reduce(0) { $0 + $1.filenames.count }
        updateProgress()
        for host in outbox.waiting { drain(host, in: store) }
        return queued.map(\.id)
    }

    private func drain(_ host: MoldHost.ID, in store: LibraryStore) {
        // One task per machine, so its chain stays a chain. A second edit
        // arriving mid-drain joins the queue the running task is already
        // walking.
        guard draining.insert(host).inserted else { return }
        Task {
            defer {
                draining.remove(host)
                if outbox.isEmpty, draining.isEmpty { finishProgress() }
            }
            // Round and round until the outbox is EMPTY, not until the chain
            // is walked once: this host stays in `draining` for the whole
            // task, including the trailing re-list and collection reload
            // below, and an edit enqueued during one of those round trips
            // calls `drain`, meets the guard and returns. Nothing would ever
            // walk it -- it sat on screen, applied optimistically, and reached
            // the machine only when some later unrelated edit kicked the
            // chain.
            while await drainOnce(host, in: store) {}
        }
    }

    /// Walks the chain once and does the work that follows it. `true` when
    /// something arrived while that was happening.
    private func drainOnce(_ host: MoldHost.ID, in store: LibraryStore) async -> Bool {
        var touchedCollections = false
        var lastError: Error?
        loop: while true {
            switch outbox.next(for: host) {
            case .idle:
                break loop
            case let .send(entry):
                if case .collection = entry.change { touchedCollections = true }
                updateProgress()
                lastError = await attempt(entry, on: host, in: store)
            case let .wait(duration, then: entry):
                if case .collection = entry.change { touchedCollections = true }
                updateProgress(retrying: true)
                try? await Task.sleep(for: duration)
                lastError = await attempt(entry, on: host, in: store)
            case let .giveUp(entry, orphaned: _):
                // `relist` re-reads the machine and replays what is still
                // queued, which is how these rows get repaired -- the same
                // thing `attempt`'s own give-up below does.
                store.hosts.report(lastError ?? MoldClientError.malformedResponse,
                                   on: host, doing: entry.change.verb)
                settleProgress(entry)
                await store.live.relist(host, in: store)
            }
        }
        // A frame from this machine was skipped as our own echo while the
        // chain was running. Ours is now settled, so what that frame might
        // ALSO have been saying -- another client editing the same row -- is
        // the only thing left, and reading the listing again is how it is
        // recovered.
        if store.live.echo.takeStale(host) { await store.live.relist(host, in: store) }
        // Membership moved, so every machine's collection counts are stale.
        if touchedCollections { await store.reloadCollections() }
        // Asked without mutating: `next(for:)` can retire an entry, and this
        // is a question, not a step.
        return !outbox.chain(for: host).isEmpty
    }

    /// Re-applies every edit still queued for a machine, in order, onto rows
    /// just read from it. Applied locally with `mutate`, never `apply`: these
    /// edits are already on their way to the machine, and re-enqueuing them
    /// would send them twice.
    func replayPending(on host: MoldHost.ID, in store: LibraryStore) {
        for pending in outbox.chain(for: host) {
            store.mutate(PrintEdit(change: pending.change, targets: [host: pending.filenames]))
        }
    }

    /// Called only after an outbox entry has left its chain for good. A retry
    /// deliberately does not advance the count: the same update is still in
    /// flight, using the same idempotency key.
    func settleProgress(_ entry: MutationOutbox.Entry) {
        destinations[entry.id] = nil
        settledCount += entry.filenames.count
        updateProgress()
    }

    func destination(for entry: MutationOutbox.Entry) -> Destination? {
        destinations[entry.id]
    }

    /// Retires an edit whose original machine identity disappeared. The
    /// optimistic row and its undo entry both become untrue, so the current
    /// machine (if the row was edited in place) is re-read without ever
    /// receiving the old identity's filenames.
    func rejectStale(_ entry: MutationOutbox.Entry, in store: LibraryStore) async {
        outbox.failed(entry.id)
        settleProgress(entry)
        store.undo.forget(entry: entry.id)
        if store.hosts.host(entry.host) != nil {
            await store.live.relist(entry.host, in: store)
        }
    }

    private func updateProgress(retrying: Bool = false) {
        guard targetCount > 0 else {
            progress = nil
            return
        }
        let verb = retrying ? "Retrying" : "Updating"
        progress = "\(verb) \(settledCount.formatted()) of \(targetCount.formatted()) print changes…"
    }

    private func finishProgress() {
        progress = nil
        targetCount = 0
        settledCount = 0
    }
}
