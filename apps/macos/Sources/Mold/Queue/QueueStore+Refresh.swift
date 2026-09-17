import Foundation
import MoldClient

// How often this app is allowed to ask a machine for its queue. Split
// from `QueueStore.swift` past the file-size advisory; the read itself
// (`poll`) stays there, because every mutation calls it directly.
@MainActor
extension QueueStore {
    func refresh() async {
        isLoading = true
        defer { isLoading = false }
        await withTaskGroup(of: Void.self) { group in
            for host in hosts.hosts {
                group.addTask { await self.refresh(on: host.id) }
            }
        }
    }

    /// One machine's queue, then its batches -- what every fallback calls.
    ///
    /// ONE read in flight per machine, and at most ONE queued behind it. A
    /// caller arriving while a read is running is asking "read once more
    /// after this one", and however many of them arrive that is still one
    /// more read -- so they all join the same promise.
    ///
    /// Why it has to be throttled at all: the stream emits one
    /// `.resyncRequired` per DROPPED frame, and `QueueStore+Live` answers each
    /// with a refresh. `Task.cancel()` on the pending coalescer stops
    /// nothing, because neither `poll` nor `hydrateNow` is at a cancellation
    /// point, so K markers used to be K concurrent `GET /api/queue` calls and
    /// K chained batch-status reads -- on the main actor, which is what the
    /// consumer was already behind on. The repair widened the gap it existed
    /// to close.
    func refresh(on host: MoldHost.ID) async {
        // Somebody has already promised a read for after the one in flight.
        // That promise answers this caller too.
        if let promised = queuedRefreshes[host] { return await promised.value }

        guard let inFlight = refreshes[host] else {
            let mine = Task { [weak self] in
                guard let self else { return }
                await readOnce(on: host)
            }
            refreshes[host] = mine
            await mine.value
            if refreshes[host] == mine { refreshes[host] = nil }
            return
        }

        // No `await` between reading `refreshes[host]` above and both writes
        // below, so on the main actor this claim is atomic.
        let follow = Task { [weak self] in
            await inFlight.value
            guard let self else { return }
            // It is the one in flight now, and nothing is queued behind it --
            // so the NEXT caller promises a further read rather than joining
            // one that has already started.
            queuedRefreshes[host] = nil
            await readOnce(on: host)
        }
        queuedRefreshes[host] = follow
        refreshes[host] = follow
        await follow.value
        if refreshes[host] == follow { refreshes[host] = nil }
    }

    /// The actual read. Not `private`: `QueueStore+Batches` is a different
    /// file, and `private` does not cross one.
    func readOnce(on host: MoldHost.ID) async {
        await poll(host)
        // A superseded read stops here rather than spending a
        // batch-status call on a listing nobody is waiting for.
        guard !Task.isCancelled else { return }
        await hydrate(on: host)
    }
}
