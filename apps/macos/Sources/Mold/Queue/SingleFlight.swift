import Foundation
import MoldClient

/// One piece of work at a time per machine, and at most ONE queued behind it.
///
/// A caller arriving while work is running is asking "do it once more after
/// this", and however many arrive that is still once more -- so they all join
/// the same promise instead of each starting their own. Its own type rather
/// than more of `QueueStore`, which is already at the type-size budget.
///
/// Why anything needs this: `/api/events` emits one `.resyncRequired` per
/// DROPPED frame, and each one asks the queue to re-read. `Task.cancel()` is
/// no throttle -- a read that is not at a cancellation point does not stop --
/// so K markers were K concurrent `GET /api/queue` calls on the main actor,
/// which is what the consumer was already behind on. The repair widened the
/// gap it existed to close.
@MainActor
final class SingleFlight {
    private var inFlight: [MoldHost.ID: Task<Void, Never>] = [:]
    private var promised: [MoldHost.ID: Task<Void, Never>] = [:]

    /// Runs `work`, or joins what is already promised. Returns when the work
    /// this caller is owed has finished, so a caller cannot tell it was
    /// throttled except by counting requests.
    func once(per host: MoldHost.ID, do work: @escaping (MoldHost.ID) async -> Void) async {
        // Somebody has already promised a run for after the one in flight.
        // That promise answers this caller too.
        if let promised = promised[host] { return await promised.value }

        // Nothing running: this caller IS the one in flight.
        guard let running = inFlight[host] else {
            let mine = Task { await work(host) }
            inFlight[host] = mine
            await mine.value
            if inFlight[host] == mine { inFlight[host] = nil }
            return
        }

        // No `await` between the reads above and the writes below, so on the
        // main actor this claim is atomic.
        let follow = Task { [weak self] in
            await running.value
            guard let self else { return }
            // It is the one in flight now, and nothing is queued behind it --
            // so the NEXT caller promises a further run rather than joining
            // one that has already started.
            promised[host] = nil
            await work(host)
        }
        promised[host] = follow
        inFlight[host] = follow
        await follow.value
        if inFlight[host] == follow { inFlight[host] = nil }
    }
}
