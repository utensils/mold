import Foundation
import MoldClient

// Reacting to what the machines say, instead of asking them on a timer.
@MainActor
extension QueueStore {

    /// One machine's frame. Registered from `init`, so it is `internal`
    /// rather than `private`: `private` does not cross a file boundary, even
    /// within one type.
    ///
    /// Nothing here re-lists on every event. `job_queued` and `job_started`
    /// change a row's state and its lane, which the listing owns, so they
    /// mark the machine dirty and a coalescing task re-reads it once -- a
    /// batch of 64 admitted at once is 64 frames and must be one read.
    /// `job_ended` is NOT a success signal (`types.rs:13153-13157`); it drops
    /// the row and says nothing about why, which is fine here because the
    /// re-read is what actually removes it.
    func apply(_ event: MoldEvent, from host: MoldHost.ID) {
        switch event {
        // `stateCommitted` is one durable child, `statesCommitted` is many
        // at once (a cancel-all, a batch cancel) -- the server emits the
        // latter explicitly so a bulk commit reconciles once
        // (`types.rs:13163-13167`). The coalescer already treats a burst of
        // any of these as one re-read, so every case takes the same path.
        case .job:
            markDirty(host)
        case let .queue(change):
            apply(change, on: host)
        case .resyncRequired:
            // The stream admitted it dropped deltas, so nothing on screen for
            // this machine can be trusted and a coalescing delay would only
            // widen the hole. A pending coalesce is redundant once this runs,
            // not wrong, so it is cancelled rather than left to fire again.
            coalescers[host]?.cancel()
            coalescers[host] = nil
            Task { await refresh(on: host) }
        // Gallery, machine identity and device lifecycle are the other
        // stores' concerns -- see `LibraryStore+Live` and `MachineStore`.
        case .gallery, .authority, .deviceStateChanged:
            break
        }
    }

    private func apply(_ change: MoldEvent.Queue, on host: MoldHost.ID) {
        switch change {
        // Edge-triggered on the wire (`types.rs:13228-13231`), so a value
        // rather than a toggle is the correct mirror of it here too.
        case .paused: queuePaused[host] = true
        case .resumed: queuePaused[host] = false
        // The V2 scheduler's own plan. This app draws no lane plan, so
        // decoding one only to discard it would be a wire surface with no
        // reader -- there is nothing to reconcile against it.
        case .planChanged: break
        }
    }

    /// Marks one machine's rows stale and, after `coalesceDelay`, re-reads it
    /// once. A new frame for the same machine replaces the pending wait
    /// rather than stacking a second read behind it.
    private func markDirty(_ host: MoldHost.ID) {
        coalescers[host]?.cancel()
        // No `[weak self]`: this store lives as long as the app, the same
        // reasoning `HostStore+Events.watch` already uses for its own task.
        coalescers[host] = Task {
            try? await Task.sleep(for: coalesceDelay)
            guard !Task.isCancelled else { return }
            await refresh(on: host)
        }
    }
}
