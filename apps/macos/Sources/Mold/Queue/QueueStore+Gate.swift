import Foundation
import MoldClient

// The whole-queue gate. It names the QUEUE, not a row: pausing stops the NEXT
// job starting and leaves the one already on the GPU alone.
@MainActor
extension QueueStore {
    /// The verb every failure about this is keyed on.
    static let gateVerb = "pause its queue"

    /// Whether this machine is dispatching, as of the last thing heard.
    ///
    /// READ before it is decided. `queuePaused` is written by the frames and
    /// by the verb below, but on a machine this app has only just met neither
    /// has happened yet -- and the status poll already carries the answer
    /// (`/api/status.queue_paused`). Desktop's own bug was deciding from a
    /// snapshot nobody had fetched: the first press paused for real, the
    /// value stayed false, and the second press paused again.
    func isQueuePaused(on host: MoldHost.ID) -> Bool {
        if let known = queuePaused[host] { return known }
        if case let .up(status) = hosts.reachability[host] { return status.queuePaused ?? false }
        return false
    }

    /// Whether this machine offers the control at all. Absence is a
    /// definitive no, and the control is then ABSENT rather than inert.
    func canPauseQueue(on host: MoldHost.ID) -> Bool {
        hosts.capabilities[host]?.canPauseQueue == true
    }

    /// Pauses or resumes one machine's whole queue.
    ///
    /// The VERB is the writer and the `queue_paused` / `queue_resumed` frame
    /// is the confirmation. Nothing double-applies, because what is written
    /// is the SERVER's answer rather than the intent that was sent, and the
    /// frame carries a value rather than a toggle -- so arriving twice, or
    /// arriving before this returns, lands on the same state.
    func setQueuePaused(_ paused: Bool, on host: MoldHost.ID) async {
        guard !isSeeded, canPauseQueue(on: host), let backend = hosts.backend(for: host) else {
            return
        }
        do {
            let state = paused ? try await backend.pauseQueue() : try await backend.resumeQueue()
            queuePaused[host] = state.paused
            hosts.succeeded(on: host, doing: Self.gateVerb)
            // The gate moved, so what is waiting and what is running is about
            // to read differently.
            await refresh(on: host)
        } catch {
            hosts.report(error, on: host, doing: Self.gateVerb)
        }
    }

    /// The other way round from where it is now, on this machine.
    func toggleQueuePaused(on host: MoldHost.ID) async {
        await setQueuePaused(!isQueuePaused(on: host), on: host)
    }

    /// The machines that advertise the control, in the order they are listed.
    /// Pure enough to pin without a rendered menu -- `emptyQueueTargets`'
    /// own shape.
    static func gateTargets(
        _ hosts: [MoldHost], capabilities: [MoldHost.ID: Capabilities]
    ) -> [MoldHost] {
        hosts.filter { capabilities[$0.id]?.canPauseQueue == true }
    }
}
