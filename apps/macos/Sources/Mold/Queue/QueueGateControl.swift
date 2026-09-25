import Foundation
import MoldClient

/// The whole-queue gate: stop a machine dispatching new work, and start it
/// again.
///
/// Its own small type rather than more of `QueueStore`, which is at the
/// type-size budget -- and it genuinely is a rule rather than more state: the
/// one piece of state, `queuePaused`, already lives on the store because the
/// event frames write it there.
///
/// It names the QUEUE, not a row. Pausing stops the NEXT job starting; the
/// one already on the GPU keeps going.
@MainActor
struct QueueGateControl {
    let hosts: HostStore
    let queue: QueueStore

    /// The verb every failure about this is keyed on, so a retry replaces its
    /// predecessor in the banner rather than piling up.
    static let verb = "pause its queue"

    /// Whether this machine is dispatching, as of the last thing heard.
    ///
    /// READ before it is decided. `queuePaused` is written by the frames and
    /// by the verb below, but on a machine this app has only just met neither
    /// has happened -- and the status poll already carries the answer
    /// (`/api/status.queue_paused`). Desktop's own bug was deciding from a
    /// snapshot nobody had fetched: the first press paused for real, the
    /// value stayed false, and the second press paused again.
    func isPaused(on host: MoldHost.ID) -> Bool {
        if let known = queue.queuePaused[host] { return known }
        if case let .up(status) = hosts.reachability[host] { return status.queuePaused ?? false }
        return false
    }

    /// Whether this machine offers the control at all. Absence is a
    /// definitive no, and the control is then ABSENT rather than inert.
    func canPause(on host: MoldHost.ID) -> Bool {
        hosts.capabilities[host]?.canPauseQueue == true
    }

    /// Pauses or resumes one machine's whole queue.
    ///
    /// The VERB is the writer and the `queue_paused` / `queue_resumed` frame
    /// is the confirmation. Nothing double-applies, because what is written
    /// is the SERVER's answer rather than the intent that was sent, and the
    /// frame carries a value rather than a toggle -- so arriving twice, or
    /// arriving before this returns, lands on the same state.
    func set(_ paused: Bool, on host: MoldHost.ID) async {
        guard !queue.isSeeded, canPause(on: host), let backend = hosts.backend(for: host) else {
            return
        }
        do {
            let state = paused ? try await backend.pauseQueue() : try await backend.resumeQueue()
            queue.queuePaused[host] = state.paused
            hosts.succeeded(on: host, doing: Self.verb)
            // The gate moved, so what is waiting and what is running is about
            // to read differently.
            await queue.refresh(on: host)
        } catch {
            hosts.report(error, on: host, doing: Self.verb)
        }
    }

    /// The other way round from where it is now.
    func toggle(on host: MoldHost.ID) async {
        await set(!isPaused(on: host), on: host)
    }

    /// Every machine that advertises the control and is not already there,
    /// concurrently -- each reports its own failure, so one unreachable
    /// machine never hides what the others did.
    func setAll(_ paused: Bool) async {
        let targets = Self.targets(hosts.hosts, capabilities: hosts.capabilities)
            .filter { isPaused(on: $0.id) != paused }
        await withTaskGroup(of: Void.self) { group in
            for host in targets {
                group.addTask { await set(paused, on: host.id) }
            }
        }
    }

    func perform(_ target: QueueGateOffer.Target) async {
        switch target {
        case let .machine(host): await toggle(on: host)
        case let .all(paused): await setAll(paused)
        }
    }

    /// The machines that advertise the control, in the order they are listed.
    /// Pure, so it pins without a rendered menu -- `emptyQueueTargets`' own
    /// shape.
    static func targets(
        _ hosts: [MoldHost], capabilities: [MoldHost.ID: Capabilities]
    ) -> [MoldHost] {
        hosts.filter { capabilities[$0.id]?.canPauseQueue == true }
    }

    /// What the Queue menu and the pane's own control both draw.
    var offer: QueueGateOffer {
        QueueGateOffer(
            machines: Self.targets(hosts.hosts, capabilities: hosts.capabilities)
                .map { QueueGateOffer.Machine(id: $0.id, name: $0.name,
                                              isPaused: isPaused(on: $0.id)) },
            toggle: { target in Task { await perform(target) } })
    }
}
