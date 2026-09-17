import MoldClient
import SwiftUI

// What the Queue menu (`QueueCommands.swift`) offers for the pane's current
// `List` selection, published as a `FocusedValue` so every item is reachable
// from the keyboard and read by VoiceOver -- `LibraryCommands.swift`'s own
// reason. Split from the main file purely for size.
extension QueuePane {
    var queueSelection: QueueSelection? {
        let job = selectedJob
        let emptyQueue = emptyQueueAction
        let gate = queueGate
        guard job != nil || emptyQueue != nil || !gate.machines.isEmpty else { return nil }
        return QueueSelection(job: job, gate: gate, emptyQueue: emptyQueue)
    }

    /// The whole-queue gate, per machine that advertises it. The pane's own
    /// toolbar control reads this same value, so the two cannot disagree
    /// about the word on them.
    var queueGate: QueueGateOffer {
        QueueGateOffer(
            machines: QueueStore.gateTargets(hosts.hosts, capabilities: hosts.capabilities)
                .map { QueueGateOffer.Machine(id: $0.id, name: $0.name,
                                              isPaused: queue.isQueuePaused(on: $0.id)) },
            toggle: { host in Task { await queue.toggleQueuePaused(on: host) } })
    }

    /// `selection` is one id across every host's flat rows AND every batch
    /// child -- `List`'s own automatic `Identifiable`-based tagging, since
    /// `QueueGroup.id` and `QueueEntry.id` share one string space. A batch's
    /// own disclosure row resolves to no entry here, which is correct: none
    /// of this menu's items are batch-wide (`QueueBatchRow`'s own buttons
    /// already cover that).
    private var selectedJob: QueueSelection.Job? {
        guard let selection else { return nil }
        for host in hosts.hosts {
            let entries = queue.entries(on: host.id)
            guard let entry = entries.first(where: { $0.id == selection }) else { continue }
            let canReorder = hosts.capabilities[host.id]?.canReorderQueue == true
            // The same authority the row's own buttons and contextual menu
            // read, so this menu can never offer something they do not.
            let actions = QueueRowActions.resolve(entry, on: hosts.capabilities[host.id])
            return QueueSelection.Job(
                canPause: actions.pause,
                canResume: actions.resume,
                // Narrower than `actions.retry` on purpose: this menu knows
                // the row's BATCH CHILD, which is the only place `error_code`
                // and the host's own `retryable` live (`routes.rs:2951-2956`).
                // A missing-model hold's own button is Pull-then-Retry
                // (`QueueHoldRow.swift`), which needs the download store this
                // menu item does not carry.
                canRetry: plainlyRetryable(queue.hold(for: entry, on: host.id)),
                canMoveUp: canReorder && QueueRow.canMove(entry.id, .up, in: entries),
                canMoveDown: canReorder && QueueRow.canMove(entry.id, .down, in: entries),
                canCancel: actions.cancel,
                moveToDestinations: entry.state == .held ? transfers.transferDestinations(from: host.id) : [],
                pause: { act(.pause, on: entry, host: host) },
                resume: { act(.resume, on: entry, host: host) },
                retry: { act(.retry, on: entry, host: host) },
                moveUp: { move(entry.id, .up, host: host, entries: entries) },
                moveDown: { move(entry.id, .down, host: host, entries: entries) },
                cancel: { act(.cancel, on: entry, host: host) },
                moveTo: { moveTo(entry, from: host, to: $0) })
        }
        return nil
    }

    /// Only `.prose(_, retryable: true)`.
    private func plainlyRetryable(_ hold: QueueHold?) -> Bool {
        if case let .prose(_, retryable) = hold { return retryable }
        return false
    }

    /// The first machine that offers it -- the exact gate the toolbar's own
    /// button reads (`emptyQueueTargets`), so the two can never disagree
    /// about whether it is offered.
    private var emptyQueueAction: (() -> Void)? {
        guard let host = emptyQueueTargets.first else { return nil }
        return { confirmEmptyQueue(on: host) }
    }
}
