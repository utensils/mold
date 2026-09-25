import MoldClient
import SwiftUI

// What the Queue menu (`QueueCommands.swift`) offers for the pane's current
// `List` selection, published as a `FocusedValue` so every item is reachable
// from the keyboard and read by VoiceOver -- `LibraryCommands.swift`'s own
// reason. Split from the main file purely for size.
extension QueuePane {
    var queueSelection: QueueSelection? {
        let job = selectedJob
        let emptyQueues = emptyQueueActions
        let gate = queueGate
        guard job != nil || !emptyQueues.isEmpty || !gate.machines.isEmpty else { return nil }
        return QueueSelection(job: job, gate: gate, emptyQueues: emptyQueues)
    }

    /// The whole-queue gate, per machine that advertises it. The pane's own
    /// toolbar control reads this same value, so the two cannot disagree
    /// about the word on them.
    var queueGate: QueueGateOffer { gate.offer }

    /// The gate itself. A value, built where it is needed -- it holds no
    /// state of its own, only the rule.
    var gate: QueueGateControl { QueueGateControl(hosts: hosts, queue: queue) }

    /// `selection` is a GROUP id -- `List`'s own automatic `Identifiable`
    /// tagging over `rows(host:entries:)`'s `ForEach(groups)` -- resolved by
    /// `QueueGroup.selectedEntry`, which says why an entry-id lookup missed
    /// every row this app queues.
    private var selectedJob: QueueSelection.Job? {
        guard let selection else { return nil }
        for host in hosts.hosts {
            let entries = queue.entries(on: host.id)
            guard let entry = QueueGroup.selectedEntry(selection, in: queue.groups(on: host.id))
            else { continue }
            let canReorder = hosts.capabilities[host.id]?.canReorderQueue == true
            // The same authority the row's own buttons and contextual menu
            // read, so this menu can never offer something they do not.
            let actions = QueueRowActions.resolve(entry, on: hosts.capabilities[host.id])
            return QueueSelection.Job(
                target: .init(host: host.id, entry: entry.id),
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

    /// Every machine the toolbar's own chooser names. The menu must retain
    /// the same choice rather than silently acting on the first host.
    private var emptyQueueActions: [QueueSelection.EmptyQueue] {
        let targets = emptyQueueTargets
        let machines = targets.map { host in
            QueueSelection.EmptyQueue(id: host.id, name: host.name) {
                confirmEmptyQueue(on: host)
            }
        }
        guard targets.count > 1 else { return machines }
        let all = QueueSelection.EmptyQueue(id: nil, name: "All Machines") {
            confirmEmptyAllQueues()
        }
        return [all] + machines
    }
}
