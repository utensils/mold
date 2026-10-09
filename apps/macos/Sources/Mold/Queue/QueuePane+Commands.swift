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

    /// Plain rows use group ids and expanded batch children use entry ids.
    /// The shared resolver preserves singleton mapping and refuses a batch
    /// header, which does not identify one job for these focused commands.
    private var selectedJob: QueueSelection.Job? {
        guard let selection else { return nil }
        for host in hosts.hosts {
            let entries = queue.entries(on: host.id)
            guard let entry = QueueGroup.selectedEntry(selection, in: queue.groups(on: host.id))
            else { continue }
            let canReorder = hosts.isUp(host) && !queue.isActing(entry, on: host.id) && hosts.capabilities[host.id]?.canReorderQueue == true
            // The same authority the row's own buttons and contextual menu
            // read, so this menu can never offer something they do not.
            let actions = queue.actions(for: entry, on: host.id)
            return QueueSelection.Job(
                target: .init(host: host.id, entry: entry.id),
                canPause: actions.pause,
                canResume: actions.resume,
                // Missing-model recovery stays on the explicit Download and
                // Retry row/menu control rather than dispatching a plain retry.
                canRetry: actions.retry && !missingModel(queue.hold(for: entry, on: host.id)),
                canMoveUp: canReorder && QueueRow.canMove(entry.id, .up, in: entries),
                canMoveDown: canReorder && QueueRow.canMove(entry.id, .down, in: entries),
                canCancel: actions.cancel,
                moveToDestinations: queue.canTransfer(entry, on: host.id) && transfers.transferring == nil ? transfers.transferDestinations(from: host.id) : [],
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

    /// A plain retry cannot repair a missing installation.
    private func missingModel(_ hold: QueueHold?) -> Bool {
        if case .missingModel = hold { return true }
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
