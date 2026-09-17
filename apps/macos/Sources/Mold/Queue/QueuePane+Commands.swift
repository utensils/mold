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
        guard job != nil || emptyQueue != nil else { return nil }
        return QueueSelection(job: job, emptyQueue: emptyQueue)
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
            let canPauseJob = hosts.capabilities[host.id]?.canPauseOneJob == true
            return QueueSelection.Job(
                canPause: (entry.state == .running || entry.state == .queued) && canPauseJob,
                canResume: entry.state == .paused && canPauseJob,
                canRetry: plainlyRetryable(queue.hold(for: entry, on: host.id)),
                canMoveUp: canReorder && QueueRow.canMove(entry.id, .up, in: entries),
                canMoveDown: canReorder && QueueRow.canMove(entry.id, .down, in: entries),
                canCancel: entry.state.isLive,
                pause: { act(.pause, on: entry, host: host) },
                resume: { act(.resume, on: entry, host: host) },
                retry: { act(.retry, on: entry, host: host) },
                moveUp: { move(entry.id, .up, host: host, entries: entries) },
                moveDown: { move(entry.id, .down, host: host, entries: entries) },
                cancel: { act(.cancel, on: entry, host: host) })
        }
        return nil
    }

    /// Only `.prose(_, retryable: true)` -- a missing-model hold's own
    /// button is Pull-then-Retry (`QueueHoldRow.swift`), which needs the
    /// download store this menu item does not carry.
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
