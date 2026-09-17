import MoldClient
import SwiftUI

/// The Queue menu.
///
/// `LibraryCommands.swift`'s own reason: the menu bar is what Help ▸ Search
/// searches and VoiceOver reads, and an action that exists only in a row's
/// contextual menu is unreachable from the keyboard. Every job item here is
/// the same call the selected row's own controls make; Empty Queue… is the
/// same call the toolbar's own button makes (`QueuePane+Commands.swift`).
struct QueueCommands: Commands {
    @FocusedValue(\.queueSelection) private var selection

    var body: some Commands {
        CommandMenu("Queue") {
            if let job = selection?.job {
                if job.canPause { Button("Pause Job", action: job.pause) }
                if job.canResume { Button("Resume Job", action: job.resume) }
                if job.canRetry { Button("Try Again", action: job.retry) }
                if job.canMoveUp || job.canMoveDown {
                    Divider()
                    if job.canMoveUp { Button("Move Up", action: job.moveUp) }
                    if job.canMoveDown { Button("Move Down", action: job.moveDown) }
                }
                if !job.moveToDestinations.isEmpty {
                    Divider()
                    MoveToMenu(destinations: job.moveToDestinations, send: job.moveTo)
                }
                if job.canCancel {
                    Divider()
                    Button("Cancel Job", role: .destructive, action: job.cancel)
                }
            }
            if let emptyQueue = selection?.emptyQueue {
                if selection?.job != nil { Divider() }
                Button("Empty Queue…", action: emptyQueue)
            }
        }
    }
}

/// What the Queue pane's current selection can do, and how -- resolved once
/// per body pass, the same "items and closures together" shape
/// `ModelSelection` and `LibrarySelection` already take.
struct QueueSelection: Equatable {
    let job: Job?
    /// `nil` when no machine advertises it (design decision 5) --
    /// `QueuePane+Commands.emptyQueueAction` mirrors
    /// `QueuePane+Toolbar.emptyQueueTargets` exactly.
    let emptyQueue: (() -> Void)?

    struct Job: Equatable {
        let canPause, canResume, canRetry, canMoveUp, canMoveDown, canCancel: Bool
        /// Empty off a held row, or when nothing else on the fleet is up and
        /// generating -- `MoveToMenu`'s own "absent, not disabled" rule.
        let moveToDestinations: [TransferStore.TransferDestination]
        let pause, resume, retry, moveUp, moveDown, cancel: () -> Void
        let moveTo: (MoldHost.ID) -> Void

        static func == (lhs: Self, rhs: Self) -> Bool {
            lhs.canPause == rhs.canPause && lhs.canResume == rhs.canResume
                && lhs.canRetry == rhs.canRetry && lhs.canMoveUp == rhs.canMoveUp
                && lhs.canMoveDown == rhs.canMoveDown && lhs.canCancel == rhs.canCancel
                && lhs.moveToDestinations == rhs.moveToDestinations
        }
    }

    static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.job == rhs.job && (lhs.emptyQueue == nil) == (rhs.emptyQueue == nil)
    }

    /// The titles this selection actually offers, in the order the menu
    /// draws them -- pulled out of `body` so a test can pin exactly what
    /// shows without rendering a menu (design M6 S3).
    var offeredTitles: [String] {
        var titles: [String] = []
        if let job {
            if job.canPause { titles.append("Pause Job") }
            if job.canResume { titles.append("Resume Job") }
            if job.canRetry { titles.append("Try Again") }
            if job.canMoveUp { titles.append("Move Up") }
            if job.canMoveDown { titles.append("Move Down") }
            if !job.moveToDestinations.isEmpty { titles.append("Move to") }
            if job.canCancel { titles.append("Cancel Job") }
        }
        if emptyQueue != nil { titles.append("Empty Queue…") }
        return titles
    }
}

struct QueueSelectionKey: FocusedValueKey {
    typealias Value = QueueSelection
}

extension FocusedValues {
    var queueSelection: QueueSelection? {
        get { self[QueueSelectionKey.self] }
        set { self[QueueSelectionKey.self] = newValue }
    }
}
