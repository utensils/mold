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
            if let selection {
                RowActionMenu(actions: selection.offered, perform: selection.perform) { item in
                    // ⌘⌫, the Library's own Move to Trash chord: the selected
                    // row leaves the queue from the keyboard, and Help ▸
                    // Search finds it. The menu bar is the only surface that
                    // carries chords -- a contextual menu shows none.
                    item == .act(.cancel) ? KeyboardShortcut(.delete, modifiers: .command) : nil
                }
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

    /// Everything the Queue menu can offer: a row's own actions, a machine to
    /// send the job to, and emptying the whole queue.
    enum Item: Hashable {
        case act(QueueRowActions.Kind)
        case moveTo(MoldHost.ID)
        case emptyQueue
    }

    /// THE list the menu draws, grouped the way the menu bar groups it. It
    /// used to be `body` and a hand-maintained `offeredTitles` beside it,
    /// restating the same seven titles and the same seven gates -- and no
    /// test could read `body`, so they agreed by hand.
    ///
    /// The separators are declared, so `RowAction.rendered` keeps this order
    /// and only trims what the gating left out: an empty Move to ▸, and any
    /// divider with nothing on one side of it.
    var offered: [RowAction<Item>] {
        var items: [RowAction<Item>] = []
        if let job {
            let row = QueueRowActions(pause: job.canPause, resume: job.canResume,
                                      retry: job.canRetry, cancel: job.canCancel)
            items += row.offered().filter { $0.kind != .cancel }.map { $0.mapKind(Item.act) }
            items.append(.separator)
            if job.canMoveUp { items.append(QueueRowActions.item(.moveUp).mapKind(Item.act)) }
            if job.canMoveDown { items.append(QueueRowActions.item(.moveDown).mapKind(Item.act)) }
            items.append(.separator)
            items.append(RowAction(title: "Move to", children: job.moveToDestinations.map {
                RowAction(kind: Item.moveTo($0.id), title: $0.caption)
            }))
            items.append(.separator)
            if job.canCancel { items.append(QueueRowActions.item(.cancel).mapKind(Item.act)) }
        }
        items.append(.separator)
        if emptyQueue != nil { items.append(RowAction(kind: .emptyQueue, title: "Empty Queue…")) }
        return items
    }

    /// The titles this selection actually offers, in the order the menu draws
    /// them -- so a test can pin exactly what shows without rendering a menu
    /// (design M6 S3). A submenu is named, not enumerated.
    var offeredTitles: [String] {
        RowAction.rendered(offered).filter { !$0.isSeparator }.map(\.title)
    }

    func perform(_ item: Item) {
        switch item {
        case .act(.pause): job?.pause()
        case .act(.resume): job?.resume()
        case .act(.retry): job?.retry()
        case .act(.moveUp): job?.moveUp()
        case .act(.moveDown): job?.moveDown()
        case .act(.cancel): job?.cancel()
        case let .moveTo(host): job?.moveTo(host)
        case .emptyQueue: emptyQueue?()
        }
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
