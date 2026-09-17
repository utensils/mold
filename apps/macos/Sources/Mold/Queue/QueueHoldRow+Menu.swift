import MoldClient
import SwiftUI

// What a held row offers, and the one door that performs it. Split from the
// row's own shape for size; `internal` rather than `private` because
// `private` does not cross a file boundary even within one type.
extension QueueHoldRow {
    /// Never includes "Move to ▾": that is drawn unconditionally, on every
    /// hold, as its own element (`// S4: MoveToMenu`) -- these are only the
    /// retry-shaped choices, which differ by hold.
    enum Action: Hashable {
        case pullThenRetry(model: String)
        case tryAgain

        /// The words, once: the inline button and the menu item are the same
        /// offer and used to spell it in two places.
        var title: String {
            switch self {
            case let .pullThenRetry(model): "Pull \(model), then Retry"
            case .tryAgain: "Try Again"
            }
        }
    }

    /// Everything the row offers, including the two things the retry-shaped
    /// buttons are not: a machine to send the job to, and giving up on it.
    enum Item: Hashable {
        case act(Action)
        case moveTo(MoldHost.ID)
        case cancel
    }

    /// Pure: what this hold offers, pulled out of the view so a test can pin
    /// it without rendering anything (design M6 S3, decision 11).
    ///
    /// `.missingModel` never offers `.tryAgain` -- the row's OWN sentence has
    /// already been produced by a plain retry once, which is how it got
    /// here; the resolvable action is Pull-then-Retry, not the same retry
    /// again. `.prose(_, retryable: false)` offers nothing: the machine has
    /// said trying again will not help, and a button that reproduces the
    /// hold is worse than no button.
    static func actions(for hold: QueueHold) -> [Action] {
        switch hold {
        case let .missingModel(model, _): [.pullThenRetry(model: model)]
        case let .prose(_, retryable): retryable ? [.tryAgain] : []
        }
    }

    /// Everything the row offers, in order, so a test can pin that Cancel Job
    /// is always there and always last without rendering a menu. Move to is a
    /// submenu, and `RowAction.rendered` drops it where there is nowhere to
    /// send the job -- the gate `MoveToMenu` keeps for the inline control.
    static func offered(
        for hold: QueueHold, destinations: [TransferStore.TransferDestination]
    ) -> [RowAction<Item>] {
        var items = actions(for: hold).map { RowAction(kind: Item.act($0), title: $0.title) }
        items.append(RowAction(title: "Move to", children: destinations.map {
            RowAction(kind: Item.moveTo($0.id), title: $0.caption)
        }))
        items.append(RowAction(kind: .cancel, title: "Cancel Job", isDestructive: true))
        return RowAction.ordered(items)
    }

    func perform(_ item: Item) {
        switch item {
        case let .act(.pullThenRetry(model)): pullThenRetry(model)
        case .act(.tryAgain): tryAgain()
        case let .moveTo(host): moveTo(host)
        case .cancel: cancel()
        }
    }
}
