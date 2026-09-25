import MoldClient
import SwiftUI

/// What the Queue pane's current selection can do, and how -- resolved once
/// per body pass, the same "items and closures together" shape
/// `ModelSelection` and `LibrarySelection` already take.
struct QueueSelection: Equatable {
    let job: Job?
    var gate = QueueGateOffer(machines: [], toggle: { _ in })
    let emptyQueues: [EmptyQueue]

    /// `id == nil` is Empty Queue on All Machines.
    struct EmptyQueue: Equatable {
        let id: MoldHost.ID?
        let name: String
        let run: () -> Void

        static func == (lhs: Self, rhs: Self) -> Bool {
            lhs.id == rhs.id && lhs.name == rhs.name
        }
    }

    struct Job: Equatable {
        /// Selected-row identity keeps SwiftUI from retaining menu closures
        /// for a previous row with the same capabilities.
        struct Target: Equatable {
            let host: MoldHost.ID
            let entry: String
        }

        let target: Target
        let canPause, canResume, canRetry, canMoveUp, canMoveDown, canCancel: Bool
        let moveToDestinations: [TransferStore.TransferDestination]
        let pause, resume, retry, moveUp, moveDown, cancel: () -> Void
        let moveTo: (MoldHost.ID) -> Void

        static func == (lhs: Self, rhs: Self) -> Bool {
            lhs.target == rhs.target
                && lhs.canPause == rhs.canPause && lhs.canResume == rhs.canResume
                && lhs.canRetry == rhs.canRetry && lhs.canMoveUp == rhs.canMoveUp
                && lhs.canMoveDown == rhs.canMoveDown && lhs.canCancel == rhs.canCancel
                && lhs.moveToDestinations == rhs.moveToDestinations
        }
    }

    static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.job == rhs.job && lhs.gate == rhs.gate && lhs.emptyQueues == rhs.emptyQueues
    }

    enum Item: Hashable {
        case act(QueueRowActions.Kind)
        case moveTo(MoldHost.ID)
        case pauseQueue(QueueGateOffer.Target)
        /// `nil` is every machine.
        case emptyQueue(MoldHost.ID?)
    }

    /// The list the menu draws, with separators trimmed by `RowAction`.
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
        items += gate.items().map { $0.mapKind(Item.pauseQueue) }
        let machines = emptyQueues.filter { $0.id != nil }
        if machines.count == 1, let target = machines.first {
            items.append(RowAction(kind: .emptyQueue(target.id), title: "Empty Queue…"))
        } else {
            if emptyQueues.contains(where: { $0.id == nil }) {
                items.append(RowAction(kind: .emptyQueue(nil), title: QueueEmptyConfirm.allMachinesItem))
            }
            items += machines.map { target in
                RowAction(kind: .emptyQueue(target.id),
                          title: "Empty Queue on \(target.name)…")
            }
        }
        return items
    }

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
        case let .pauseQueue(host): gate.toggle(host)
        case let .emptyQueue(host): emptyQueues.first { $0.id == host }?.run()
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
