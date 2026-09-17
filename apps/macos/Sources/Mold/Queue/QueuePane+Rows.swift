import MoldClient
import SwiftUI

// One machine's section of the list: its groups, and which view each
// group draws as. Split from `QueuePane.swift` past the file-size
// advisory; the pane itself now holds the scene and its actions.
extension QueuePane {
    /// One host's rows. `.onMove` is attached only when the machine
    /// advertises reorder -- SwiftUI draws no drag affordance without it,
    /// so a duplicated `ForEach` is what keeps the modifier truly absent
    /// rather than present-but-inert.
    ///
    /// Not `private`: the pane's own `body` draws this, and `private` does
    /// not cross a file boundary even within one type.
    @ViewBuilder
    func rows(host: MoldHost, entries: [QueueEntry]) -> some View {
        let groups = queue.groups(on: host.id)
        let canReorder = hosts.capabilities[host.id]?.canReorderQueue == true
        if canReorder {
            ForEach(groups) { row($0, host: host, entries: entries, groups: groups, canReorder: true) }
                .onMove { source, destination in
                    let calls = QueuePane.reorderCalls(
                        source: source, destination: destination, groups: groups, entries: entries)
                    Task { await queue.reorder(calls, on: host.id) }
                }
        } else {
            ForEach(groups) { row($0, host: host, entries: entries, groups: groups, canReorder: false) }
        }
    }

    @ViewBuilder
    func row(
        _ group: QueueGroup, host: MoldHost, entries: [QueueEntry], groups: [QueueGroup], canReorder: Bool
    ) -> some View {
        if group.isExpandable {
            QueueBatchRow(
                group: group,
                actions: QueueRowActions.group(group.rows, on: hosts.capabilities[host.id]),
                childActions: { QueueRowActions.resolve($0, on: hosts.capabilities[host.id]) },
                rowAct: { action, entry in act(action, on: entry, host: host) },
                groupAct: { action in Task { await queue.act(action, onLiveChildrenOf: group, host: host.id) } },
                canMoveUp: canReorder && QueueBatchRow.canMove(group, .up, in: groups),
                canMoveDown: canReorder && QueueBatchRow.canMove(group, .down, in: groups),
                moveUp: { moveBatch(group, .up, host: host, groups: groups, entries: entries) },
                moveDown: { moveBatch(group, .down, host: host, groups: groups, entries: entries) })
        } else {
            let entry = group.rows[0]
            if entry.state == .held, let hold = queue.hold(for: entry, on: host.id) {
                QueueHoldRow(
                    entry: entry, hold: hold,
                    pullThenRetry: { model in pullThenRetry(model, entry: entry, host: host) },
                    tryAgain: { act(.retry, on: entry, host: host) },
                    moveToDestinations: transfers.transferDestinations(from: host.id),
                    moveTo: { moveTo(entry, from: host, to: $0) },
                    cancel: { act(.cancel, on: entry, host: host) })
            } else {
                let reorderable = canReorder && entry.state.isReorderable
                QueueRow(
                    entry: entry,
                    actions: QueueRowActions.resolve(entry, on: hosts.capabilities[host.id]),
                    isReorderable: reorderable,
                    canMoveUp: reorderable && QueueRow.canMove(entry.id, .up, in: entries),
                    canMoveDown: reorderable && QueueRow.canMove(entry.id, .down, in: entries),
                    moveUp: { move(entry.id, .up, host: host, entries: entries) },
                    moveDown: { move(entry.id, .down, host: host, entries: entries) },
                    act: { act($0, on: entry, host: host) })
            }
        }
    }
}
