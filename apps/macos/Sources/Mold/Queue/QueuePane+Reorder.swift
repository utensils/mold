import MoldClient
import SwiftUI

// The pane's own translation from a SwiftUI drag gesture to what the
// machine's reorder route actually wants. Neither the screen index NOR the
// row's own `position` is the PATCH's index (design M6 fact 2) -- pulled out
// of the view so a test can pin it without a rendered `List`.
extension QueuePane {
    /// `source`/`destination` are `.onMove`'s own arguments over `groups` as
    /// drawn on screen. Only the moved group's REORDERABLE rows travel; a
    /// batch mixing live and settled children moves only the live ones,
    /// same as a group's own Cancel/Pause button.
    static func reorderCalls(
        source: IndexSet, destination: Int, groups: [QueueGroup], entries: [QueueEntry]
    ) -> [(id: String, position: Int)] {
        guard let sourceIndex = source.first, groups.indices.contains(sourceIndex) else { return [] }
        let movedID = groups[sourceIndex].id
        let ids = groups[sourceIndex].rows.filter { $0.state.isReorderable }.map(\.id)
        guard !ids.isEmpty else { return [] }

        // `Array.move(fromOffsets:toOffset:)` is `.onMove`'s own semantics --
        // reused here rather than re-derived, so this asks "what group ended
        // up immediately before the moved one" instead of hand-rolling the
        // same index arithmetic a second time.
        var afterMove = groups
        afterMove.move(fromOffsets: source, toOffset: destination)
        guard let newIndex = afterMove.firstIndex(where: { $0.id == movedID }) else { return [] }
        let neighbourID = afterMove[..<newIndex]
            .reversed()
            .compactMap { $0.rows.last { $0.state.isReorderable } }
            .first?.id

        return QueueOrder.moves(ids, after: neighbourID, in: entries)
    }
}

extension QueuePane {
    /// The batch keyboard twin's own action: `QueueBatchRow.moveCall`
    /// already did the translation, this only sends it and only when there
    /// was anywhere to go.
    func moveBatch(
        _ group: QueueGroup, _ direction: QueueRow.MoveDirection, host: MoldHost,
        groups: [QueueGroup], entries: [QueueEntry]
    ) {
        let calls = QueueBatchRow.moveCall(group, direction, groups: groups, entries: entries)
        guard !calls.isEmpty else { return }
        Task { await queue.reorder(calls, on: host.id) }
    }
}
