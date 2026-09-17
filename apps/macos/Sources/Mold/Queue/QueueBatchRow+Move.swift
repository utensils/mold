import MoldClient
import SwiftUI

// Split from `QueueBatchRow.swift` past the file-size advisory.

/// Where a whole batch asks to move to -- against its NEIGHBOURING GROUP, the
/// same "compute source/destination, then let `QueuePane.reorderCalls` do the
/// translation" shape a drag already takes, so there is exactly one place
/// that knows what `Array.move(fromOffsets:toOffset:)`'s own semantics mean
/// (design M6 S3).
extension QueueBatchRow {
    static func canMove(_ group: QueueGroup, _ direction: QueueRow.MoveDirection, in groups: [QueueGroup]) -> Bool {
        guard group.rows.contains(where: { $0.state.isReorderable }),
              let index = groups.firstIndex(where: { $0.id == group.id })
        else { return false }
        return direction == .up ? index > 0 : index < groups.count - 1
    }

    static func moveCall(
        _ group: QueueGroup, _ direction: QueueRow.MoveDirection, groups: [QueueGroup], entries: [QueueEntry]
    ) -> [(id: String, position: Int)] {
        guard let index = groups.firstIndex(where: { $0.id == group.id }) else { return [] }
        switch direction {
        case .up:
            guard index > 0 else { return [] }
            return QueuePane.reorderCalls(
                source: IndexSet(integer: index), destination: index - 1, groups: groups, entries: entries)
        case .down:
            guard index < groups.count - 1 else { return [] }
            return QueuePane.reorderCalls(
                source: IndexSet(integer: index), destination: index + 2, groups: groups, entries: entries)
        }
    }
}
