import MoldClient
import SwiftUI

/// A batch's children, collapsed behind one row. A batch of ONE draws as a
/// plain `QueueRow` instead (`QueueGroup.isExpandable`) -- a disclosure
/// triangle over a single child is a control that reveals nothing.
struct QueueBatchRow: View {
    let group: QueueGroup
    /// What one CHILD's own row asked for.
    let rowAct: (QueueRow.Action, QueueEntry) -> Void
    /// What the GROUP's own row asked for -- every live child at once,
    /// serialized, then one re-read (`QueueStore.act(_:onLiveChildrenOf:)`).
    let groupAct: (QueueRow.Action) -> Void
    /// Drag alone is unreachable from the keyboard -- the batch's own
    /// keyboard twin of `QueueRow`'s Move Up/Down, gated the same way: only
    /// on a machine that advertises reorder, and only at an edge that has
    /// somewhere to go.
    var canMoveUp = false
    var canMoveDown = false
    var moveUp: () -> Void = {}
    var moveDown: () -> Void = {}

    var body: some View {
        DisclosureGroup {
            ForEach(group.rows) { entry in
                QueueRow(entry: entry, caption: entry.batchWaitDescription,
                         act: { rowAct($0, entry) })
                    .padding(.leading, 20)
            }
        } label: {
            HStack(alignment: .firstTextBaseline, spacing: 12) {
                Image(systemName: "square.stack")
                    .foregroundStyle(.tertiary)
                    .frame(width: 16)
                VStack(alignment: .leading, spacing: 2) {
                    Text("\(group.rows.first?.model ?? "Unknown model") · \(group.rows.count) renders")
                    Text(Self.caption(group.rows))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer(minLength: 12)
                buttons
            }
        }
        .padding(.vertical, 3)
        .contextMenu { menu }
    }

    @ViewBuilder private var menu: some View {
        if canMoveUp { Button("Move Up", action: moveUp) }
        if canMoveDown { Button("Move Down", action: moveDown) }
    }

    @ViewBuilder private var buttons: some View {
        HStack(spacing: 4) {
            if group.rows.contains(where: { $0.state == .running || $0.state == .queued }) {
                Button { groupAct(.pause) } label: { Image(systemName: "pause") }
                    .help("Pause every live job in this batch")
            }
            if group.rows.contains(where: { $0.state == .paused }) {
                Button { groupAct(.resume) } label: { Image(systemName: "play") }
                    .help("Resume every paused job in this batch")
            }
            if group.rows.contains(where: { $0.state.isLive }) {
                Button { groupAct(.cancel) } label: { Image(systemName: "xmark") }
                    .help("Cancel every live job in this batch")
            }
        }
        .buttonStyle(.borderless)
        .labelStyle(.iconOnly)
    }

    /// "2 waiting, 1 rendering" -- the children's own states, never a
    /// repeated model name.
    static func caption(_ rows: [QueueEntry]) -> String {
        func count(_ state: QueueState) -> Int { rows.filter { $0.state == state }.count }
        let clauses = [
            (count(.queued), "waiting"), (count(.running), "rendering"),
            (count(.held), "held"), (count(.paused), "paused"),
        ].compactMap { n, word in n > 0 ? "\(n) \(word)" : nil }
        return clauses.isEmpty ? "Done" : clauses.joined(separator: ", ")
    }
}

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

/// Inside a batch, the machine's overall queue POSITION answers a question
/// nobody is asking -- this reads `batchIndex` instead (design M6 S3: "Its
/// children are `QueueRow`s showing `batchIndex` rather than `position`").
private extension QueueEntry {
    var batchWaitDescription: String {
        guard state == .queued || state == .unknown else { return waitDescription }
        switch batchIndex {
        case .some(0): return "Next in this batch"
        case let .some(n): return "#\(n + 1) in this batch"
        case nil: return waitDescription
        }
    }
}
