import MoldClient
import SwiftUI

/// A batch's children, collapsed behind one row. A batch of ONE draws as a
/// plain `QueueRow` instead (`QueueGroup.isExpandable`) -- a disclosure
/// triangle over a single child is a control that reveals nothing.
struct QueueBatchRow: View {
    let group: QueueGroup
    /// What the machine will honour for ANY of this batch's children, and
    /// the same answer per child -- one authority for the group's buttons,
    /// its menu, and each child row (`QueueRowActions`).
    let actions: QueueRowActions
    let childActions: (QueueEntry) -> QueueRowActions
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
                QueueRow(entry: entry, actions: childActions(entry),
                         caption: entry.batchWaitDescription,
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

    /// The batch's own buttons a second way, in the Queue menu's words and
    /// order -- "every" because a group item reaches every child it applies
    /// to, which is what distinguishes it from a child row's own item.
    @ViewBuilder private var menu: some View {
        if actions.pause { Button("Pause Every Job") { groupAct(.pause) } }
        if actions.resume { Button("Resume Every Job") { groupAct(.resume) } }
        if canMoveUp { Button("Move Up", action: moveUp) }
        if canMoveDown { Button("Move Down", action: moveDown) }
        if actions.cancel {
            Divider()
            Button("Cancel Every Job", role: .destructive) { groupAct(.cancel) }
        }
    }

    /// The titles `menu` draws, in order, so a test pins them without
    /// rendering a menu -- `QueueHoldRow.menuTitles`'s shape.
    static func menuTitles(
        _ actions: QueueRowActions, canMoveUp: Bool, canMoveDown: Bool
    ) -> [String] {
        var titles: [String] = []
        if actions.pause { titles.append("Pause Every Job") }
        if actions.resume { titles.append("Resume Every Job") }
        if canMoveUp { titles.append("Move Up") }
        if canMoveDown { titles.append("Move Down") }
        if actions.cancel { titles.append("Cancel Every Job") }
        return titles
    }

    @ViewBuilder private var buttons: some View {
        HStack(spacing: 4) {
            if actions.pause {
                Button { groupAct(.pause) } label: { Image(systemName: "pause") }
                    .help("Pause every waiting job in this batch")
            }
            if actions.resume {
                Button { groupAct(.resume) } label: { Image(systemName: "play") }
                    .help("Resume every paused job in this batch")
            }
            if actions.cancel {
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
