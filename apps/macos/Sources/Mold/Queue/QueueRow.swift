import MoldClient
import SwiftUI

struct QueueRow: View {
    enum Action { case cancel, pause, resume, retry }
    enum MoveDirection { case up, down }

    let entry: QueueEntry
    /// What this machine will actually honour for this row, resolved once by
    /// `QueueRowActions` and read by both the glyph buttons and the
    /// contextual menu -- so the two can never offer different things.
    let actions: QueueRowActions
    /// Overrides the second line. A batch child names its place IN THE
    /// BATCH rather than the machine's overall queue position -- see
    /// `QueueBatchRow`.
    var caption: String?
    /// Drawn only on a `queued` row on a machine that advertises reorder --
    /// SwiftUI's own "absent, not disabled" rule the whole app follows.
    var isReorderable = false
    var canMoveUp = false
    var canMoveDown = false
    var moveUp: () -> Void = {}
    var moveDown: () -> Void = {}
    let act: (Action) -> Void

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 12) {
            if isReorderable {
                Image(systemName: "line.3.horizontal")
                    .foregroundStyle(.tertiary)
                    .accessibilityLabel("Drag to reorder")
            }
            Image(systemName: symbol)
                .foregroundStyle(.tertiary)
                .frame(width: 16)
            VStack(alignment: .leading, spacing: 2) {
                Text(entry.model ?? "Unknown model")
                Text(caption ?? entry.waitDescription)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
            }
            Spacer(minLength: 12)
            if let started = entry.startedAt {
                Text(started, format: .relative(presentation: .numeric))
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            }
            buttons
        }
        .padding(.vertical, 3)
        // THE list its buttons are built from, rendered rather than mirrored
        // -- and no menu at all on a settled row, which used to open an empty
        // one. Destructive last behind a divider is `RowActionMenu`'s rule,
        // not a `@ViewBuilder` this file could reorder by accident.
        .rowActionMenu(
            actions.offered(
                canMoveUp: isReorderable && canMoveUp,
                canMoveDown: isReorderable && canMoveDown),
            perform: perform)
        .help(entry.id)
    }

    private func perform(_ kind: QueueRowActions.Kind) {
        switch kind {
        case .pause: act(.pause)
        case .resume: act(.resume)
        case .retry: act(.retry)
        case .moveUp: moveUp()
        case .moveDown: moveDown()
        case .cancel: act(.cancel)
        }
    }

    @ViewBuilder private var buttons: some View {
        HStack(spacing: 4) {
            if actions.retry {
                Button { act(.retry) } label: { Image(systemName: "arrow.clockwise") }
                    .help("Try this job again")
            }
            if actions.pause {
                Button { act(.pause) } label: { Image(systemName: "pause") }
                    .help("Pause this job")
            }
            if actions.resume {
                Button { act(.resume) } label: { Image(systemName: "play") }
                    .help("Resume this job")
            }
            if actions.cancel {
                Button { act(.cancel) } label: { Image(systemName: "xmark") }
                    .help("Cancel this job")
            }
        }
        .buttonStyle(.borderless)
        .labelStyle(.iconOnly)
    }

    private var symbol: String {
        switch entry.state {
        case .running: "circle.dotted"
        case .held, .paused: "pause.circle"
        case .failed: "exclamationmark.triangle"
        case .cancelled, .cancelling: "xmark.circle"
        case .complete: "checkmark.circle"
        case .queued, .unknown: "clock"
        }
    }
}

/// Where a row asks to move to, against the machine's own REORDERABLE
/// candidates -- never the row's neighbour on screen, which may be a batch
/// or a held row that isn't reorderable at all (design M6 fact 2).
extension QueueRow {
    static func canMove(_ id: String, _ direction: MoveDirection, in entries: [QueueEntry]) -> Bool {
        let candidates = entries.filter(\.state.isReorderable)
        guard let index = candidates.firstIndex(where: { $0.id == id }) else { return false }
        return direction == .up ? index > 0 : index < candidates.count - 1
    }

    static func moveCall(
        _ id: String, _ direction: MoveDirection, in entries: [QueueEntry]
    ) -> (id: String, position: Int)? {
        let candidates = entries.filter(\.state.isReorderable)
        guard let index = candidates.firstIndex(where: { $0.id == id }) else { return nil }
        switch direction {
        case .up:
            guard index > 0 else { return nil }
            let neighbour = index > 1 ? candidates[index - 2].id : nil
            return QueueOrder.move(id, after: neighbour, in: entries)
        case .down:
            guard index < candidates.count - 1 else { return nil }
            return QueueOrder.move(id, after: candidates[index + 1].id, in: entries)
        }
    }
}
