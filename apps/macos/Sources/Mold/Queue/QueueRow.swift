import MoldClient
import SwiftUI

struct QueueRow: View {
    let entry: QueueEntry

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 12) {
            Image(systemName: symbol)
                .foregroundStyle(entry.state == .failed || entry.state == .held
                                 ? AnyShapeStyle(.secondary) : AnyShapeStyle(.tertiary))
                .frame(width: 16)
            VStack(alignment: .leading, spacing: 2) {
                Text(entry.model ?? "Unknown model")
                Text(entry.waitDescription)
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
        }
        .padding(.vertical, 3)
        .help(entry.id)
    }

    private var symbol: String {
        switch entry.state {
        case .running: "circle.dotted"
        case .held: "pause.circle"
        case .failed: "exclamationmark.triangle"
        case .cancelled, .cancelling: "xmark.circle"
        case .complete: "checkmark.circle"
        case .paused: "pause.circle"
        case .accepted, .unknown: "clock"
        }
    }
}
