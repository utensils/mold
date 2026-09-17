import MoldClient
import SwiftUI

struct QueueRow: View {
    enum Action { case cancel, pause, resume, retry }

    let entry: QueueEntry
    let act: (Action) -> Void

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 12) {
            Image(systemName: symbol)
                .foregroundStyle(.tertiary)
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
            buttons
        }
        .padding(.vertical, 3)
        .contextMenu { menu }
        .help(entry.id)
    }

    @ViewBuilder private var buttons: some View {
        HStack(spacing: 4) {
            // Retry is offered only where the host said it would help. A held
            // job whose cause is unfixed will just hold again.
            if entry.state == .held, entry.retryable != false {
                Button { act(.retry) } label: { Image(systemName: "arrow.clockwise") }
                    .help("Try this job again")
            }
            if entry.state == .running || entry.state == .queued {
                Button { act(.pause) } label: { Image(systemName: "pause") }
                    .help("Pause this job")
            }
            if entry.state == .paused {
                Button { act(.resume) } label: { Image(systemName: "play") }
                    .help("Resume this job")
            }
            if entry.state.isLive {
                Button { act(.cancel) } label: { Image(systemName: "xmark") }
                    .help("Cancel this job")
            }
        }
        .buttonStyle(.borderless)
        .labelStyle(.iconOnly)
    }

    @ViewBuilder private var menu: some View {
        if entry.state == .held { Button("Try Again") { act(.retry) } }
        if entry.state.isLive {
            Button("Cancel Job", role: .destructive) { act(.cancel) }
        }
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
