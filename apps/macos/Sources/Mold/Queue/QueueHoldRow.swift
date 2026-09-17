import MoldClient
import SwiftUI

/// The only row that draws its cause as a paragraph and offers named
/// buttons instead of glyphs -- because it is the only row asking for a
/// decision (design M6 "Where everything goes").
struct QueueHoldRow: View {
    /// Never includes "Move to ▾": that is drawn unconditionally, on every
    /// hold, as its own element (`// S4: MoveToMenu`) -- these are only the
    /// retry-shaped choices, which differ by hold.
    enum Action: Hashable {
        case pullThenRetry(model: String)
        case tryAgain
    }

    let entry: QueueEntry
    let hold: QueueHold
    let pullThenRetry: (String) -> Void
    let tryAgain: () -> Void

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 12) {
            Image(systemName: "exclamationmark.triangle")
                .foregroundStyle(.orange)
                .frame(width: 16)
            VStack(alignment: .leading, spacing: 6) {
                Text(entry.model ?? "Unknown model")
                Text(sentence)
                    .font(.callout)
                    .foregroundStyle(.secondary)
                HStack(spacing: 8) {
                    ForEach(Self.actions(for: hold), id: \.self) { action in
                        button(for: action)
                    }
                    // S4: MoveToMenu
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }
            Spacer(minLength: 12)
        }
        .padding(.vertical, 4)
        .help(entry.id)
    }

    private var sentence: String {
        switch hold {
        case let .missingModel(_, sentence): sentence
        case let .prose(sentence, _): sentence
        }
    }

    @ViewBuilder
    private func button(for action: Action) -> some View {
        switch action {
        case let .pullThenRetry(model):
            Button("Pull \(model), then Retry") { pullThenRetry(model) }
        case .tryAgain:
            Button("Try Again", action: tryAgain)
        }
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
}

/// The Pull-then-Retry button's own orchestration (design decision 12):
/// `downloads.install`, then a retry ONLY once that machine's download for
/// that model settles as a success -- never a second install path, and
/// never an automatic retry on a cancelled or failed one, where it would
/// just hold again.
extension QueueHoldRow {
    static func pullThenRetry(
        _ model: String, entry: QueueEntry, host: MoldHost,
        downloads: DownloadStore, queue: QueueStore
    ) async {
        await downloads.install(model, on: host)
        guard await downloads.awaitSettlement(of: model, on: host.id) else { return }
        await queue.retry(entry, on: host.id)
    }
}
