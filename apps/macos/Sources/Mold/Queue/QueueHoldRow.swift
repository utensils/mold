import MoldClient
import SwiftUI

/// The only row that draws its cause as a paragraph and offers named
/// buttons instead of glyphs -- because it is the only row asking for a
/// decision (design M6 "Where everything goes").
struct QueueHoldRow: View {
    let entry: QueueEntry
    let hold: QueueHold
    let pullThenRetry: (String) -> Void
    let tryAgain: () -> Void
    let moveToDestinations: [TransferStore.TransferDestination]
    let moveTo: (MoldHost.ID) -> Void
    /// `DELETE /api/queue/:id` is the documented way to clear a held row
    /// (`routes.rs:7495-7499`), and it is the one action EVERY hold has --
    /// a hold the machine says retrying will not fix, on a fleet with
    /// nowhere to send it, used to offer nothing at all.
    let cancel: () -> Void

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
                    MoveToMenu(destinations: moveToDestinations, send: moveTo)
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }
            Spacer(minLength: 12)
            // The same glyph, in the same place, as every other row's --
            // so the eye finds Cancel at one edge whatever the row is.
            Button(action: cancel) { Image(systemName: "xmark") }
                .buttonStyle(.borderless)
                .help("Cancel this job")
        }
        .padding(.vertical, 4)
        // The row's own buttons a second way -- a contextual menu is where a
        // Mac user looks first for "get rid of this", and the row had none.
        // The SAME list, drawn by the app's one renderer: it was a
        // hand-written `@ViewBuilder` beside a `menuTitles` a test read, which
        // is two lists that agreed by hand.
        .rowActionMenu(Self.offered(for: hold, destinations: moveToDestinations),
                       perform: perform)
        .help(entry.id)
    }

    private var sentence: String {
        switch hold {
        case let .missingModel(_, sentence): sentence
        case let .prose(sentence, _): sentence
        }
    }

    private func button(for action: Action) -> some View {
        Button(action.title) {
            switch action {
            case let .pullThenRetry(model): pullThenRetry(model)
            case .tryAgain: tryAgain()
            }
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
