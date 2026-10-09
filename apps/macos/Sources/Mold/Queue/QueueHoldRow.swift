import MoldClient
import SwiftUI

/// The only row that draws its cause as a paragraph and offers named
/// buttons instead of glyphs -- because it is the only row asking for a
/// decision (design M6 "Where everything goes").
struct QueueHoldRow: View {
    @Environment(DownloadStore.self) private var downloads
    let entry: QueueEntry
    let hold: QueueHold
    var sourceHost: MoldHost? = nil
    let pullThenRetry: (String) -> Void
    let tryAgain: () -> Void
    let moveToDestinations: [TransferStore.TransferDestination]
    let moveTo: (MoldHost.ID) -> Void
    /// `DELETE /api/queue/:id` is the documented way to clear a held row
    /// (`routes.rs:7495-7499`), and it is the one action EVERY hold has --
    /// a hold the machine says retrying will not fix, on a fleet with
    /// nowhere to send it, used to offer nothing at all.
    let cancel: () -> Void
    var inspect: (() -> Void)? = nil
    let actions: QueueRowActions

    var body: some View {
        HStack(alignment: .center, spacing: 12) {
            if let sourceHost { QueueSourceThumbnail(entry: entry, host: sourceHost) }
            Image(systemName: "exclamationmark.triangle")
                .foregroundStyle(.orange)
                .frame(width: 16)
            VStack(alignment: .leading, spacing: 6) {
                Button { inspect?() } label: { Text(entry.modelHeadline) }
                    .buttonStyle(.plain)
                    .disabled(inspect == nil)
                    .accessibilityLabel("Details for \(entry.modelHeadline)")
                Text(sentence)
                    .font(.callout)
                    .foregroundStyle(.secondary)
                if let recovery {
                    Text(recovery.message).font(.callout).foregroundStyle(.secondary)
                        .accessibilityIdentifier("queue-download-status-" + entry.id)
                    if let fraction = recovery.fraction { ProgressView(value: fraction).accessibilityLabel("Model download") }
                    else if recovery.isBusy { ProgressView().controlSize(.small).accessibilityLabel(recovery.message) }
                }
                ViewThatFits(in: .horizontal) {
                    recoveryControls(horizontal: true)
                    recoveryControls(horizontal: false)
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }
            Spacer(minLength: 12)
            // The same glyph, in the same place, as every other row's --
            // so the eye finds Cancel at one edge whatever the row is.
            if actions.cancel {
                Button(action: cancel) { Image(systemName: "xmark") }
                    .buttonStyle(.borderless)
                    .help("Cancel this job")
                    .accessibilityLabel("Cancel Job")
            }
        }
        .padding(.vertical, 4)
        // The row's own buttons a second way -- a contextual menu is where a
        // Mac user looks first for "get rid of this", and the row had none.
        // The SAME list, drawn by the app's one renderer: it was a
        // hand-written `@ViewBuilder` beside a `menuTitles` a test read, which
        // is two lists that agreed by hand.
        .rowActionMenu(Self.offered(for: hold, destinations: moveToDestinations, actions: actions),
                       perform: perform)
        .help("Show this job’s details and why it is waiting")
    }

    private func recoveryControls(horizontal: Bool) -> some View {
        let layout = horizontal ? AnyLayout(HStackLayout(spacing: 8)) : AnyLayout(VStackLayout(alignment: .leading, spacing: 8))
        return layout {
            ForEach(actions.retry ? Self.actions(for: hold) : [], id: \.self) { action in
                button(for: action)
            }
            MoveToMenu(destinations: moveToDestinations, send: moveTo)
            if let inspect {
                Button("Failure Details", action: inspect)
                    .help("Show the machine’s saved diagnostic for this job")
                    .accessibilityIdentifier("queue-failure-details-" + entry.id)
            }
        }
        .fixedSize(horizontal: horizontal, vertical: true)
    }

    private var sentence: String {
        hold.summary(modelName: entry.modelHeadline, hostName: sourceHost?.name ?? "this machine")
    }

    var recovery: QueueDownloadRecovery.State? {
        sourceHost.flatMap { downloads.queueDownloads.state(host: $0.id, job: entry.id) }
    }

    private func button(for action: Action) -> some View {
        Button(recovery?.isBusy == true ? "Downloading…" : action.title) {
            switch action {
            case let .pullThenRetry(model): pullThenRetry(model)
            case .tryAgain: tryAgain()
            }
        }
        .disabled(recovery?.isBusy == true)
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
        downloads.recover(entry, on: host, queue: queue)
        while downloads.queueDownloads.state(host: host.id, job: entry.id)?.isBusy == true {
            do { try await Task.sleep(for: .milliseconds(100)) } catch { return }
        }
    }
}
