import MoldClient
import SwiftUI

/// The canvas (DESIGN.md §5.1): guidance when empty; while a render runs, its
/// denoise preview under a glass plate that says what is happening in words
/// and in mono ("Adding detail — about 12s left · denoise 18/28"); then the
/// result, paging through a batch, with its actions along the bottom.
struct GenerateCanvas: View {
    @Environment(GenerateController.self) private var generate
    @Environment(LibraryStore.self) private var library

    var body: some View {
        switch generate.run {
        case .idle:
            EmptyState(title: generate.kind == .clip ? String(localized: "Describe a clip below")
                                                     : String(localized: "Describe a picture below"),
                       symbol: Destination.generate.symbol,
                       message: String(localized: "What you make appears here, and in the Library on its machine."))
        case .submitting:
            ProgressView("Sending to the machine…")
        case let .running(status, progress):
            RunningView(status: status, progress: progress)
        case let .finished(outcome, host):
            ResultPager(outcome: outcome, host: host)
        case let .failed(reason):
            EmptyState(title: String(localized: "That didn't finish"), symbol: "exclamationmark.triangle",
                       message: reason)
                .sensoryFeedback(.warning, trigger: reason)
        }
    }
}

/// A render in flight: the latest preview, and the progress in words.
private struct RunningView: View {
    @Environment(GenerateController.self) private var generate
    let status: BatchStatus
    let progress: JobProgress?

    var body: some View {
        ZStack(alignment: .bottom) {
            if let data = progress?.previewData, let preview = UIImage(data: data) {
                Image(uiImage: preview).resizable().scaledToFit()
                    .clipShape(.rect(cornerRadius: 12))
                    .accessibilityLabel("Preview of the render in progress")
            } else {
                Color.clear
            }
            ProgressPlate(status: status, progress: progress, waiting: generate.queued.count) {
                generate.stop()
            } stopAll: {
                generate.stop(everything: true)
            }
            .padding(12)
        }
        .padding(.horizontal, 16)
    }
}

/// The glass plate over a running render. The sentence is primary; the mono
/// figure is secondary; at large sizes the figure wraps under it.
struct ProgressPlate: View {
    let status: BatchStatus
    let progress: JobProgress?
    let waiting: Int
    let stop: () -> Void
    let stopAll: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .firstTextBaseline) {
                Text(ProgressWords.sentence(progress, position: queuePosition))
                    .font(.headline)
                    .fixedSize(horizontal: false, vertical: true)
                Spacer(minLength: 8)
                Menu {
                    Button("Stop Everything from Here", role: .destructive, action: stopAll)
                } label: {
                    Label("Stop", systemImage: "stop.fill")
                } primaryAction: {
                    stop()
                }
                .buttonStyle(.bordered)
            }
            if let figure = ProgressWords.figure(progress) {
                Text(verbatim: figure).font(.caption.monospacedDigit()).foregroundStyle(.secondaryText)
            }
            if let done = progress?.step, let total = progress?.total, total > 0 {
                ProgressView(value: Double(done), total: Double(total))
                    .accessibilityValue(ProgressWords.spoken(progress))
            } else {
                ProgressView()
            }
            if waiting > 0 {
                Text("+\(waiting) waiting").font(.caption).foregroundStyle(.secondaryText)
            }
        }
        .padding(14)
        .glassEffect(.regular, in: .rect(cornerRadius: 16))
        .accessibilityElement(children: .contain)
    }

    private var queuePosition: Int? {
        guard status.children.allSatisfy({ $0.state == .accepted }) else { return nil }
        return progress?.queuePosition
    }
}
