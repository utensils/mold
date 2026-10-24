import MoldClient
import MoldStyle
import SwiftUI

/// A duration parent or an authored sequence, in `QueueRow`'s own chrome: the
/// fixed thumbnail slot, a state glyph, the model headline over the shared
/// caption, icon-only buttons and the row's contextual menu. Clicking it opens
/// the details sheet. Every surface dispatches through `NativeVideoQueue`.
struct VideoQueueRow: View {
    let item: NativeVideoQueueItem
    let queue: NativeVideoQueue
    let inspect: () -> Void

    var body: some View {
        HStack(alignment: .center, spacing: 12) {
            Button(action: inspect) {
                HStack(alignment: .center, spacing: 12) {
                    VStack(alignment: .leading, spacing: 3) {
                        MediaPreviewPlaceholder(state: .noStillPreview)
                            .frame(width: 48, height: 48)
                            .clipShape(.rect(cornerRadius: 6))
                            .accessibilityHidden(true)
                        Text("Video").font(.caption).foregroundStyle(.secondary).lineLimit(1)
                    }
                    .frame(width: 80, alignment: .leading)
                    Image(systemName: Self.glyph(item.row.symbol))
                        .foregroundStyle(.tertiary)
                        .frame(width: 16)
                    VStack(alignment: .leading, spacing: 2) {
                        Text(item.headline)
                        Text(caption)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .lineLimit(2)
                        if let note = item.note {
                            Text(note).font(.caption).foregroundStyle(.secondary).lineLimit(2)
                        }
                        if let progress = item.row.progress {
                            ProgressView(value: progress).controlSize(.small).frame(maxWidth: 160)
                        }
                    }
                    Spacer(minLength: 12)
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Details for \(item.headline)")
            buttons
        }
        .padding(.vertical, 3)
        .rowActionMenu(item.menu(stopRequested: queue.stopRequested(item)), perform: { queue.perform($0, on: item) })
        .help("Show this video’s progress, settings and available actions")
        .nativeVideoQueueRow(item, queue: queue)
        .accessibilityIdentifier("video-queue-row-" + item.operation)
    }

    private var caption: String {
        item.row.detail.isEmpty ? item.row.caption : "\(item.row.caption) · \(item.row.detail)"
    }

    /// Stop and Resume as glyphs, like Cancel and Resume on an ordinary row.
    /// An explicit button style keeps each one its own gesture inside List.
    @ViewBuilder private var buttons: some View {
        HStack(spacing: 4) {
            if item.offers(.resume) {
                Button { queue.perform(.resume, on: item) } label: { Image(systemName: "play") }
                    .help("Resume this video from its last finished clip")
                    .accessibilityLabel("Resume")
            }
            if item.offers(.openInLibrary) {
                Button { queue.perform(.openInLibrary, on: item) } label: { Image(systemName: "photo.on.rectangle") }
                    .help("Show the finished video in the Library")
                    .accessibilityLabel("Open in Library")
            }
            if item.offers(.dismiss) {
                Button { queue.perform(.dismiss, on: item) } label: { Image(systemName: "checkmark.circle") }
                    .help("Dismiss this finished video from the Queue on this Mac")
                    .accessibilityLabel("Dismiss from Queue")
            }
            if item.offers(.stop) {
                Button { queue.perform(.stop, on: item) } label: { Image(systemName: "xmark") }
                    .help("Stop this video. Finished clips are kept so it can resume.")
                    .accessibilityLabel(item.title(.stop, stopRequested: queue.stopRequested(item)))
            }
        }
        .buttonStyle(.borderless)
        .labelStyle(.iconOnly)
        .disabled(item.isBusy)
    }

    static func glyph(_ symbol: VideoJobRow.Symbol) -> String {
        switch symbol {
        case .waiting: "clock"
        case .rendering, .joining: "circle.dotted"
        case .stopping, .stopped: "xmark.circle"
        case .paused, .interrupted: "pause.circle"
        case .failed: "exclamationmark.triangle"
        case .done: "checkmark.circle"
        }
    }
}

/// The details sheet a video row opens.
struct VideoQueueDetailSheet: View {
    let queue: NativeVideoQueue
    let id: String
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            NativeVideoQueueDetails(queue: queue, id: id)
            HStack {
                Spacer()
                Button("Done") { dismiss() }.keyboardShortcut(.defaultAction)
            }
        }
        .padding(20)
        .frame(minWidth: 420, idealWidth: 480)
    }
}
