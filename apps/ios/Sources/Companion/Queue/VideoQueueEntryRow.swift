import MoldClient
import MoldStyle
import SwiftUI

/// A duration parent or an authored sequence, drawn like `QueueEntryRow`: the
/// placeholder slot, the model headline, the shared caption, and the native
/// swipes and contextual menu. Trailing **Stop** is Cancel's destructive
/// swipe; leading **Resume** / **Start again…** appear where allowed. Every
/// gesture dispatches through `NativeVideoQueue`, so one gesture is one action.
struct VideoQueueEntryRow: View {
    @Environment(\.dynamicTypeSize) private var size
    @ScaledMetric(relativeTo: .body) private var thumb = 52
    let item: NativeVideoQueueItem
    let queue: NativeVideoQueue

    var body: some View {
        let layout = RowAxis.for(size) == .horizontal
            ? AnyLayout(HStackLayout(alignment: .top, spacing: 12))
            : AnyLayout(VStackLayout(alignment: .leading, spacing: 8))
        VStack(alignment: .leading, spacing: 8) {
            Button { queue.presenter.inspecting = item.id } label: {
                layout {
                    MediaPreviewPlaceholder(state: .noStillPreview)
                        .frame(width: thumb, height: thumb)
                        .accessibilityHidden(true)
                    VStack(alignment: .leading, spacing: 6) {
                        Text(verbatim: item.headline).font(.headline).foregroundStyle(.primary)
                        if let prompt = item.prompt, !prompt.isEmpty {
                            Text(prompt).font(.callout).foregroundStyle(.primary)
                                .lineLimit(size.isAccessibilitySize ? nil : 2)
                        }
                        HStack {
                            Label(item.row.caption, systemImage: Self.glyph(item.row.symbol))
                            Spacer()
                            Image(systemName: "chevron.right").accessibilityHidden(true)
                        }
                        .font(.callout).foregroundStyle(.secondaryText)
                        if !item.row.detail.isEmpty {
                            Text(item.row.detail).font(.caption).foregroundStyle(.secondaryText)
                        }
                        if let note = item.note {
                            Text(note).font(.caption).foregroundStyle(.secondaryText)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                        if let progress = item.row.progress {
                            ProgressView(value: progress).accessibilityValue(item.row.caption)
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
                .contentShape(.rect)
            }
            .buttonStyle(.plain)
            .accessibilityIdentifier("video-queue-open-" + item.operation)
            .accessibilityHint("Show video details and controls")
            // A finished parent's visible one-tap actions keep their own
            // independent button style inside the List row.
            if item.offers(.dismiss) || item.offers(.openInLibrary) {
                HStack(spacing: 16) {
                    if item.offers(.openInLibrary) {
                        Button("Open in Library", systemImage: "photo.on.rectangle") { queue.perform(.openInLibrary, on: item) }
                            .frame(minHeight: 44)
                    }
                    if item.offers(.dismiss) {
                        Button("Dismiss from Queue", systemImage: "checkmark.circle") { queue.perform(.dismiss, on: item) }
                            .frame(minHeight: 44)
                    }
                }
                .buttonStyle(.borderless)
                .disabled(item.isBusy)
            }
        }
        .padding(.vertical, 8)
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("video-queue-entry-" + item.operation)
        .swipeActions(edge: .trailing, allowsFullSwipe: false) {
            if item.offers(.stop) {
                Button(role: .destructive) { queue.perform(.stop, on: item) } label: {
                    Label(item.title(.stop, stopRequested: queue.stopRequested(item)), systemImage: "xmark")
                }
                .accessibilityIdentifier("video-queue-swipe-stop-" + item.operation)
            }
        }
        .swipeActions(edge: .leading, allowsFullSwipe: false) {
            if item.offers(.resume) {
                Button { queue.perform(.resume, on: item) } label: { Label("Resume", systemImage: "play") }
                    .tint(.orange)
                    .accessibilityIdentifier("video-queue-swipe-resume-" + item.operation)
            } else if item.offers(.startAgain) {
                Button { queue.perform(.startAgain, on: item) } label: {
                    Label("Start again…", systemImage: "arrow.counterclockwise")
                }
                .accessibilityIdentifier("video-queue-swipe-restart-" + item.operation)
            } else {
                Button("Details", systemImage: "info.circle") { queue.presenter.inspecting = item.id }
            }
        }
        .contextMenu { menu }
        .nativeVideoQueueRow(item, queue: queue)
    }

    /// The same list the swipes and buttons offer; destructive last.
    @ViewBuilder private var menu: some View {
        let drawn = item.menu(stopRequested: queue.stopRequested(item))
        let firstDestructive = drawn.firstIndex { $0.isDestructive }
        ForEach(Array(drawn.enumerated()), id: \.offset) { offset, action in
            if let kind = action.kind {
                if offset == firstDestructive, offset > 0 { Divider() }
                Button(action.title, systemImage: NativeVideoQueueItem.symbol(kind),
                       role: action.isDestructive ? .destructive : nil) { queue.perform(kind, on: item) }
            }
        }
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

/// The details a video row opens.
struct VideoQueueDetailSheet: View {
    let queue: NativeVideoQueue
    let id: String
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            ScrollView {
                NativeVideoQueueDetails(queue: queue, id: id).padding()
            }
            .navigationTitle("Video Details")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar { ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } } }
        }
        .presentationDetents([.medium, .large])
    }
}
