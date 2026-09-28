import ActivityKit
import AppIntents
import SwiftUI
import WidgetKit

/// The render on the Lock Screen and in the Dynamic Island (DESIGN.md §5.7).
struct GenerationLiveActivity: Widget {
    var body: some WidgetConfiguration {
        ActivityConfiguration(for: GenerationActivityAttributes.self) { context in
            LockScreenActivity(context: context)
                .padding(16)
                .activityBackgroundTint(nil)
                .widgetURL(link(context))
        } dynamicIsland: { context in
            DynamicIsland {
                DynamicIslandExpandedRegion(.leading) {
                    ActivityPreview(context: context).frame(width: 52, height: 52)
                }
                DynamicIslandExpandedRegion(.trailing) {
                    if context.state.phase == .running {
                        Button(intent: StopRenderIntent(clientBatchId: context.attributes.clientBatchId)) {
                            Label("Stop", systemImage: "stop.fill").labelStyle(.iconOnly)
                        }
                        .tint(.red)
                    }
                }
                DynamicIslandExpandedRegion(.bottom) {
                    ActivityDetail(context: context)
                }
            } compactLeading: {
                // a11y: the ring beside it carries the progress in words.
                Image(systemName: "wand.and.sparkles").accessibilityHidden(true)
            } compactTrailing: {
                ActivityRing(state: context.state)
            } minimal: {
                ActivityRing(state: context.state)
            }
            .widgetURL(link(context))
        }
    }

    private func link(_ context: ActivityViewContext<GenerationActivityAttributes>) -> URL {
        DeepLink.queue(job: nil).url
    }
}

private struct LockScreenActivity: View {
    let context: ActivityViewContext<GenerationActivityAttributes>

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            ActivityPreview(context: context).frame(width: 64, height: 64)
            VStack(alignment: .leading, spacing: 4) {
                ActivityDetail(context: context)
            }
            if context.state.phase == .running {
                Button(intent: StopRenderIntent(clientBatchId: context.attributes.clientBatchId)) {
                    Label("Stop", systemImage: "stop.fill").labelStyle(.iconOnly)
                }
                .buttonStyle(.bordered)
                .tint(.red)
            }
        }
    }
}

private struct ActivityDetail: View {
    let context: ActivityViewContext<GenerationActivityAttributes>

    var body: some View {
        let state = context.state
        VStack(alignment: .leading, spacing: 4) {
            HStack(alignment: .firstTextBaseline) {
                Text(context.isStale ? String(localized: "Open Mold Studio to refresh") : state.sentence)
                    .font(.headline)
                Spacer(minLength: 4)
                if state.phase == .running, let end = state.endsAt, end > .now, !context.isStale {
                    Text(timerInterval: Date.now...end, countsDown: true)
                        .font(.caption.monospacedDigit())
                        .multilineTextAlignment(.trailing)
                        .frame(maxWidth: 64)
                }
            }
            Text(context.attributes.prompt).font(.callout).lineLimit(2)
            if state.phase == .running {
                if let fraction = state.fraction { ProgressView(value: fraction) } else { ProgressView(value: 0) }
            }
            if let figure = state.figure {
                Text(verbatim: figure).font(.caption.monospaced())
            }
            if state.waiting > 0 {
                Text("+\(state.waiting) waiting").font(.caption)
            }
        }
    }
}

private struct ActivityPreview: View {
    let context: ActivityViewContext<GenerationActivityAttributes>

    var body: some View {
        Group {
            if let name = context.state.preview,
               let image = UIImage(contentsOfFile: AppGroup.activityPreviews.appending(path: name).path) {
                Image(uiImage: image).resizable().scaledToFill()
            } else {
                Image(systemName: "wand.and.sparkles").font(.title2).frame(maxWidth: .infinity, maxHeight: .infinity)
                    .background(.quaternary)
            }
        }
        .clipShape(.rect(cornerRadius: 8))
        .accessibilityHidden(true) // a11y: the sentence beside it says what is happening.
    }
}

private struct ActivityRing: View {
    let state: GenerationActivityAttributes.ContentState

    var body: some View {
        switch state.phase {
        case .running:
            ProgressView(value: state.fraction ?? 0)
                .progressViewStyle(.circular)
                .accessibilityLabel(Text(state.sentence))
        case .finished:
            Image(systemName: "checkmark.circle.fill").foregroundStyle(.green).accessibilityLabel("Finished")
        case .failed:
            Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.orange).accessibilityLabel("Didn't finish")
        }
    }
}
