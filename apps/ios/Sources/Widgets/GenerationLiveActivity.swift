import ActivityKit
import AppIntents
import SwiftUI
import WidgetKit

/// The render on the Lock Screen and in the Dynamic Island (DESIGN.md §5.7).
struct GenerationLiveActivity: Widget {
    var body: some WidgetConfiguration {
        ActivityConfiguration(for: GenerationActivityAttributes.self) { context in
            LockScreenActivity(context: context)
                .padding(ActivityCardLayout.inset)
                .activityBackgroundTint(Color(uiColor: .systemBackground).opacity(0.88))
                .widgetURL(link(context))
        } dynamicIsland: { context in
            DynamicIsland {
                DynamicIslandExpandedRegion(.leading) {
                    ActivityPreview(context: context).frame(width: 52, height: 52)
                }
                DynamicIslandExpandedRegion(.trailing) {
                    if context.state.phase == .running {
                        ActivityStop(clientBatchId: context.attributes.clientBatchId)
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
        context.attributes.link(for: context.state).url
    }
}

struct ActivityPreview: View {
    let context: ActivityViewContext<GenerationActivityAttributes>

    private var symbol: String {
        switch context.state.phase {
        case .running: "wand.and.sparkles"
        case .finished: "checkmark"
        case .failed: "exclamationmark"
        }
    }

    var body: some View {
        Group {
            if let name = context.state.preview,
               let image = UIImage(contentsOfFile: AppGroup.activityPreviews.appending(path: name).path) {
                Image(uiImage: image).resizable().scaledToFill()
            } else {
                Image(systemName: symbol).font(.title3.weight(.medium))
                    .foregroundStyle(.white)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .background(LinearGradient(colors: [.cyan, .indigo], startPoint: .topLeading, endPoint: .bottomTrailing))
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
