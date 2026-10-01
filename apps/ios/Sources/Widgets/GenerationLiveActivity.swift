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
                ActivityBrandIcon().accessibilityHidden(true)
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

    var body: some View {
        ActivityBrandIcon()
        .clipShape(.rect(cornerRadius: 8))
        .accessibilityHidden(true) // a11y: the sentence beside it says what is happening.
    }
}

private struct ActivityRing: View {
    let state: GenerationActivityAttributes.ContentState

    var body: some View {
        switch state.phase {
        case .running:
            ZStack {
                ActivityBrandIcon().padding(3)
                ProgressView(value: state.fraction ?? 0)
                    .progressViewStyle(.circular)
            }
            .accessibilityElement(children: .ignore)
            .accessibilityLabel(Text(state.sentence))
        case .finished:
            ActivityBrandIcon().accessibilityLabel("Finished")
        case .failed:
            ActivityBrandIcon().accessibilityLabel("Didn't finish")
        }
    }
}
