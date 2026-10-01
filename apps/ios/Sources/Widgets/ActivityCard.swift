import ActivityKit
import SwiftUI
import WidgetKit

struct LockScreenActivity: View {
    let context: ActivityViewContext<GenerationActivityAttributes>

    var body: some View {
        // ActivityKit caps the Lock Screen at 160 pt. Drop secondary content
        // before clipping the status or Stop control at larger text sizes.
        ViewThatFits(in: .vertical) {
            ActivityCard(context: context, compact: false)
            ActivityCard(context: context, compact: true)
        }
        .frame(maxHeight: ActivityCardLayout.contentHeight, alignment: .center)
    }
}

private struct ActivityCard: View {
    let context: ActivityViewContext<GenerationActivityAttributes>
    let compact: Bool

    var body: some View {
        let content = ActivityCardContent(state: context.state, machine: context.attributes.machine, isStale: context.isStale)
        VStack(alignment: .leading, spacing: ActivityCardLayout.rowSpacing) {
            HStack(alignment: .center, spacing: 12) {
                ActivityPreview(context: context).frame(width: ActivityCardLayout.previewSide, height: ActivityCardLayout.previewSide)
                VStack(alignment: .leading, spacing: ActivityCardLayout.titleSpacing) {
                    if !compact {
                        Text("Mold Studio").font(.caption2.weight(.semibold)).foregroundStyle(.secondary)
                    }
                    Text(content.title).font(.subheadline.weight(.semibold)).lineLimit(2)
                    if !compact {
                        Text(context.attributes.prompt).font(.caption).foregroundStyle(.secondary).lineLimit(2)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                if content.canStop { ActivityStop(clientBatchId: context.attributes.clientBatchId) }
            }
            ActivityProgress(context: context)
            if !compact, context.state.phase != .finished || context.state.waiting > 0 {
                HStack(spacing: 8) {
                    if context.state.phase != .finished {
                        Label(content.machine, systemImage: "desktopcomputer").lineLimit(1)
                    }
                    Spacer(minLength: 0)
                    if context.state.waiting > 0 {
                        Text("+\(context.state.waiting) waiting").lineLimit(1)
                    }
                }
                .font(.caption2)
                .foregroundStyle(.secondary)
            }
        }
    }
}

struct ActivityStop: View {
    let clientBatchId: String

    var body: some View {
        Button(intent: StopRenderIntent(clientBatchId: clientBatchId)) {
            Image(systemName: "stop.fill").font(.caption.weight(.semibold))
                .frame(width: ActivityCardLayout.stopSide, height: ActivityCardLayout.stopSide)
        }
        .buttonStyle(.plain)
        .foregroundStyle(.red)
        .background(.red.opacity(0.12), in: .circle)
        .accessibilityLabel("Stop Render")
    }
}

private struct ActivityProgress: View {
    let context: ActivityViewContext<GenerationActivityAttributes>

    var body: some View {
        let content = ActivityCardContent(state: context.state, machine: context.attributes.machine, isStale: context.isStale)
        if context.state.phase == .running, !context.isStale {
            VStack(alignment: .leading, spacing: ActivityCardLayout.progressSpacing) {
                if let progress = content.progress {
                    ProgressView(value: progress).tint(.cyan)
                        .frame(height: ActivityCardLayout.progressHeight)
                        .accessibilityLabel("Render progress")
                }
                HStack {
                    if let detail = content.detail {
                        Text(verbatim: detail).lineLimit(1)
                    }
                    Spacer(minLength: 4)
                    if let end = context.state.endsAt, end > .now {
                        Text(timerInterval: Date.now...end, countsDown: true)
                            .multilineTextAlignment(.trailing).frame(maxWidth: 64)
                    }
                }
                .font(.caption2.monospacedDigit())
                .foregroundStyle(.secondary)
            }
        }
    }
}

struct ActivityDetail: View {
    let context: ActivityViewContext<GenerationActivityAttributes>

    var body: some View {
        let content = ActivityCardContent(state: context.state, machine: context.attributes.machine, isStale: context.isStale)
        VStack(alignment: .leading, spacing: 8) {
            Text(content.title).font(.headline)
            Text(context.attributes.prompt).font(.caption).foregroundStyle(.secondary).lineLimit(2)
            ActivityProgress(context: context)
            HStack {
                if context.state.phase != .finished { Text(context.attributes.machine).lineLimit(1) }
                Spacer(minLength: 4)
                if context.state.waiting > 0 { Text("+\(context.state.waiting) waiting") }
            }
            .font(.caption2).foregroundStyle(.secondary)
        }
    }
}

