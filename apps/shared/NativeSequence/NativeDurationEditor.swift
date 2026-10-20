import MoldClient
import MoldStyle
import SwiftUI

/// One duration beside the ordinary prompt; optional scenes share its recipe.
struct NativeDurationEditor: View {
    @Environment(\.dynamicTypeSize) private var typeSize
    @Binding var draft: VideoDurationDraft
    @Bindable var planner: VideoPlanController
    let prompt: String
    let singleClipMaxFrames: Int?
    let fps: Int
    var sourceTitle: String? = nil
    var sourceMachine: String? = nil
    var busy = false
    let retry: () -> Void
    let startFresh: () -> Void

    private var scenesActive: Bool { draft.scenesEnabled && !draft.scenes.isEmpty }
    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            if let source = draft.source {
                Text("Extending \(sourceTitle ?? source.filename)").font(.headline)
                    .fixedSize(horizontal: false, vertical: true)
                if let sourceMachine { Text("On \(sourceMachine)").font(.caption).foregroundStyle(.secondaryText) }
                Text("The original and new footage save together as a new video.")
                    .font(.caption).foregroundStyle(.secondaryText)
                Button("Start new video", action: startFresh).frame(minHeight: 44)
                    .accessibilityIdentifier("duration-start-fresh")
            }
            if scenesActive {
                Text("\(draft.source == nil ? "Duration" : "Add duration"): \(draft.seconds) seconds").font(.headline)
                LazyVStack(alignment: .leading, spacing: 12) {
                    ForEach($draft.scenes) { $scene in
                        NativeDurationScene(scene: $scene,
                            number: (draft.scenes.firstIndex { $0.id == scene.id } ?? 0) + 1,
                            remove: { draft.scenes.removeAll { $0.id == scene.id } }, canRemove: draft.scenes.count > 1)
                    }
                }
                let actions = typeSize.isAccessibilitySize ? AnyLayout(VStackLayout(alignment: .leading, spacing: 8)) : AnyLayout(HStackLayout(spacing: 12))
                actions {
                    Button("Add scene", systemImage: "plus") { draft.scenes.append(.init(prompt: "", seconds: "5")) }
                    if !typeSize.isAccessibilitySize { Spacer(minLength: 8) }
                    Button("Use one prompt") { draft.scenesEnabled = false }
                }.frame(minHeight: 44)
                Text("References apply to every scene. Opening and closing frames bound the whole video.")
                    .font(.caption).foregroundStyle(.secondaryText)
            } else {
                NativeDurationField(title: draft.source == nil ? "Duration" : "Add duration", seconds: $draft.seconds)
                Button("Define scenes") {
                    if draft.scenes.isEmpty { draft.scenes = [.init(prompt: prompt, seconds: draft.seconds)] }
                    draft.scenesEnabled = true
                    draft.updateDurationFromScenes()
                }.frame(minHeight: 44).accessibilityIdentifier("duration-define-scenes")
            }
            if let maximum = planner.plan?.singleClipMaxFrames ?? singleClipMaxFrames {
                let rate = max(1, planner.plan?.fps ?? fps)
                Text("Single generation: up to \(Double(maximum) / Double(rate), format: .number.precision(.fractionLength(0 ... 2))) seconds")
                    .font(.caption).foregroundStyle(.secondaryText)
                    .accessibilityIdentifier("duration-single-limit")
            }
            if planner.isCalculating {
                Text("Calculating length…").font(.caption).foregroundStyle(.secondaryText)
            } else if let plan = planner.plan {
                Text(plan.clipCount == 1 ? "1 clip · one video" : "\(plan.clipCount) clips joined into one video")
                    .font(.callout).accessibilityIdentifier("duration-plan-clips")
                if plan.joinMode == .blendedShots, plan.clipCount > 1 {
                    Text("Shots blend together; motion does not carry between them.")
                        .font(.caption).foregroundStyle(.secondaryText)
                }
                Text("\(draft.source == nil ? "Video" : "New total"): \(VideoDurationDraft.seconds(milliseconds: plan.totalDurationMs)) seconds")
                    .font(.caption).foregroundStyle(.secondaryText)
                ForEach(Array(plan.warnings.enumerated()), id: \.offset) { _, warning in
                    Text(warning).font(.caption).foregroundStyle(.secondaryText)
                }
            }
            if let error = planner.error {
                Text(error).font(.callout).fixedSize(horizontal: false, vertical: true)
                Button("Retry length check", action: retry).frame(minHeight: 44)
            }
        }
        .disabled(busy)
        .onChange(of: draft.scenes) { _, _ in draft.updateDurationFromScenes() }
    }
}
