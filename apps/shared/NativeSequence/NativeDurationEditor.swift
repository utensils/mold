import MoldClient
import MoldStyle
import SwiftUI

/// What a Duration video says beside the ordinary prompt -- the Extend source,
/// the scenes of a Longer video, and the machine's plan. The length itself is
/// `NativeLengthControl`; optional scenes share its recipe.
struct NativeDurationEditor: View {
    @Environment(\.dynamicTypeSize) private var typeSize
    @Binding var draft: VideoDurationDraft
    @Bindable var planner: VideoPlanController
    let prompt: String
    var grid: VideoLengthGrid? = nil
    var reuseNotice: String? = nil
    var sourceTitle: String? = nil
    var sourceMachine: String? = nil
    var busy = false
    var originalReferencesReplaced = false
    var originalReferences: [VideoOriginalReference]? = nil
    var inheritedRecipeNotes: [String] = []
    var sourceNotice: String? = nil
    var retrySource: () -> Void = {}
    let retry: () -> Void
    let startFresh: () -> Void

    var singleGenerationSeconds: Double? {
        guard let plan = planner.plan else { return nil }
        return Double(plan.singleClipMaxFrames) / Double(max(1, plan.fps))
    }

    private var scenesActive: Bool { draft.scenesEnabled && !draft.scenes.isEmpty }
    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            if draft.source == nil, let reuseNotice {
                Text(reuseNotice).font(.callout).fixedSize(horizontal: false, vertical: true)
            }
            if let source = draft.source {
                Text("Extending \(sourceTitle ?? source.filename)").font(.headline)
                    .fixedSize(horizontal: false, vertical: true)
                if let sourceMachine { Text("On \(sourceMachine)").font(.caption).foregroundStyle(.secondaryText) }
                Text("The original and new footage save together as a new video.")
                    .font(.caption).foregroundStyle(.secondaryText)
                Text("Previous opening and closing frames are not reused for the new footage.")
                    .font(.caption).foregroundStyle(.secondaryText)
                ForEach(inheritedRecipeNotes, id: \.self) { note in
                    Text(note).font(.caption).foregroundStyle(.secondaryText)
                }
                if let originalReferences {
                    if !originalReferences.isEmpty {
                        Text(originalReferencesReplaced ? "Original references · replaced by your current references" : "Original references")
                            .font(.caption).foregroundStyle(.secondaryText)
                    }
                    ForEach(originalReferences) { reference in
                        VStack(alignment: .leading, spacing: 4) {
                            Label(reference.descriptor.name ?? reference.member?.displayName ?? "Reference \(reference.id + 1)",
                                  systemImage: reference.member?.sourceSymbol ?? "doc")
                                .font(.callout)
                            Text(originalReferencesReplaced ? "Your current references replace this original input."
                                 : (reference.member == nil ? "Original file unavailable. Attach a replacement reference." : "Retained original reference"))
                                .font(.caption).foregroundStyle(.secondaryText)
                        }
                    }
                } else {
                    Text("Checking original references…").font(.caption).foregroundStyle(.secondaryText)
                }
                if let sourceNotice {
                    Text(sourceNotice).font(.callout).fixedSize(horizontal: false, vertical: true)
                    Button("Check original video", action: retrySource).frame(minHeight: 44)
                }
                Button("Start new video", action: startFresh).frame(minHeight: 44)
                    .accessibilityIdentifier("duration-start-fresh")
            }
            if scenesActive, let grid {
                LazyVStack(alignment: .leading, spacing: 12) {
                    ForEach(Array(draft.scenes.enumerated()), id: \.element.id) { offset, scene in
                        NativeDurationScene(draft: $draft, id: scene.id, grid: grid, number: offset + 1)
                    }
                }
                Button("Add scene", systemImage: "plus") { draft.addScene(on: grid) }
                    .frame(minHeight: 44).accessibilityIdentifier("duration-add-scene")
                Text("Each scene is one generation of up to \(VideoLengthGrid.seconds(tenths: grid.sceneMaxTenths)) seconds and starts where the previous one ends. References apply to every scene; opening and closing frames bound the whole video.")
                    .font(.caption).foregroundStyle(.secondaryText)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if let seconds = singleGenerationSeconds {
                Text("Single generation: up to \(seconds, format: .number.precision(.fractionLength(0 ... 2))) seconds")
                    .font(.caption).foregroundStyle(.secondaryText)
                    .accessibilityIdentifier("duration-single-limit")
            }
            if planner.isCalculating {
                Text("Calculating length…").font(.caption).foregroundStyle(.secondaryText)
            } else if let plan = planner.plan {
                Text(plan.clipCount == 1 ? "1 clip · one video" : "\(Plural.count(Int(clamping: plan.clipCount), of: "clip")) joined into one video")
                    .font(.callout).accessibilityIdentifier("duration-plan-clips")
                if plan.joinMode == .blendedShots, plan.clipCount > 1 {
                    Text("Shots blend together; motion does not carry between them.")
                        .font(.caption).foregroundStyle(.secondaryText)
                }
                // The plan's total is on the frame clock (14.3 s at 24 fps is 344
                // frames, 14.334 s); it reads to the same tenth as the slider.
                Text("\(draft.source == nil ? "Video" : "New total"): \(VideoLengthGrid.label(milliseconds: plan.totalDurationMs))")
                    .font(.caption).foregroundStyle(.secondaryText)
                ForEach(Array(plan.warnings.enumerated()), id: \.offset) { _, warning in
                    Label(warning, systemImage: "exclamationmark.triangle")
                        .font(.callout).fixedSize(horizontal: false, vertical: true)
                        .accessibilityIdentifier("duration-plan-warning")
                }
            }
            if let error = planner.error {
                Text(error).font(.callout).fixedSize(horizontal: false, vertical: true)
                Button("Retry length check", action: retry).frame(minHeight: 44)
            }
        }
        .disabled(busy)
    }
}
