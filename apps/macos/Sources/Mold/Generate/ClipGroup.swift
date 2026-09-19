import MoldClient
import MoldStyle
import SwiftUI

/// Everything that shapes the clip itself: LTX-2's audio branch, keyframe
/// interpolation, an extend continuation, and the two conditioning wells
/// (`ClipGroup+Media.swift`). Every row exists only where the recipe
/// advertises it, so a temporal recipe with no audio branch at all -- wan's
/// `extend`, say -- still has something to show, which is why `isShown`
/// below reads every row rather than just `supportsAudio`.
struct ClipGroup: View {
    let recipe: GenerationRecipe?
    @Binding var draft: RenderDraft

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            soundSection
            keyframesSection
            extendSection
            audioFileSection
            sourceVideoSection
        }
    }

    @ViewBuilder private var soundSection: some View {
        if draft.offersAudioControl {
            LabeledSection("Sound") {
                VStack(alignment: .leading, spacing: 6) {
                    Toggle("Generate audio", isOn: audioBinding)
                    if draft.enableAudio, recipe?.capabilities.output?.audioRequiresMp4 == true {
                        Text("Delivered as MP4.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    videoOnlyRow
                }
            }
        } else if draft.requiresAudio {
            LabeledSection("Sound") {
                Text("Audio is always included.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        } else if draft.audioUnavailableForModel {
            LabeledSection("Sound") {
                Text("Audio is unavailable for this checkpoint.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }

    /// Never a disabled switch: a blocked opt-in says why, in its own words,
    /// rather than a toggle nobody can touch (decision 14, M4 design).
    @ViewBuilder private var videoOnlyRow: some View {
        if let reason = VideoOnlyPolicy.blockedReason(draft.videoOnlyInputs) {
            Text(reason).font(.caption).foregroundStyle(.secondary)
        } else {
            Toggle("Skip the audio branch", isOn: $draft.videoOnly)
        }
    }

    private var audioBinding: Binding<Bool> {
        Binding(
            get: { draft.enableAudio },
            set: { draft = draft.enablingAudio($0, capabilities: recipe?.capabilities) }
        )
    }
}

extension ClipGroup {
    static func isShown(capabilities: RecipeCapabilities) -> Bool {
        capabilities.supportsAudio == true
            || capabilities.acceptsKeyframes
            || capabilities.supportsExtend == true
            || capabilities.acceptsSourceAudio
            || capabilities.acceptsSourceVideo
    }
}
