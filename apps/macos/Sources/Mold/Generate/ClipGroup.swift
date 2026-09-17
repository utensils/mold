import MoldClient
import MoldStyle
import SwiftUI

/// Everything that shapes the clip itself -- LTX-2's audio branch, in this
/// slice. S6b adds keyframes, an extend continuation, and the two
/// conditioning wells beneath Sound; those rows widen `isShown` to cover a
/// temporal recipe with no audio branch (wan's `extend`, say), which is why
/// this slice keeps the gate narrow -- a group with a title and nothing
/// inside it is the same trap `GenerateInspector`'s own doc comment warns
/// against.
struct ClipGroup: View {
    let recipe: GenerationRecipe?
    @Binding var draft: RenderDraft

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            soundSection
        }
    }

    @ViewBuilder private var soundSection: some View {
        if recipe?.capabilities.supportsAudio == true {
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
    }
}
