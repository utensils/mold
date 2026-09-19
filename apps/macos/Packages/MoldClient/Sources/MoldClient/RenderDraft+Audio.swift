import Foundation

// LTX-2's audio branch and `video_only`, split out of `RenderDraft+Recipe.swift`
// and `RenderDraft+Request.swift` purely for size -- both files were already
// close to the 150-line lint before this pair of concerns existed.
public extension RenderDraft {
    /// Turning audio on pins the output format where the recipe's own
    /// `audio_requires_mp4` says so (`validation.rs:3550-3555`); the Output
    /// group's own caption explains why.
    func enablingAudio(_ enabled: Bool, capabilities: RecipeCapabilities?) -> RenderDraft {
        var draft = self
        draft.enableAudio = enabled
        if draft.enableAudio, capabilities?.output?.audioRequiresMp4 == true {
            draft.outputFormat = "mp4"
        }
        return draft
    }

    /// Records a real format choice and reconciles LTX's container contract:
    /// optional generated audio can only be delivered in MP4.
    func selectingOutputFormat(
        _ format: String, output: OutputCapabilities?
    ) -> RenderDraft {
        var draft = self
        draft.outputFormat = format
        if draft.supportsAudio,
           draft.usesOptionalAudioBranch,
           output?.audioRequiresMp4 == true,
           format.lowercased() != "mp4" {
            draft.enableAudio = false
        }
        return draft
    }

    /// Reconciles recipe support, the model-row asset veto and fixed-audio
    /// families without changing the person's parked preference.
    mutating func reconcileAudio(
        recipe: GenerationRecipe, family: String?, modelSupportsAudio: Bool?
    ) {
        let normalized = family?.trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        let isH3 = ["minimax-h3", "minimax_h3", "minimaxh3"].contains(normalized)
        let isLTX2 = ["ltx2", "ltx-2"].contains(normalized)
        let recipeSupportsAudio = recipe.temporal != nil
            && (recipe.capabilities.supportsAudio ?? (isH3 || isLTX2))
        audioUnavailableForModel = recipe.temporal != nil && isLTX2
            && (recipe.capabilities.supportsAudio == false || modelSupportsAudio == false)
        supportsAudio = recipeSupportsAudio && !audioUnavailableForModel
        requiresAudio = supportsAudio && (isH3 || recipe.requestSelector?.pipeline == "t2a")
        usesOptionalAudioBranch = recipe.temporal != nil && isLTX2
            && recipe.requestSelector?.pipeline != "t2a"
        offersAudioControl = supportsAudio && !requiresAudio
            && isLTX2
    }

    /// `VideoOnlyPolicy`'s four conflicts, read off THIS draft -- so the Clip
    /// group's sentence and `RenderDraft+Request.swift`'s wire value agree.
    var videoOnlyInputs: VideoOnlyPolicy.Inputs {
        VideoOnlyPolicy.Inputs(
            audioEnabled: enableAudio,
            audioOnlyPipeline: pipeline == "t2a",
            hasConditioningAudio: media.audioFile != nil,
            isExtend: media.extendVideo != nil
        )
    }
}
