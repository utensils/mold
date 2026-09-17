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
        if enabled, capabilities?.output?.audioRequiresMp4 == true {
            draft.outputFormat = "mp4"
        }
        return draft
    }

    /// `VideoOnlyPolicy`'s four conflicts, read off THIS draft -- so the Clip
    /// group's sentence and `RenderDraft+Request.swift`'s wire value agree.
    var videoOnlyInputs: VideoOnlyPolicy.Inputs {
        VideoOnlyPolicy.Inputs(
            audioEnabled: enableAudio,
            audioOnlyPipeline: pipeline == "t2a",
            hasConditioningAudio: audioFile != nil,
            isExtend: extendVideo != nil
        )
    }
}
