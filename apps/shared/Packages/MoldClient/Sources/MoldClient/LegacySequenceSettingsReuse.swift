import Foundation

/// Older authored videos retain a real per-clip editor, not a flattened prompt
/// or a newly inferred scene schedule. Media is filled only from its archive.
public struct LegacySequenceSettingsReuse: Hashable, Sendable {
    let metadata: OutputMetadata
    public let draft: RenderDraft
    public var sequence: SequenceDraft
    public let motionTailFrames: Int
    public let notice: String

    public init(metadata: OutputMetadata) throws {
        guard let chain = metadata.chain, let recorded = chain.stages, !recorded.isEmpty,
              let tail = chain.motionTailFrames, tail >= 0,
              recorded.count == (chain.stageCount ?? recorded.count) else {
            throw SequenceDraftError.invalid("This older video does not record all of its clip settings. Reopen its original request rather than reusing an incomplete recipe.")
        }
        guard chain.continueFrom == nil else {
            throw SequenceDraftError.invalid("This part depends on an original incoming motion carry. Use its saved job’s Start again when supported, or its original CLI request; it cannot be restored as an independent video.")
        }
        guard metadata.icLoraControl == nil else {
            throw SequenceDraftError.invalid("This video's original conditioning controls cannot be restored here. Use its original retained request.")
        }
        var restored = RenderDraft(reusing: metadata)
        let baseSeed: UInt64
        if let seed = metadata.seed { baseSeed = seed }
        else if let text = recorded.first?.seed, !text.isEmpty,
                text.utf8.allSatisfy({ $0 >= 48 && $0 <= 57 }), let seed = UInt64(text) { baseSeed = seed }
        else { throw SequenceDraftError.invalid("This older video's original seed was not recorded. Reopen its original request rather than inventing clip seeds.") }
        restored.seed = baseSeed; restored.locksSeed = true
        var clips: [SequenceDraft.Stage] = []
        for (index, stage) in recorded.enumerated() {
            guard let prompt = stage.prompt, let frames = stage.frames, frames > 0,
                  let transition = stage.transition, ["smooth", "cut", "fade"].contains(transition) else {
                throw SequenceDraftError.invalid("This older video has incomplete clip prompts, lengths or joins. Reopen its original request.")
            }
            var clip = SequenceDraft.Stage(prompt: prompt, frames: frames, transition: transition)
            clip.input.fadeFrames = stage.fadeFrames
            clip.input.loras = stage.loras ?? []
            clip.input.negativePrompt = index == 0 ? metadata.negativePrompt : nil
            clip.input.generationReferences = try (stage.generationReferences ?? []).map {
                guard let reference = $0.generationReference else { throw MoldClientError.malformedResponse }
                return reference
            }
            if let text = stage.seed {
                guard !text.isEmpty, text.utf8.allSatisfy({ $0 >= 48 && $0 <= 57 }), let seed = UInt64(text) else {
                    throw MoldClientError.malformedResponse
                }
                clip.input.seedOffset = baseSeed ^ seed
            }
            clips.append(clip)
        }
        var sequence = SequenceDraft(prompt: clips[0].prompt, frames: clips[0].frames)
        sequence.stages = clips
        restored.frames = clips[0].frames
        restored.prompt = clips[0].prompt
        // Per-clip values own these controls. A global fallback must not
        // overwrite a deliberately empty stack or an unrecorded later prompt.
        restored.negativePrompt = ""
        restored.media = DraftMedia()
        if let fit = metadata.sourceFit { restored.media.sourceFit = fit }
        restored.prefersAutomaticDuration = false
        self.metadata = metadata
        self.draft = restored; self.sequence = sequence; motionTailFrames = tail
        notice = "Restored the recorded clip settings. Older videos do not record per-clip negative prompts; only the recorded first-clip value is restored."
            + (recorded.contains { $0.seed == nil } ? " Some original clip seeds were not recorded." : "")
    }
}
