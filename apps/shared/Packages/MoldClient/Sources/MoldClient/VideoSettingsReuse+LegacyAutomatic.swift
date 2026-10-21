import Foundation

public extension VideoSettingsReuse {
    /// Only an explicitly automatic, demonstrably uniform old recipe can be
    /// expressed as one prompt. Saved length is a fact, not an invented request.
    init(legacyAutomatic legacy: LegacySequenceSettingsReuse) throws {
        let metadata = legacy.metadata, stages = legacy.sequence.stages
        func refuse() -> SequenceDraftError {
            .invalid("This older automatic video's exact inputs or clip choices cannot be represented as one prompt and duration. Use its original request or Start again from the saved job.")
        }
        guard metadata.outputMode == "one-shot", let first = stages.first, let baseSeed = legacy.draft.seed,
              let frames = metadata.frames, frames > 0, let fps = metadata.fps,
              fps.isFinite, fps > 0, fps <= Double(Int32.max), fps.rounded() == fps,
              (metadata.negativePrompt ?? "").isEmpty,
              first.keyframes?.allSatisfy({ $0.frame == 0 }) ?? true else { throw refuse() }
        let records = metadata.chain?.stages ?? []
        guard records.count == stages.count, records.allSatisfy({ $0.seed != nil }) else { throw refuse() }
        let uniformSeeds = stages.allSatisfy { $0.input.seedOffset == first.input.seedOffset }
        let indexedSeeds = stages.enumerated().allSatisfy { index, stage in
            stage.input.seedOffset == (first.input.seedOffset ?? 0) ^ UInt64(index)
        }
        guard uniformSeeds || indexedSeeds else { throw refuse() }
        var delivered = first.frames
        for (index, stage) in stages.enumerated() {
            guard stage.prompt == first.prompt, stage.transition == "smooth", stage.input.fadeFrames == nil,
                  stage.input.loras == first.input.loras,
                  (stage.generationReferences ?? []) == (first.generationReferences ?? []),
                  index == 0 || (stage.input.sourceImage == nil && (stage.keyframes ?? []).isEmpty),
                  (stage.generationReferences ?? []).allSatisfy({ $0.media.authority == "inline" }) else { throw refuse() }
            if index > 0 {
                guard stage.frames > legacy.motionTailFrames else { throw refuse() }
                let (next, overflow) = delivered.addingReportingOverflow(stage.frames - legacy.motionTailFrames)
                guard !overflow else { throw refuse() }
                delivered = next
            }
        }
        guard delivered == frames else { throw refuse() }
        var restored = legacy.draft
        restored.prompt = first.prompt
        restored.seed = baseSeed ^ (first.input.seedOffset ?? 0)
        restored.media.sourceImage = first.input.sourceImage
        restored.media.keyframes = first.keyframes ?? []
        restored.media.generationReferences = (first.generationReferences ?? []).map(AuthoringReference.canonical)
        restored.media.loras = (first.input.loras ?? []).map {
            .init(path: $0.path, scale: $0.scale, name: $0.name ?? URL(fileURLWithPath: $0.path).lastPathComponent)
        }
        let message = "This older video's duration comes from its saved length; current automatic joins may differ."
        draft = restored
        duration = .init(draft: .init(seconds: try VideoDurationDraft.initialSeconds(frames: frames, fps: Int(fps)),
                                      scenesEnabled: false), reuseNotice: message)
        notice = message
    }
}
