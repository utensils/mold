import Foundation
import MoldClient

extension NativeSequenceSession {
    struct SavedDraft: Codable {
        struct Clip: Codable {
            let id: UUID
            let prompt: String
            let frames: Int
            let transition: String
            let negativePrompt: String?
            let seedOffset: UInt64?
            let fadeFrames: Int?
            let loras: [ChainLora]?
            init(_ stage: SequenceDraft.Stage) {
                id = stage.id; prompt = stage.prompt; frames = stage.frames; transition = stage.transition
                negativePrompt = stage.input.negativePrompt; seedOffset = stage.input.seedOffset
                fadeFrames = stage.input.fadeFrames; loras = stage.input.loras
            }
            var stage: SequenceDraft.Stage {
                var value = SequenceDraft.Stage(prompt: prompt, frames: frames, transition: transition)
                value.id = id; value.input.negativePrompt = negativePrompt
                value.input.seedOffset = seedOffset; value.input.fadeFrames = fadeFrames; value.input.loras = loras
                return value
            }
        }
        struct Source: Codable {
            let host: UUID
            let origin: String
            let installation: String
            let job: String
            let take: Int
            let filename: String
            var hadReferences: Bool?
            var title: String? = nil

            func matchesAuthority(_ other: Source?) -> Bool {
                guard let other else { return false }
                return host == other.host && origin == other.origin && installation == other.installation
                    && job == other.job && take == other.take && filename == other.filename
            }
        }
        let clips: [Clip]
        let source: Source?
        var reusedMotionTailFrames: Int? = nil
        var settingsHadInputs: Bool? = nil
        var settingsReuseNotice: String? = nil
    }

    func restoreSettings(_ restored: LegacySequenceSettingsReuse) {
        startFresh()
        reusedMotionTailFrames = restored.motionTailFrames
        settingsReuseNotice = restored.notice
        settingsHadInputs = restored.sequence.stages.contains { $0.input.sourceImage != nil
            || !($0.keyframes?.isEmpty ?? true) || !($0.generationReferences?.isEmpty ?? true) }
        settingsInputsNeedRestoration = false
        draft = restored.sequence
        message = restored.notice
    }
    func resetSettingsReuse() {
        reusedMotionTailFrames = nil; settingsInputsNeedRestoration = false
        settingsHadInputs = false; settingsReuseNotice = nil
    }
}

extension NativeSequenceSession {
    /// Saved per-clip values must never silently win over visible global edits.
    func settingsOverrideRefusal(base: GenerateRequest) -> String? {
        guard settingsReuseNotice != nil, let draft else { return nil }
        if let image = base.sourceImage, let saved = draft.stages.first?.input.sourceImage, image != saved {
            return "The first saved clip already has an original image. Clear its saved inputs in Clip options before using the global first frame."
        }
        let adapters = base.loras?.map { ChainLora(path: $0.path, scale: $0.scale, name: $0.name) }
        for stage in draft.stages {
            if let references = base.references, !references.isEmpty,
               let saved = stage.generationReferences, saved != references {
                return "Saved clips have their own references. Choose Use global inputs and controls in their Clip options before replacing them."
            }
            if let adapters, !adapters.isEmpty, let saved = stage.input.loras, adapters != saved {
                return "Saved clips have their own adapters. Choose Use global inputs and controls in their Clip options before replacing them."
            }
            if let negative = base.negativePrompt, let saved = stage.input.negativePrompt, negative != saved {
                return "A saved clip has its own negative prompt. Choose Use global inputs and controls in its Clip options before replacing it."
            }
        }
        return nil
    }

    func useGlobalSettings(for id: UUID) {
        guard !settingsInputsNeedRestoration, var next = draft,
              let index = next.stages.firstIndex(where: { $0.id == id }) else { return }
        var input = ChainStageInput(prompt: next.stages[index].prompt, frames: next.stages[index].frames,
                                    transition: next.stages[index].transition)
        input.fadeFrames = next.stages[index].input.fadeFrames
        next.stages[index].input = input
        settingsHadInputs = next.stages.contains { $0.input.sourceImage != nil
            || !($0.keyframes?.isEmpty ?? true) || !($0.generationReferences?.isEmpty ?? true) }
        draft = next
    }
}
