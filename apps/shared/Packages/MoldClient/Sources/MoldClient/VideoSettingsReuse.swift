import Foundation

/// A fresh editable generation recipe. Reuse never attaches an old movie as
/// a prefix; Extend is the separate action that authorizes that source.
public struct VideoSettingsReuse: Hashable, Sendable {
    public let draft: RenderDraft
    public let duration: VideoDurationAuthoring?
    public let notice: String?

    public init(metadata: OutputMetadata) throws {
        guard metadata.icLoraControl == nil else {
            throw SequenceDraftError.invalid("This video's original conditioning controls cannot be restored in this editor. Use Start again from its saved job when available, or the CLI with its retained controls.")
        }
        var restored = RenderDraft(reusing: metadata)
        guard let parent = metadata.videoJob else {
            draft = restored; duration = nil; notice = nil
            return
        }
        guard let authoring = parent.authoring, authoring.durationMs > 0,
              authoring.boundaryFrames > 0, authoring.boundaryFrames <= UInt32.max,
              authoring.inputFrames.map({ $0 > 0 && $0 <= UInt32.max }) ?? true else {
            throw SequenceDraftError.invalid("This older video does not record its original duration and scenes. Use Start again from its saved job, or open a new draft and choose its duration explicitly.")
        }
        let scenes = authoring.scenes ?? []
        guard scenes.count <= 4_096,
              scenes.reduce(0, { $0 + $1.prompt.utf8.count }) <= 1_048_576 else {
            throw SequenceDraftError.invalid("This video's saved scenes exceed the editable authoring limit.")
        }
        let editor = VideoDurationDraft(seconds: VideoDurationDraft.seconds(milliseconds: authoring.durationMs),
            scenes: scenes.map { .init(prompt: $0.prompt, seconds: VideoDurationDraft.seconds(milliseconds: $0.durationMs)) },
            scenesEnabled: !scenes.isEmpty)
        // This validates exact integer duration, all scene text and their sum.
        _ = try editor.request(generation: GenerateRequest(prompt: "", model: metadata.model ?? "", width: restored.width,
            height: restored.height, steps: restored.steps, guidance: restored.guidance))
        restored.prompt = metadata.prompt ?? ""
        restored.frames = authoring.inputFrames ?? authoring.boundaryFrames
        restored.prefersAutomaticDuration = false
        if let text = authoring.seed {
            guard !text.isEmpty, text.utf8.allSatisfy({ $0 >= 48 && $0 <= 57 }), let seed = UInt64(text) else {
                throw MoldClientError.malformedResponse
            }
            restored.seed = seed; restored.locksSeed = true
        }
        let reuseNotice = parent.sourceFrames > 0
            ? "Restoring settings for the added footage. The original video is not attached; choose Extend Video to add to it again."
            : nil
        draft = restored; duration = .init(draft: editor, reuseNotice: reuseNotice); notice = reuseNotice
    }
}
