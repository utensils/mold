import Foundation

/// LTX-2 video-only (#1037) -- the one client-side policy for when the opt-in
/// may ride a request, ported verbatim from `studio/lib/videoOnly.ts` so
/// neither surface invents its own conflict table. The server
/// (`mold_core::validation`) stays the authority; this exists so a blocked
/// toggle explains itself inline instead of surfacing as a 422.
///
/// Skipping the audio branch is output-changing for the VIDEO (the branch
/// feeds the video stream through the a2v cross-attention), which is why the
/// control is labelled that way and is never a default.
public enum VideoOnlyPolicy {
    /// The four conflicts `video_only` can hit, mirroring
    /// `VideoOnlyConflictInputs` in `studio/lib/videoOnly.ts`.
    public struct Inputs: Hashable, Sendable {
        /// The draft's explicit audio-output opt-in (`enable_audio == true`).
        public var audioEnabled: Bool
        /// `pipeline == "t2a"` -- there is no video to keep.
        public var audioOnlyPipeline: Bool
        /// An attached or server-path conditioning audio file.
        public var hasConditioningAudio: Bool
        /// A staged continuation (`extend_video`/`extend_video_path`).
        public var isExtend: Bool

        public init(
            audioEnabled: Bool = false, audioOnlyPipeline: Bool = false,
            hasConditioningAudio: Bool = false, isExtend: Bool = false
        ) {
            self.audioEnabled = audioEnabled
            self.audioOnlyPipeline = audioOnlyPipeline
            self.hasConditioningAudio = hasConditioningAudio
            self.isExtend = isExtend
        }
    }

    /// Why `video_only` cannot ride the current form, or `nil` when it can.
    /// The order matches `videoOnly.ts`'s: the pipeline explanation wins over
    /// every other conflict, because it is the one nothing else can fix.
    public static func blockedReason(_ inputs: Inputs) -> String? {
        if inputs.audioOnlyPipeline {
            return "Text-to-audio renders sound only; video-only does not apply."
        }
        if inputs.audioEnabled {
            return "Turn off Generate audio first — video-only skips the branch that renders it."
        }
        if inputs.hasConditioningAudio {
            return "Remove the conditioning audio first — video-only skips the branch it drives."
        }
        if inputs.isExtend {
            return "A continuation keeps its source clip's rendering path."
        }
        return nil
    }

    /// The wire value: `true` only for an enabled, conflict-free opt-in;
    /// `nil` otherwise so the field stays absent and the server's default
    /// multimodal path remains authoritative.
    public static func requestValue(enabled: Bool, _ inputs: Inputs) -> Bool? {
        enabled && blockedReason(inputs) == nil ? true : nil
    }
}
