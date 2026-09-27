import Foundation

// `applyClip`, split out of `RenderRequest.swift` purely for size -- same
// reason `applyIdentity`/`applyControl` are their own functions and not inline
// in `one(_:model:maxIdentityPhotos:)`.
extension RenderRequest {
    /// Fills in keyframes, the extend continuation, and the two conditioning
    /// media fields.
    ///
    /// An extend is the strongest claimant (`RenderDraft+Recipe.swift`'s
    /// `adopting`, `RenderDraft+Clip.swift`'s `settingExtend`): keyframes and
    /// `source_video` both go stale to it here too, the same belt
    /// `one(_:model:maxIdentityPhotos:)` applies to `sourceImage` -- two
    /// independent wells can each set their own field with no knowledge of
    /// the other, and only the request builder sees both at once.
    /// `extend_overlap_frames` rides only with `extend_video`
    /// (`validation.rs:1826-1829`) and only when somebody set one (decision
    /// 12, M4 design) -- absence lets the server fill in the family's own
    /// default.
    static func applyClip(_ draft: RenderDraft, to request: inout GenerateRequest) {
        let isExtend = draft.media.extendVideo != nil
        request.keyframes = (!isExtend && !draft.media.keyframes.isEmpty)
            ? draft.media.keyframes : nil
        request.extendVideo = draft.media.extendVideo
        request.extendOverlapFrames = isExtend ? draft.media.extendOverlapFrames : nil
        request.audioFile = draft.media.audioFile
        request.sourceVideo = isExtend ? nil : draft.media.sourceVideo
    }
}
