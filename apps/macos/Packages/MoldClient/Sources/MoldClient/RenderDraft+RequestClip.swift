import Foundation

// `applyClip`, split out of `RenderDraft+Request.swift` purely for size --
// same reason `applyIdentity`/`applyControl` live there and not inline in
// `request(model:maxIdentityPhotos:)`.
extension RenderDraft {
    /// Fills in keyframes, the extend continuation, and the two conditioning
    /// media fields.
    ///
    /// An extend is the strongest claimant (`RenderDraft+Recipe.swift`'s
    /// `adopting`, `RenderDraft+Clip.swift`'s `settingExtend`): keyframes and
    /// `source_video` both go stale to it here too, the same belt
    /// `request(model:maxIdentityPhotos:)` applies to `sourceImage` -- two
    /// independent wells can each set their own field with no knowledge of
    /// the other, and only the request builder sees both at once.
    /// `extend_overlap_frames` rides only with `extend_video`
    /// (`validation.rs:1826-1829`) and only when somebody set one (decision
    /// 12, M4 design) -- absence lets the server fill in the family's own
    /// default.
    func applyClip(to request: inout GenerateRequest) {
        let isExtend = media.extendVideo != nil
        request.keyframes = (!isExtend && !media.keyframes.isEmpty) ? media.keyframes : nil
        request.extendVideo = media.extendVideo
        request.extendOverlapFrames = isExtend ? media.extendOverlapFrames : nil
        request.audioFile = media.audioFile
        request.sourceVideo = isExtend ? nil : media.sourceVideo
    }
}
