import Foundation

// Keyframes and an extend continuation, within ONE recipe -- unlike
// `DraftMedia+Park.swift`/`+ParkClip.swift`, which reconcile a recipe
// SWITCH, these are the pure functions the Clip group's own controls call
// as a person edits, so the two never ride together on the wire
// (`validation.rs:1851-1853`) and an extend never rides with the source
// image (`validation.rs:1845-1849`).
public extension DraftMedia {
    /// Adds a keyframe, parking any staged extend first -- "the one just
    /// set wins" (S6b report). Also brings back any keyframes an EARLIER
    /// extend parked: adding one is how a person switches back into
    /// keyframe mode, and those rows must not stay stranded in `parked`
    /// with nothing left to surface them until the next recipe switch.
    /// Rows stay sorted by frame, the order `KeyframeTable` draws them in.
    mutating func addingKeyframe(_ keyframe: KeyframeCondition) {
        if extendVideo != nil {
            parked.extendVideo = extendVideo
            parked.extendVideoName = extendVideoName
            parked.extendOverlapFrames = extendOverlapFrames
            extendVideo = nil
            extendVideoName = nil
            extendOverlapFrames = nil
        }
        if !parked.keyframes.isEmpty {
            keyframes = parked.keyframes
            parked.keyframes = []
        }
        keyframes.append(keyframe)
        keyframes.sort { $0.frame < $1.frame }
    }

    /// Stages an extend continuation, parking any keyframes AND the source
    /// image first -- an extend's first frames are pinned by the source
    /// clip's own tail, so neither may ride with it.
    mutating func settingExtend(video: String, name: String) {
        if !keyframes.isEmpty {
            parked.keyframes = keyframes
            keyframes = []
        }
        if sourceImage != nil {
            parked.sourceImage = sourceImage
            parked.sourceImageName = sourceImageName
            sourceImage = nil
            sourceImageName = nil
        }
        extendVideo = video
        extendVideoName = name
    }
}
