import Foundation

// Keyframes and an extend continuation, within ONE recipe -- unlike
// `RenderDraft+Park.swift`/`+ParkClip.swift`, which reconcile a recipe
// SWITCH, these are the pure functions the Clip group's own controls call
// as a person edits, so the two never ride together on the wire
// (`validation.rs:1851-1853`) and an extend never rides with the source
// image (`validation.rs:1845-1849`).
public extension RenderDraft {
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

    /// Snaps a requested overlap onto `temporal`'s own grid (`step·k+1`,
    /// `validation.rs:1855-1875`) and keeps it strictly below this draft's
    /// own frame count (`validation.rs:1876-1880`) -- applied where the
    /// Overlap field writes, so an out-of-grid value never reaches the wire.
    func snappedOverlap(_ requested: Int, temporal: TemporalProfile) -> Int {
        let step = Swift.max(temporal.frames.step, 1)
        let ceiling = Swift.max((frames ?? temporal.frames.default) - 1, 1)
        let kMax = Swift.max((ceiling - 1) / step, 0)
        let wanted = Swift.max(requested - 1, 0)
        let kRounded = (wanted + step / 2) / step
        let k = Swift.min(Swift.max(kRounded, 0), kMax)
        return k * step + 1
    }
}
