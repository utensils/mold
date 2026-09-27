import Foundation

// Snapping a requested overlap onto a clip recipe's own temporal grid.
// Stays on `RenderDraft`, unlike `addingKeyframe`/`settingExtend`
// (`DraftMedia+Clip.swift`), because it reads the draft's own frame count
// rather than any conditioning field `DraftMedia` owns.
public extension RenderDraft {
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
