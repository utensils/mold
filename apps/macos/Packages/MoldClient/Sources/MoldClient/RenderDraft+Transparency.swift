import Foundation

// The transparent-background toggle on a draft. Port of the
// `coerceFormatForTransparency` / `transparencyRequestFields` pair in
// `studio/lib/transparency.ts`.
//
// The person's choice lives in `transparentBackground` and is PARKED, never
// dropped, on a recipe that cannot honour it: `adopting` records the recipe's
// `transparencyControl`, and only the pair of them reaches the wire. An
// ordinary render is therefore byte-identical to one from a client that
// predates the field.
public extension RenderDraft {
    /// Whether a request built now carries `transparent_background: true`.
    var transparencyActive: Bool { transparentBackground && transparency != nil }

    /// Turns the toggle on or off. On moves a format with no alpha channel
    /// (JPEG) to the recipe's first alpha format -- with no pick made, the
    /// recipe's own default decides whether one is needed. Off leaves the
    /// format alone.
    func settingTransparentBackground(_ on: Bool, output: OutputCapabilities?) -> RenderDraft {
        var draft = self
        draft.transparentBackground = on
        draft.coerceFormatForTransparency(output: output)
        return draft
    }

    /// Whether the Format picker must refuse `format` right now.
    func transparencyBlocksFormat(_ format: String) -> Bool {
        guard transparencyActive, let transparency else { return false }
        return !transparency.formats.contains(format)
    }

    /// Moves the chosen (or, unchosen, the recipe's default) format onto an
    /// alpha container while the toggle is active. Called on every toggle
    /// and every adopt, so a restored JPEG draft cannot submit a pair
    /// admission refuses.
    internal mutating func coerceFormatForTransparency(output: OutputCapabilities?) {
        guard transparencyActive, let transparency,
              let first = transparency.formats.first else { return }
        let effective = outputFormat ?? output?.defaultFormat
        guard let effective, !transparency.formats.contains(effective) else { return }
        outputFormat = first
    }
}
