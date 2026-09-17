import Foundation

// Snapping a carried-over size onto a recipe's resolution contract. Split
// from `RenderDraft+Recipe` purely for size -- `adopting` calls straight into
// `fit(to:)` for a KEPT draft, which is the one path that carries a size
// across models with a different contract.
public extension RenderDraft {
    /// Snaps `width`/`height` onto what `resolution` will actually accept.
    ///
    /// Only meaningful on a size carried over from elsewhere -- a fresh
    /// model's defaults are valid for its own recipe by construction. On
    /// `.buckets`, the nearest advertised preset wins; on `.dynamic`, the
    /// size is clamped to the advertised bounds and rounded to the grid.
    /// `.sourceDriven`, `.none` and `.unknown` are left alone: the size
    /// either comes from somewhere else, or there is no canvas to size.
    mutating func fit(to resolution: ResolutionProfile) {
        switch resolution.domain {
        case .buckets:
            // `warn` means the host ADMITS an off-ladder size and says so
            // (`validation.rs:1366-1367` refuses one only on `reject`), so
            // snapping it would silently re-render a reused print at a
            // different shape. Wan is the one family that advertises it;
            // absence still means `reject`, which is the safe reading.
            guard (resolution.offBucket ?? .reject) != .warn else { return }
            guard let nearest = resolution.presets.min(by: {
                distanceSquared(to: $0) < distanceSquared(to: $1)
            }) else { return }
            width = nearest.width
            height = nearest.height
        case .dynamic:
            if let minWidth = resolution.minWidth { width = Swift.max(width, minWidth) }
            if let minHeight = resolution.minHeight { height = Swift.max(height, minHeight) }
            if let maxAxis = resolution.maxAxisPixels {
                width = Swift.min(width, maxAxis)
                height = Swift.min(height, maxAxis)
            }
            // Total pixel budget trades off both axes together, so it is
            // applied by scaling the pair rather than clamping either alone.
            if let maxPixels = resolution.maxPixels, width * height > maxPixels {
                let scale = (Double(maxPixels) / Double(width * height)).squareRoot()
                width = Swift.max(1, Int(Double(width) * scale))
                height = Swift.max(1, Int(Double(height) * scale))
            }
            if let alignment = resolution.alignment, alignment > 1 {
                let unaligned = (width: width, height: height)
                width = Self.aligned(width, to: alignment, atLeast: resolution.minWidth)
                height = Self.aligned(height, to: alignment, atLeast: resolution.minHeight)
                // Rounding to the NEAREST multiple can grow both axes back
                // past the budget just enforced -- 1788x1006 (1,798,728 under
                // FLUX's 1,800,000) aligns to 1792x1008 = 1,806,336, which
                // `validate_resolution` refuses. A size the app itself fitted
                // must never be rejected at submit, so the budget wins and
                // the alignment falls to the multiple BELOW the scaled pair
                // (not below the already-rounded-up one) (finding 01#8).
                if let maxPixels = resolution.maxPixels, width * height > maxPixels {
                    width = Self.alignedDown(unaligned.width, to: alignment,
                                             atLeast: resolution.minWidth)
                    height = Self.alignedDown(unaligned.height, to: alignment,
                                              atLeast: resolution.minHeight)
                }
            }
        case .sourceDriven, .none, .unknown:
            break
        }
    }

    private func distanceSquared(to preset: SizePreset) -> Int {
        let dw = preset.width - width
        let dh = preset.height - height
        return dw * dw + dh * dh
    }

    /// Rounds to the nearest multiple of `alignment`, never below `floor` --
    /// rounding down at the floor would hand back a size smaller than the
    /// recipe's own minimum.
    private static func aligned(_ value: Int, to alignment: Int, atLeast floor: Int?) -> Int {
        let rounded = Swift.max(alignment, Int((Double(value) / Double(alignment)).rounded()) * alignment)
        return raised(rounded, to: alignment, atLeast: floor)
    }

    /// The multiple at or BELOW `value` -- what a pixel budget needs, since
    /// rounding to the nearest can grow past it. The recipe's own minimum
    /// still wins: a size under it is refused whatever the budget says.
    private static func alignedDown(_ value: Int, to alignment: Int, atLeast floor: Int?) -> Int {
        let rounded = Swift.max(alignment, (value / alignment) * alignment)
        return raised(rounded, to: alignment, atLeast: floor)
    }

    private static func raised(_ value: Int, to alignment: Int, atLeast floor: Int?) -> Int {
        guard let floor, value < floor else { return value }
        return ((floor + alignment - 1) / alignment) * alignment
    }
}
