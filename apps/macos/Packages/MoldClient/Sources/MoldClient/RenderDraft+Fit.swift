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
                width = Self.aligned(width, to: alignment, atLeast: resolution.minWidth)
                height = Self.aligned(height, to: alignment, atLeast: resolution.minHeight)
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
        guard let floor, rounded < floor else { return rounded }
        return ((floor + alignment - 1) / alignment) * alignment
    }
}
