import Foundation

/// What a recipe's resolution contract will actually accept, as arithmetic.
///
/// Its own namespace rather than more of `RenderDraft`: none of this reads or
/// writes a draft. It is a size and a `ResolutionProfile`, and the answer is
/// the size that profile would admit -- the client half of
/// `validate_resolution` (`generation_profile.rs:1327-1404`). The draft calls
/// it wherever a size arrives from somewhere that cannot vouch for it: a model
/// change, a stored default, a source picture's own shape.
public enum CanvasFit {
    /// Snaps a size onto what `resolution` will actually accept.
    ///
    /// Only meaningful on a size carried over from elsewhere -- a fresh
    /// model's defaults are valid for its own recipe by construction.
    /// `.sourceDriven`, `.none` and `.unknown` are left alone: the size either
    /// comes from somewhere else, or there is no canvas to size.
    public static func fitted(
        _ size: (width: Int, height: Int), to resolution: ResolutionProfile
    ) -> (width: Int, height: Int) {
        switch resolution.domain {
        case .buckets:
            // `warn` switches off the BUCKET-MEMBERSHIP check and nothing
            // else. `validate_resolution` (`generation_profile.rs:1327-1404`)
            // still enforces the minimums, the alignment, `max_pixels`,
            // `max_axis_pixels` and the aspect band for a `Warn` profile --
            // only the last block is gated on `Reject`. Returning early here
            // traded a silent reshape for a guaranteed 422: wan advertises
            // `warn` and its checkpoints carry a real grid (32 on
            // `wan22-ti2v-5b`), so a 1360x768 carried off a FLUX recipe is
            // refused outright where the old snap at least rendered.
            var fitted = size
            if (resolution.offBucket ?? .reject) == .warn {
                fitted = clampedToContract(size, resolution)
                // Aspect cannot be clamped without changing the shape the
                // size is FOR, so a size still outside the band falls back to
                // the ladder -- the only legal answer left.
                if satisfiesAspect(fitted, resolution) { return fitted }
            }
            return snappedToNearestPreset(fitted, resolution)
        case .dynamic:
            return clampedToContract(size, resolution)
        case .sourceDriven, .none, .unknown:
            return size
        }
    }

    /// Everything `validate_resolution` enforces for EVERY domain: the
    /// minimums, the axis ceiling, the pixel budget and the alignment grid.
    /// Shared by the `.dynamic` arm and by a `warn` bucket profile, because
    /// the server applies one rule to both.
    private static func clampedToContract(
        _ size: (width: Int, height: Int), _ resolution: ResolutionProfile
    ) -> (width: Int, height: Int) {
        var width = size.width
        var height = size.height
        if let minWidth = resolution.minWidth { width = Swift.max(width, minWidth) }
        if let minHeight = resolution.minHeight { height = Swift.max(height, minHeight) }
        if let maxAxis = resolution.maxAxisPixels {
            width = Swift.min(width, maxAxis)
            height = Swift.min(height, maxAxis)
        }
        // Total pixel budget trades off both axes together, so it is applied
        // by scaling the pair rather than clamping either alone.
        if let maxPixels = resolution.maxPixels, width * height > maxPixels {
            let scale = (Double(maxPixels) / Double(width * height)).squareRoot()
            width = Swift.max(1, Int(Double(width) * scale))
            height = Swift.max(1, Int(Double(height) * scale))
        }
        guard let alignment = resolution.alignment, alignment > 1 else { return (width, height) }
        let unaligned = (width: width, height: height)
        width = aligned(width, to: alignment, atLeast: resolution.minWidth)
        height = aligned(height, to: alignment, atLeast: resolution.minHeight)
        // Rounding to the NEAREST multiple can grow both axes back past the
        // budget just enforced -- 1788x1006 (1,798,728 under FLUX's
        // 1,800,000) aligns to 1792x1008 = 1,806,336, which
        // `validate_resolution` refuses. A size the app itself fitted must
        // never be rejected at submit, so the budget wins and the alignment
        // falls to the multiple BELOW the scaled pair (not below the
        // already-rounded-up one) (finding 01#8).
        if let maxPixels = resolution.maxPixels, width * height > maxPixels {
            width = alignedDown(unaligned.width, to: alignment, atLeast: resolution.minWidth)
            height = alignedDown(unaligned.height, to: alignment, atLeast: resolution.minHeight)
        }
        return (width, height)
    }

    private static func satisfiesAspect(
        _ size: (width: Int, height: Int), _ resolution: ResolutionProfile
    ) -> Bool {
        guard size.height > 0 else { return false }
        let aspect = Double(size.width) / Double(size.height)
        if let minimum = resolution.minAspectRatio, aspect < minimum { return false }
        if let maximum = resolution.maxAspectRatio, aspect > maximum { return false }
        return true
    }

    private static func snappedToNearestPreset(
        _ size: (width: Int, height: Int), _ resolution: ResolutionProfile
    ) -> (width: Int, height: Int) {
        guard let nearest = resolution.presets.min(by: {
            distanceSquared(from: size, to: $0) < distanceSquared(from: size, to: $1)
        }) else { return size }
        return (nearest.width, nearest.height)
    }

    private static func distanceSquared(
        from size: (width: Int, height: Int), to preset: SizePreset
    ) -> Int {
        let dw = preset.width - size.width
        let dh = preset.height - size.height
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
