import Foundation

// The two canvases a source picture's own shape offers: the source at its own
// size, and the model-authored tier nearest it. Which of them a draft actually
// takes is `RenderDraft.attachSourceShape`'s decision, from the recorded
// `CanvasIntent`.
public extension CanvasFit {
    /// The source at its own size, snapped onto what the recipe accepts --
    /// `fitted(_:to:)` is already that rule, and there must not be a second
    /// copy.
    static func sourceExact(
        _ source: (width: Int, height: Int), resolution: ResolutionProfile
    ) -> (width: Int, height: Int) {
        let size = (width: Swift.max(source.width, 1), height: Swift.max(source.height, 1))
        // `.buckets` would snap to the ladder, which is `automatic`'s job;
        // source-exact means the source's own aligned, capped size, so it is
        // measured against the CONTRACT rather than the preset list.
        return fitted(size, to: resolution.domain == .buckets
            ? ResolutionProfile(
                domain: .dynamic, alignment: resolution.alignment,
                minWidth: resolution.minWidth, minHeight: resolution.minHeight,
                maxPixels: resolution.maxPixels, maxAxisPixels: resolution.maxAxisPixels,
                minAspectRatio: nil, maxAspectRatio: nil, offBucket: nil, aspectGroups: nil)
            : resolution)
    }

    /// The model-authored canvas nearest the source's shape. Port of
    /// `resolveDefaultSourceResolution` (`sourceResolution.ts:83-130`): aspect
    /// distance measured LOGARITHMICALLY so portrait and landscape are treated
    /// symmetrically, and once the closest aspect is known, the tier nearest
    /// the recipe's own default pixel area. `nil` where the recipe advertises
    /// no presets at all.
    static func automatic(
        _ source: (width: Int, height: Int), recipe: GenerationRecipe
    ) -> (width: Int, height: Int)? {
        let presets = recipe.resolution.presets.filter { $0.width > 0 && $0.height > 0 }
        guard !presets.isEmpty, source.width > 0, source.height > 0 else { return nil }
        let sourceRatio = Double(source.width) / Double(source.height)
        let defaultArea = Double(recipe.defaults.width * recipe.defaults.height)
        let ranked = presets.min { left, right in
            let leftRatio = abs(log(Double(left.width) / Double(left.height) / sourceRatio))
            let rightRatio = abs(log(Double(right.width) / Double(right.height) / sourceRatio))
            if abs(leftRatio - rightRatio) > .ulpOfOne { return leftRatio < rightRatio }
            guard defaultArea > 0 else {
                return left.width * left.height > right.width * right.height
            }
            return abs(Double(left.width * left.height) - defaultArea)
                < abs(Double(right.width * right.height) - defaultArea)
        }
        return ranked.map { ($0.width, $0.height) }
    }
}
