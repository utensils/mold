import Foundation

// The canvas a newly attached source picture moves the draft to, and the
// intent that decides whether it moves at all.
public extension RenderDraft {
    /// Attaches a source picture's SHAPE.
    ///
    /// The decision is the recorded `canvasIntent` and nothing else -- never
    /// "is the canvas still the value I last computed", which no surface can
    /// honour because choosing a model writes that model's defaults BEFORE
    /// any source watcher runs (#1166, `sourceResolution.ts:50-59`).
    ///
    /// `replaced` is studio's own predicate -- the attached bytes are not the
    /// bytes that were there (`CreatePage.vue:1151`), which a FIRST picture
    /// satisfies too. Putting a picture in re-arms the automatic choice even
    /// over a canvas somebody chose, because attaching a subject is an
    /// instruction about the shape as much as about the pixels, and the
    /// intent is re-recorded to say so (`CreatePage.vue:1174-1180`).
    ///
    /// `preserveReplacement` is the exception that PINS it manual instead: a
    /// programmatic Reuse or edit import is not a fresh drag, and the canvas
    /// it restored is the one that was authored.
    mutating func attachSourceShape(
        _ source: (width: Int, height: Int), recipe: GenerationRecipe?,
        replaced: Bool, preserveReplacement: Bool = false
    ) {
        defer { recordIntent(replaced: replaced, preserveReplacement: preserveReplacement) }
        guard let recipe, recipe.resolution.hasCanvas,
              recipe.resolution.domain != .sourceDriven else { return }
        let exact = Self.sourceExactCanvas(source, resolution: recipe.resolution)
        let automatic = Self.automaticCanvas(source, recipe: recipe) ?? exact
        guard let canvas = canvasIntent.canvas(
            sourceExact: exact, automatic: automatic, replaced: replaced,
            preserveReplacement: preserveReplacement) else { return }
        width = canvas.width
        height = canvas.height
    }

    /// The intent is re-recorded AFTER the canvas moves, because the canvas
    /// that moves is decided by the intent that was in force
    /// (`CreatePage.vue:1174-1180`).
    private mutating func recordIntent(replaced: Bool, preserveReplacement: Bool) {
        guard replaced else { return }
        if preserveReplacement {
            canvasIntent = .manual
        } else if canvasIntent != .sourceExact {
            canvasIntent = .source
        }
    }

    /// The source at its own size, snapped onto what the recipe accepts --
    /// `fit(to:)` is already that rule, and there must not be a second copy.
    static func sourceExactCanvas(
        _ source: (width: Int, height: Int), resolution: ResolutionProfile
    ) -> (width: Int, height: Int) {
        var draft = RenderDraft()
        draft.width = Swift.max(source.width, 1)
        draft.height = Swift.max(source.height, 1)
        // `.buckets` would snap to the ladder, which is `automaticCanvas`'s
        // job; `source-exact` means the source's own aligned, capped size, so
        // it is measured against the CONTRACT rather than the preset list.
        draft.fit(to: resolution.domain == .buckets
            ? ResolutionProfile(
                domain: .dynamic, alignment: resolution.alignment,
                minWidth: resolution.minWidth, minHeight: resolution.minHeight,
                maxPixels: resolution.maxPixels, maxAxisPixels: resolution.maxAxisPixels,
                minAspectRatio: nil, maxAspectRatio: nil, offBucket: nil, aspectGroups: nil)
            : resolution)
        return (draft.width, draft.height)
    }

    /// The model-authored canvas nearest the source's shape. Port of
    /// `resolveDefaultSourceResolution` (`sourceResolution.ts:83-130`): aspect
    /// distance measured LOGARITHMICALLY so portrait and landscape are treated
    /// symmetrically, and once the closest aspect is known, the tier nearest
    /// the recipe's own default pixel area. `nil` where the recipe advertises
    /// no presets at all.
    static func automaticCanvas(
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
