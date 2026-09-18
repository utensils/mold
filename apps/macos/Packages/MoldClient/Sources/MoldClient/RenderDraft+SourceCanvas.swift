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
        let exact = CanvasFit.sourceExact(source, resolution: recipe.resolution)
        let automatic = CanvasFit.automatic(source, recipe: recipe) ?? exact
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
}
