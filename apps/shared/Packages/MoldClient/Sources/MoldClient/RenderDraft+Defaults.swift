import Foundation

// The M3 per-model-defaults apply, split out of `RenderDraft+Recipe.swift`
// (which keeps `adopting` and `refusal`) purely for size.
public extension RenderDraft {
    /// Puts a machine's stored per-model defaults (`ModelDefaultsStore`) on
    /// top of a newly adopted recipe's own numbers, field by field, clamped
    /// through the same `IntegerControl`/`FloatControl` `RenderDraft+Recipe.swift`'s
    /// `adopting` uses. A no-op when this is not a new model -- a KEPT draft
    /// (a reuse) is never overwritten by what a machine has on file; the
    /// print's own numbers are the more specific instruction.
    func applying(_ defaults: ModelDefaults, recipe: GenerationRecipe, isNewModel: Bool) -> RenderDraft {
        guard isNewModel else { return self }
        var draft = self
        if let steps = defaults.steps { draft.steps = recipe.steps.clamp(steps) }
        if let guidance = defaults.guidance { draft.guidance = recipe.guidance.clamp(guidance) }
        if defaults.width != nil || defaults.height != nil {
            if let width = defaults.width { draft.width = width }
            if let height = defaults.height { draft.height = height }
            // A default outside the recipe's own bucket list or bounds must
            // not be submitted -- the same fit a carried-over size gets.
            (draft.width, draft.height) = CanvasFit.fitted(
                (draft.width, draft.height), to: recipe.resolution)
        }
        if let negativePrompt = defaults.negativePrompt,
           recipe.capabilities.negativePrompt?.isAvailable == true {
            draft.negativePrompt = negativePrompt
        }
        return draft
    }
}
