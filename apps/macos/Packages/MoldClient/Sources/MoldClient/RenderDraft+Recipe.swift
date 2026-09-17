import Foundation

// Reconciling a draft against the recipe that will run it. Split from the
// draft's own shape purely for size.
public extension RenderDraft {
    /// Adopts a recipe: takes its defaults for anything the previous model
    /// can't vouch for, and clamps what it can keep.
    ///
    /// Switching models is not a reason to lose a prompt, but it IS a reason
    /// to stop asking for 50 steps from a recipe whose maximum is 8.
    ///
    /// The parking reconciliation below (source image, edit images, mask,
    /// identity, adapters) runs unconditionally, whether `isNewModel` is
    /// `true` (a fresh model) or `false` (a recipe switch on the SAME model,
    /// e.g. one LTX-2 pipeline to another) -- either way the question is the
    /// same one: can the recipe about to run read what the draft is holding.
    public func adopting(_ recipe: GenerationRecipe, isNewModel: Bool) -> RenderDraft {
        var draft = self
        if isNewModel {
            draft.width = recipe.defaults.width
            draft.height = recipe.defaults.height
            draft.steps = recipe.defaults.steps
            draft.guidance = recipe.defaults.guidance
            draft.frames = recipe.temporal?.frames.default
            draft.fps = recipe.temporal?.fps.value
        } else {
            draft.steps = recipe.steps.clamp(draft.steps)
            draft.guidance = recipe.guidance.clamp(draft.guidance)
            // A KEPT draft is the one path that carries a size across models
            // with a different resolution contract -- reuse onto a recipe
            // with its own bucket list, or a smaller ceiling, must not submit
            // a size the new host is going to refuse.
            draft.fit(to: recipe.resolution)
        }
        // A fixed control has exactly one correct value, whatever was there.
        if recipe.steps.mode == .fixed { draft.steps = recipe.steps.default }
        if recipe.guidance.mode == .fixed { draft.guidance = recipe.guidance.default }
        if recipe.capabilities.promptRequirement == .ignored { draft.prompt = "" }

        // A still model has no clip length; a clip model's length must sit on
        // the grid its family accepts.
        if let temporal = recipe.temporal {
            draft.frames = temporal.snap(draft.frames ?? temporal.frames.default)
            if !temporal.fps.isAdjustable { draft.fps = temporal.fps.value }
            if draft.fps == nil { draft.fps = temporal.fps.value }
        } else {
            draft.frames = nil
            draft.fps = nil
        }

        // Conditioning the recipe cannot currently take is PARKED rather than
        // dropped, so it comes back if the next model can read it again
        // (`RenderDraft+Park.swift`; decision 4 in the M4 design). Edit
        // images are reconciled first so the exclusive/replaces check below
        // reads the post-truncation list, matching the order this logic ran
        // in before parking existed.
        let references = recipe.capabilities.referenceImages
        let referencesVisible = references?.mode.isVisible == true
        draft.reconcileEditImages(supported: referencesVisible, maxCount: references?.maxCount)
        if !referencesVisible { draft.referenceWeight = nil }

        // `exclusive`/`replaces` mean ONE render carries a source image OR
        // references, never both. Keeping whichever was added last would be
        // guessing, so references win -- they are the more specific
        // instruction. `readsSourceImage` is the CORRECTED reading of an
        // absent `sourceImage` block: absence means the recipe reads one
        // (fact 1 in the M4 design, `manifest.rs:265-270`), not that there is
        // no source path -- the raw `sourceImage?.isSupported` this block
        // used to read got that backwards for every still model in the fleet.
        let takenByReferences = !draft.editImages.isEmpty
            && (references?.sourceRelation == .exclusive || references?.sourceRelation == .replaces)
        draft.reconcileSourceImage(supported: recipe.capabilities.readsSourceImage && !takenByReferences)

        // The mask needs BOTH the recipe's own permission and a surviving
        // source image -- an orphaned mask over no source is meaningless
        // (`validation.rs:3101-3107`).
        draft.reconcileMask(supported: recipe.capabilities.acceptsMask && draft.sourceImage != nil)

        // Identity is positive-only: `supportsIdentity != true` means the
        // well is not drawn and the staged photo is held, because sending it
        // to an unqualified checkpoint is a refusal, not a silent ignore
        // (`identity.rs:1011-1013`).
        draft.reconcileIdentity(supported: recipe.capabilities.supportsIdentity == true)

        // ControlNet: `controlNet` is nil for a `hidden` block and for no
        // block at all (fact 3 in the M4 design) -- there is a real recipe
        // gate, so this never asks whether an adapter happens to be
        // installed, only whether THIS recipe would read one.
        draft.reconcileControl(supported: recipe.capabilities.controlNet != nil)

        draft.reconcileLoras(
            supported: recipe.capabilities.loraStack != nil,
            maxCount: recipe.capabilities.loraStack?.maxCount
        )

        if recipe.capabilities.negativePrompt?.isAvailable != true {
            draft.negativePrompt = ""
        }
        return draft
    }

    /// Puts a machine's stored per-model defaults (`ModelDefaultsStore`) on
    /// top of a newly adopted recipe's own numbers, field by field, clamped
    /// through the same `IntegerControl`/`FloatControl` this file's
    /// `adopting` uses. A no-op when this is not a new model -- a KEPT draft
    /// (a reuse) is never overwritten by what a machine has on file; the
    /// print's own numbers are the more specific instruction.
    public func applying(_ defaults: ModelDefaults, recipe: GenerationRecipe, isNewModel: Bool) -> RenderDraft {
        guard isNewModel else { return self }
        var draft = self
        if let steps = defaults.steps { draft.steps = recipe.steps.clamp(steps) }
        if let guidance = defaults.guidance { draft.guidance = recipe.guidance.clamp(guidance) }
        if defaults.width != nil || defaults.height != nil {
            if let width = defaults.width { draft.width = width }
            if let height = defaults.height { draft.height = height }
            // A default outside the recipe's own bucket list or bounds must
            // not be submitted -- the same fit a carried-over size gets.
            draft.fit(to: recipe.resolution)
        }
        if let negativePrompt = defaults.negativePrompt,
           recipe.capabilities.negativePrompt?.isAvailable == true {
            draft.negativePrompt = negativePrompt
        }
        return draft
    }

    /// Whether this draft can be submitted against the recipe, and why not.
    public func refusal(for recipe: GenerationRecipe) -> String? {
        if recipe.capabilities.promptRequirement == .required,
           prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return "Describe what you want first."
        }
        return nil
    }
}
