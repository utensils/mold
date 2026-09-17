import Foundation

// Reconciling a draft against the recipe that will run it. Split from the
// draft's own shape purely for size.
public extension RenderDraft {
    /// `adopting`, given the `Model` the recipe came from: its family, its
    /// name and its whole profile in one argument, so a caller that HAS the
    /// model does not spell three out.
    func adopting(_ recipe: GenerationRecipe, isNewModel: Bool, for model: Model) -> RenderDraft {
        adopting(recipe, isNewModel: isNewModel, family: model.family,
                 model: model.name, profile: model.generationProfile)
    }

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
    ///
    /// `family` and `model` are read ONLY by the legacy reference rule, for a
    /// host that advertises no `reference_images` block at all
    /// (`DraftMedia.reconcile(for:family:model:)`).
    ///
    /// `profile` is the model's WHOLE recipe set, read only by
    /// `AdvancedControlsOffered` -- LTX-2's guidance overrides belong to the
    /// model, not to the one recipe in hand, so a switch between its pipelines
    /// must not park them. Defaults to nil, which reads as "no profile" and
    /// offers no guidance controls.
    public func adopting(
        _ recipe: GenerationRecipe, isNewModel: Bool,
        family: String? = nil, model: String? = nil,
        profile: GenerationProfileSet? = nil
    ) -> RenderDraft {
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
            // The RATE first: the requestable ceiling moves with it
            // (`TemporalProfile.lengthBounds`), so a length settled before it
            // could be one admission narrows away.
            if !temporal.fps.isAdjustable { draft.fps = temporal.fps.value }
            if draft.fps == nil { draft.fps = temporal.fps.value }
            let bounds = temporal.lengthBounds(
                fps: draft.fps ?? temporal.fps.value, family: family, model: model,
                sourceImage: recipe.capabilities.sourceImage)
            let snapped = temporal.snap(draft.frames ?? temporal.frames.default)
            draft.frames = Swift.min(Swift.max(snapped, bounds.min), bounds.max)
        } else {
            draft.frames = nil
            draft.fps = nil
        }

        // Conditioning the recipe cannot currently take is PARKED rather than
        // dropped, so it comes back if the next model can read it again
        // (`DraftMedia+Reconcile.swift`; decision 4 in the M4 design).
        draft.media.reconcile(for: recipe.capabilities, family: family, model: model)

        // The sampler controls take the same rescue: a solver, a flow shift or
        // an STG scale the new recipe does not advertise is PARKED, so
        // stepping onto a model that cannot take it and back hands it over
        // rather than quietly resetting it (`AdvancedControls+Park.swift`).
        draft.advanced.reconcile(with: AdvancedControlsOffered.resolve(
            recipe: recipe, in: profile, family: family))

        if recipe.capabilities.negativePrompt?.isAvailable != true {
            draft.negativePrompt = ""
        }

        // Echoed straight through on every adopt. `nil` on `auto`.
        draft.pipeline = recipe.requestSelector?.pipeline

        // A format picked against a different recipe must not survive onto
        // one that cannot deliver it -- `t2a`'s `formats` is `["wav"]` alone,
        // and a carried-over "mp4" would 422 rather than fall back.
        if let output = recipe.capabilities.output, let format = draft.outputFormat,
           !output.formats.contains(format) {
            draft.outputFormat = nil
        }

        // `enable_audio`/`video_only` are LTX-2-only (`validation.rs:3255`,
        // `:3258`); carried onto a family that cannot generate an audio
        // branch, either would arm a mismatch 422.
        if recipe.capabilities.supportsAudio != true {
            draft.enableAudio = false
            draft.videoOnly = false
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
