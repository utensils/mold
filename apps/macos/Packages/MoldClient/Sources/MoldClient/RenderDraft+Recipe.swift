import Foundation

// Reconciling a draft against the recipe that will run it. Split from the
// draft's own shape purely for size.
public extension RenderDraft {
    /// Adopts a recipe: takes its defaults for anything the previous model
    /// can't vouch for, and clamps what it can keep.
    ///
    /// Switching models is not a reason to lose a prompt, but it IS a reason
    /// to stop asking for 50 steps from a recipe whose maximum is 8.
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

        // Carrying a source image to a recipe that cannot read one would send
        // bytes the host must refuse.
        if recipe.capabilities.sourceImage?.isSupported != true {
            draft.sourceImage = nil
            draft.sourceImageName = nil
        }

        let references = recipe.capabilities.referenceImages
        if references?.mode.isVisible != true {
            draft.editImages = []
            draft.referenceWeight = nil
        } else if let max = references?.maxCount, draft.editImages.count > max {
            draft.editImages = Array(draft.editImages.prefix(max))
        }
        // `exclusive` means ONE render carries a source image or references,
        // never both. Keeping whichever was added last would be guessing, so
        // references win -- they are the more specific instruction.
        if references?.sourceRelation == .exclusive, !draft.editImages.isEmpty {
            draft.sourceImage = nil
            draft.sourceImageName = nil
        }
        // `replaces` means the references ARE the conditioning: no source, and
        // no strength to apply.
        if references?.sourceRelation == .replaces, !draft.editImages.isEmpty {
            draft.sourceImage = nil
            draft.sourceImageName = nil
        }
        if recipe.capabilities.negativePrompt?.isAvailable != true {
            draft.negativePrompt = ""
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

    public func request(model: String) -> GenerateRequest {
        var request = GenerateRequest(
            prompt: prompt, model: model, width: width, height: height,
            steps: steps, guidance: guidance, batchSize: batchSize,
            negativePrompt: negativePrompt.isEmpty ? nil : negativePrompt,
            seed: locksSeed ? seed : nil
        )
        request.frames = frames
        request.fps = fps
        request.sourceImage = sourceImage
        request.sourceImageName = sourceImageName
        request.editImages = editImages.isEmpty ? nil : editImages
        request.referenceWeight = editImages.isEmpty ? nil : referenceWeight
        // Strength only means something with something to apply it to.
        request.strength = sourceImage == nil ? nil : strength
        return request
    }
}
