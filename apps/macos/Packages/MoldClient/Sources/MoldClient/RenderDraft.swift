import Foundation

/// What the Generate pane is holding before anything is submitted.
///
/// The draft is always reconciled against the chosen model's recipe: the
/// server owns what each control may be, and a value carried over from a
/// different model is only kept when the new recipe would accept it.
public struct RenderDraft: Hashable, Sendable {
    public var prompt: String = ""
    public var negativePrompt: String = ""
    public var width: Int = 1024
    public var height: Int = 1024
    public var steps: Int = 20
    public var guidance: Double = 3.5
    public var batchSize: Int = 1
    /// nil means "let the host pick", which is the default and what makes
    /// repeated renders differ.
    public var seed: UInt64?
    public var locksSeed: Bool = false

    public init() {}

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
        } else {
            draft.steps = recipe.steps.clamp(draft.steps)
            draft.guidance = recipe.guidance.clamp(draft.guidance)
        }
        // A fixed control has exactly one correct value, whatever was there.
        if recipe.steps.mode == .fixed { draft.steps = recipe.steps.default }
        if recipe.guidance.mode == .fixed { draft.guidance = recipe.guidance.default }
        if recipe.capabilities.promptRequirement == .ignored { draft.prompt = "" }
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
        GenerateRequest(
            prompt: prompt, model: model, width: width, height: height,
            steps: steps, guidance: guidance, batchSize: batchSize,
            negativePrompt: negativePrompt.isEmpty ? nil : negativePrompt,
            seed: locksSeed ? seed : nil
        )
    }
}

public extension IntegerControl {
    func clamp(_ value: Int) -> Int { Swift.min(Swift.max(value, min), max) }
}

public extension FloatControl {
    func clamp(_ value: Double) -> Double { Swift.min(Swift.max(value, min), max) }
}
