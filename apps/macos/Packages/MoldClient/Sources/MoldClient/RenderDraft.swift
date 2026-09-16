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
    /// Clip length, for the families that make one.
    public var frames: Int?
    public var fps: Int?
    /// A still to condition on, already base64-encoded, with the name the host
    /// should record for it.
    public var sourceImage: String?
    public var sourceImageName: String?
    public var strength: Double = 0.75

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
        // Strength only means something with something to apply it to.
        request.strength = sourceImage == nil ? nil : strength
        return request
    }
}

public extension IntegerControl {
    func clamp(_ value: Int) -> Int { Swift.min(Swift.max(value, min), max) }
}

public extension FloatControl {
    func clamp(_ value: Double) -> Double { Swift.min(Swift.max(value, min), max) }
}

public extension RenderDraft {
    /// Rebuilds a draft from a finished print's provenance.
    ///
    /// A sequence's recorded `prompt` is every stage newline-joined, so reuse
    /// takes the FIRST stage rather than restoring a wall of text that was
    /// never one prompt. mold makes the same reduction on its other surfaces.
    init(reusing metadata: OutputMetadata) {
        self.init()
        prompt = Self.firstStage(of: metadata)
        negativePrompt = metadata.negativePrompt ?? ""
        width = metadata.generationWidth ?? metadata.width ?? width
        height = metadata.generationHeight ?? metadata.height ?? height
        steps = metadata.steps ?? steps
        guidance = metadata.guidance ?? guidance
        frames = metadata.frames
        fps = metadata.fps.map { Int($0.rounded()) }
        // The seed is restored but NOT locked: reuse usually means "like that
        // one, but different", and pinning it would make every reuse identical.
        seed = metadata.seed
        locksSeed = false
    }

    private static func firstStage(of metadata: OutputMetadata) -> String {
        let prompt = metadata.prompt ?? ""
        guard metadata.outputMode == "sequence" else { return prompt }
        return prompt.split(separator: "\n", maxSplits: 1).first.map(String.init) ?? prompt
    }
}
