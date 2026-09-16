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
    /// Ordered reference images, base64. For a recipe whose first image is the
    /// Target, index 0 is that one.
    public var editImages: [String] = []
    public var referenceWeight: Double?

    public init() {}
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
