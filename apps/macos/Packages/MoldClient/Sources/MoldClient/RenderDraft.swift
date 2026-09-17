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
    /// `didSet` clamps a staged identity's `startStep` below the new count --
    /// the identity and steps controls live in different places on screen,
    /// so dragging Steps down after Start step was set must not silently
    /// arm a 422 (`identity.rs:560-566`). `applyIdentity` clamps again at
    /// request time as a belt; this is what keeps the ON-SCREEN bound in
    /// sync as it happens rather than only at submit.
    public var steps: Int = 20 {
        didSet {
            guard var conditioning = media.identity else { return }
            let range = Identity.startStepRange(steps: steps)
            let clamped = Swift.min(Swift.max(conditioning.startStep, range.lowerBound), range.upperBound)
            guard clamped != conditioning.startStep else { return }
            conditioning.startStep = clamped
            media.identity = conditioning
        }
    }
    public var guidance: Double = 3.5
    public var batchSize: Int = 1
    /// nil means "let the host pick", which is the default and what makes
    /// repeated renders differ.
    public var seed: UInt64?
    public var locksSeed: Bool = false
    /// Clip length, for the families that make one.
    public var frames: Int?
    public var fps: Int?
    /// LTX-2's chosen way of running, echoed straight from the adopted
    /// recipe's own `request_selector.pipeline` -- `nil` on `auto`, which
    /// means "let the server pick" and must never be spelled as the string
    /// `"auto"` (`RenderDraft+Recipe.swift`'s `adopting`).
    public var pipeline: String?
    /// The opt-in for LTX-2's audio branch. Sent only when `true`
    /// (`RenderDraft+Request.swift`) -- a `false` still reaches the wire as
    /// absence, because an explicit `false` conflicts with an audio-only
    /// pipeline (`validation.rs:3555`).
    public var enableAudio: Bool = false
    /// Skips the audio branch on a video render. Never sent while
    /// `VideoOnlyPolicy` finds a conflict -- see `RenderDraft.videoOnlyInputs`.
    public var videoOnly: Bool = false
    /// Only meaningful with a source image to apply it to
    /// (`RenderDraft+Request.swift` reads `media.sourceImage` to decide).
    /// Stays here rather than on `DraftMedia` -- it is a numeric control like
    /// `guidance`, not a conditioning input, even though it rides with one.
    public var strength: Double = 0.75
    /// What this render is conditioned on besides its prompt and numbers --
    /// the still, references, mask, identity, adapters, ControlNet,
    /// keyframes, extend continuation, audio file and source video, plus the
    /// parking rule that protects all of them across a recipe switch
    /// (`DraftMedia.swift`).
    public var media = DraftMedia()

    /// Filing: title, tags and a collection to file the finished print
    /// under, gated on `canOrganize` at the call site.
    public var title: String = ""
    public var tags: [String] = []
    public var collectionName: String?
    /// Whether the title's own words are folded into tags too. Mirrors
    /// `mold_core::organization::compose_client_tags`; the rule itself is
    /// applied at request time, not here (`ClientTags`, M3 S4).
    public var autoTagTitle: Bool = true
    public var outputFormat: String?
    public var upscaleModel: String?
    /// `false` publishes the print and moves it straight to the trash. `true`
    /// is the server's own default, so a request never has to say so.
    public var savesToGallery: Bool = true
    /// Provenance for a prompt an expand/remix wand produced.
    public var originalPrompt: String?
    public var promptTransform: PromptTransformProvenance?

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
