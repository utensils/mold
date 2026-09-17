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
            guard var conditioning = identity else { return }
            let range = Identity.startStepRange(steps: steps)
            let clamped = Swift.min(Swift.max(conditioning.startStep, range.lowerBound), range.upperBound)
            guard clamped != conditioning.startStep else { return }
            conditioning.startStep = clamped
            identity = conditioning
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
    /// A still to condition on, already base64-encoded, with the name the host
    /// should record for it.
    public var sourceImage: String?
    public var sourceImageName: String?
    public var strength: Double = 0.75
    /// Ordered reference images, base64. For a recipe whose first image is the
    /// Target, index 0 is that one.
    public var editImages: [String] = []
    public var referenceWeight: Double?
    /// A repaint mask over `sourceImage`, base64 PNG. Meaningless without a
    /// source, and dropped at request time when there is none
    /// (`validation.rs:3101-3107`).
    public var maskImage: String?
    /// Face-identity conditioning. One value whether it carries one
    /// photograph or four -- the wire shape is chosen at request time from
    /// the host's `multi_photo`, so `id_image` and `id_images` can never
    /// both be set (`IdentityConditioning.wire(maxPhotos:)`).
    public var identity: IdentityConditioning?
    /// ControlNet conditioning. Parked/restored the same way as every other
    /// conditioning input (`RenderDraft+Park.swift`) -- see
    /// `ControlConditioning`'s own doc comment for why both its halves are
    /// optional.
    public var control: ControlConditioning?
    /// The adapter stack, in the order it was added. Never written into the
    /// legacy singular `lora` field (`types.rs:3419-3444`) -- there is no
    /// Swift equivalent of it and there never will be.
    public var loras: [LoraChoice] = []
    // LTX-2 keyframe interpolation and continuation ("extend"). The
    // invariants -- mutually exclusive, an extend parks the source image,
    // overlap snapped to the recipe's own temporal grid -- live in
    // `RenderDraft+Clip.swift`, beside the pure functions that enforce them.
    public var keyframes: [KeyframeCondition] = []
    /// An existing clip to continue, base64. `extendVideoName` is display
    /// only -- there is no `extend_video_name` on the wire (`types.rs:2098-2102`).
    public var extendVideo: String?
    public var extendVideoName: String?
    /// Carryover pixel frames for the continuation. `nil` sends nothing, so
    /// the server fills in the family's own default (decision 12, M4
    /// design) -- see `RenderDraft.snappedOverlap`.
    public var extendOverlapFrames: Int?
    /// Conditioning audio for LTX-2 audio-to-video, base64. `audioFileName`
    /// is display only -- no `audio_file_name` on the wire.
    public var audioFile: String?
    public var audioFileName: String?
    /// Reference video conditioning, base64. `sourceVideoName` is display
    /// only -- no `source_video_name` on the wire.
    public var sourceVideo: String?
    public var sourceVideoName: String?

    /// What the CURRENT recipe cannot take, held so it comes back
    /// (`RenderDraft+Park.swift`).
    public var parked = ParkedConditioning()

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
