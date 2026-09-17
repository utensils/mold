import Foundation

/// One render, as the server expects it.
///
/// mold's `GenerateRequest` has roughly sixty fields; this is the text-to-image
/// subset. Everything optional is omitted rather than sent as null, because a
/// present-but-null field is not the same as an absent one to a server that
/// distinguishes "unset" from "explicitly cleared".
public struct GenerateRequest: Codable, Hashable, Sendable {
    public var prompt: String
    public var model: String
    public var width: Int
    public var height: Int
    public var steps: Int
    public var guidance: Double
    public var batchSize: Int
    public var negativePrompt: String?
    /// Absent means the host picks one and reports it back.
    public var seed: UInt64?
    public var saveToGallery: Bool?
    public var frames: Int?
    public var fps: Int?
    /// Which of LTX-2's pipelines to run -- echoed from the recipe's own
    /// `request_selector.pipeline` string, never spelled `"auto"`.
    public var pipeline: String?
    /// LTX-2's audio branch. Sent only as `true` -- see `RenderDraft.request`.
    public var enableAudio: Bool?
    /// Skips the audio branch on a video render. Sent only as `true`, and
    /// only when `VideoOnlyPolicy` finds no conflict.
    public var videoOnly: Bool?
    /// Base64, as mold encodes every byte field on the wire.
    public var sourceImage: String?
    public var sourceImageName: String?
    public var strength: Double?
    /// Base64, in order. Never sent empty -- an empty array and an absent
    /// field mean different things to the host.
    public var editImages: [String]?
    public var referenceWeight: Double?
    /// Base64 PNG, opaque grayscale. Sent only when `sourceImage` is also set
    /// -- `validation.rs:3101-3107` refuses a mask with no source.
    public var maskImage: String?
    /// The adapter stack. There is no `lora`/singular field on this Swift
    /// request and there never will be: `types.rs:3419-3444` shows the
    /// server still accepts a legacy singular `lora`, but this app always
    /// speaks the plural `loras` form.
    public var loras: [LoraChoice]?
    /// One identity photograph. Mutually exclusive with `idImages` --
    /// `IdentityConditioning.wire(maxPhotos:)` is the only place either gets
    /// set, and it produces one or the other, never both (`identity.rs:981`).
    public var idImage: String?
    public var idImageName: String?
    public var idImages: [String]?
    public var idImageNames: [String]?
    public var idWeight: Double?
    public var idStartStep: Int?
    /// ControlNet. Sent both or neither -- `RenderDraft+Request.swift`'s
    /// `applyControl` is the only place either gets set
    /// (`validation.rs:3079-3090`, a symmetric pair). There is no
    /// `control_image_name` field on the wire.
    public var controlImage: String?
    public var controlModel: String?
    public var controlScale: Double?
    /// LTX-2 keyframe interpolation (`types.rs:2367-2377`). Never sent
    /// alongside `extendVideo` (`validation.rs:1851-1853`).
    public var keyframes: [KeyframeCondition]?
    /// An existing clip to continue, base64. No wire `name` field.
    public var extendVideo: String?
    /// `step·k+1`, strictly below `frames`; absent means the family's own
    /// default (`validation.rs:1855-1880`, decision 12 in the M4 design).
    public var extendOverlapFrames: Int?
    /// Conditioning audio for audio-to-video, base64. No wire `name` field.
    public var audioFile: String?
    /// Reference video for video-to-video, base64. No wire `name` field.
    /// Never sent alongside `extendVideo` (`validation.rs:1834-1840`).
    public var sourceVideo: String?
    /// Echoed back from the recipe's own advertised `formats` -- a `String`
    /// rather than a Swift enum, so the app's whole job is to echo one back
    /// without inventing a spelling that could drift from the recipe's.
    /// The sampler, echoed from the recipe's own advertised `schedulers`. A
    /// `String` rather than a Swift enum for the same reason `outputFormat`
    /// is one: the server's `Scheduler` is a STRICT enum (`types.rs:138-154`)
    /// and inventing a spelling would refuse the whole body.
    public var scheduler: String?
    /// CFG++. Sent only as `true` -- absence IS `false` to the server
    /// (`types.rs:3293`), so an explicit `false` would be a value nobody chose.
    public var cfgPlus: Bool?
    /// Wan's flow shift (`types.rs:2152`). Absent keeps the tier's own.
    public var sampleShift: Double?
    /// Wan's Lightning distill strengths, per expert (`types.rs:2158-2161`).
    /// Absent is 1.0.
    public var distillStrengthHigh: Double?
    public var distillStrengthLow: Double?
    /// LTX-2's per-request guidance overrides. NEVER sent empty: an absent
    /// field keeps the pipeline's own constant, and `{}` is refused outright
    /// (`validation.rs:1728-1733`).
    public var guidanceOverrides: Ltx2GuidanceOverrides?
    /// Opaque client-shaped crop/pad provenance. The engine never reads it --
    /// the fitting happens here, before the bytes ship -- but recording it
    /// verbatim is what lets Reuse restore the crop controls
    /// (`types.rs:3268-3273`).
    public var sourceFit: SourceFit?
    public var outputFormat: String?
    public var upscaleModel: String?
    /// User-authored print title. Validated at admission; absent means
    /// untitled.
    public var title: String?
    /// Additive; absent means "file under nothing".
    public var tags: [String]?
    /// Resolved by name -- an id is only ever right on one host.
    public var collection: CollectionRef?
    /// Set by the client when a prompt was expanded or remixed locally, so an
    /// older host retains the root/source prompt even without the
    /// structured form below.
    public var originalPrompt: String?
    public var promptTransform: PromptTransformProvenance?
    /// Durable client-generated identifier shared by prepared batch siblings.
    /// The three `batch*` fields ride together or not at all
    /// (`queue_media_admission.rs:937-947`).
    public var batchId: String?
    public var batchIndex: Int?
    public var batchCount: Int?

    public init(
        prompt: String, model: String, width: Int, height: Int, steps: Int,
        guidance: Double, batchSize: Int = 1, negativePrompt: String? = nil,
        seed: UInt64? = nil, saveToGallery: Bool? = nil
    ) {
        self.prompt = prompt
        self.model = model
        self.width = width
        self.height = height
        self.steps = steps
        self.guidance = guidance
        self.batchSize = batchSize
        self.negativePrompt = negativePrompt
        self.seed = seed
        self.saveToGallery = saveToGallery
    }
}
