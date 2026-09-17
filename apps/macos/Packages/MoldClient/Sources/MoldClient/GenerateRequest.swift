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
    /// Echoed back from the recipe's own advertised `formats` -- a `String`
    /// rather than a Swift enum, so the app's whole job is to echo one back
    /// without inventing a spelling that could drift from the recipe's.
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

    /// Declared explicitly, rather than left to the compiler, ONLY because a
    /// synthesized `CodingKeys` is `private` and therefore invisible from
    /// `GenerateRequest+Encoding.swift`'s `encode(to:)` -- every other type
    /// in this package leaves `CodingKeys` to `MoldJSON`'s snake_case
    /// conversion (see its own doc comment) and this is the sole exception,
    /// forced by splitting the encoder out for size. Case names still match
    /// the properties one for one, so there is nothing here for a typo to
    /// hide behind.
    enum CodingKeys: String, CodingKey {
        case prompt, model, width, height, steps, guidance, batchSize, negativePrompt, seed,
             saveToGallery, frames, fps, pipeline, enableAudio, videoOnly, sourceImage,
             sourceImageName, strength, editImages, referenceWeight, maskImage, loras, idImage,
             idImageName, idImages, idImageNames, idWeight, idStartStep, controlImage,
             controlModel, controlScale, outputFormat, upscaleModel, title, tags, collection,
             originalPrompt, promptTransform, batchId, batchIndex, batchCount
    }
}

/// A batch is one atomic admission of up to 64 ordered children. There is no
/// separate "single render" path on the server -- a one-off is a batch of one.
public struct BatchAdmission: Codable, Sendable {
    /// Minted on the device and PERSISTED BEFORE SENDING. This is the
    /// idempotency fence: if the response is lost, the work is recovered by
    /// asking the host about this id, never by submitting again.
    public let clientBatchId: String
    public let requests: [GenerateRequest]

    public init(clientBatchId: String = UUID().uuidString, requests: [GenerateRequest]) {
        self.clientBatchId = clientBatchId
        self.requests = requests
    }
}
