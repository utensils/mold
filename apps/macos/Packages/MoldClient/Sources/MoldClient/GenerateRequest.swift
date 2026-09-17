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
    /// Base64, as mold encodes every byte field on the wire.
    public var sourceImage: String?
    public var sourceImageName: String?
    public var strength: Double?
    /// Base64, in order. Never sent empty -- an empty array and an absent
    /// field mean different things to the host.
    public var editImages: [String]?
    public var referenceWeight: Double?
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

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(prompt, forKey: .prompt)
        try container.encode(model, forKey: .model)
        try container.encode(width, forKey: .width)
        try container.encode(height, forKey: .height)
        try container.encode(steps, forKey: .steps)
        try container.encode(guidance, forKey: .guidance)
        try container.encode(batchSize, forKey: .batchSize)
        try container.encodeIfPresent(negativePrompt, forKey: .negativePrompt)
        try container.encodeIfPresent(seed, forKey: .seed)
        try container.encodeIfPresent(saveToGallery, forKey: .saveToGallery)
        try container.encodeIfPresent(frames, forKey: .frames)
        try container.encodeIfPresent(fps, forKey: .fps)
        try container.encodeIfPresent(sourceImage, forKey: .sourceImage)
        try container.encodeIfPresent(sourceImageName, forKey: .sourceImageName)
        try container.encodeIfPresent(strength, forKey: .strength)
        try container.encodeIfPresent(editImages, forKey: .editImages)
        try container.encodeIfPresent(referenceWeight, forKey: .referenceWeight)
        try container.encodeIfPresent(outputFormat, forKey: .outputFormat)
        try container.encodeIfPresent(upscaleModel, forKey: .upscaleModel)
        try container.encodeIfPresent(title, forKey: .title)
        try container.encodeIfPresent(tags, forKey: .tags)
        try container.encodeIfPresent(collection, forKey: .collection)
        try container.encodeIfPresent(originalPrompt, forKey: .originalPrompt)
        try container.encodeIfPresent(promptTransform, forKey: .promptTransform)
        try container.encodeIfPresent(batchId, forKey: .batchId)
        try container.encodeIfPresent(batchIndex, forKey: .batchIndex)
        try container.encodeIfPresent(batchCount, forKey: .batchCount)
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
