import Foundation

/// How a print was made -- the whole recipe, as the host recorded it
/// (`mold_core::OutputMetadata`, `types.rs:3127-3407`).
///
/// Every field but `prompt` and `model` is optional on purpose: this metadata
/// spans years of mold versions, and a print made before a field existed is a
/// normal print, not a corrupt one. There are no `CodingKeys` here and there
/// must not be -- `MoldJSON.decoder`'s snake_case conversion is what keeps
/// this in step with the Rust, and a hand-written key block is the one way to
/// silently decode a field as `nil` forever.
///
/// Every server enum is a plain `String` for the same reason
/// `GenerateRequest.scheduler` is: a host newer than this build will send a
/// sampler, a pipeline or a container it has never heard of, and losing the
/// whole print -- its prompt, its seed, its size -- over one word is not a
/// trade worth making. The values are only ever echoed back.
public struct OutputMetadata: Codable, Hashable, Sendable {

    // What was asked for.
    public let prompt: String?
    public let negativePrompt: String?
    public let originalPrompt: String?
    public let promptTransform: PromptTransformProvenance?
    public let model: String?
    /// Absence means an older print or a model the manifest cannot classify,
    /// never "no family" (`types.rs:3190-3201`).
    public let family: String?
    public let title: String?
    public let tags: [String]?
    /// The collection's DISPLAY NAME as resolved, never an id -- an id is
    /// only ever right on one machine (`types.rs:3142-3146`).
    public let collection: String?

    // The numbers.
    public let seed: UInt64?
    public let steps: Int?
    public let guidance: Double?
    public let width: Int?
    public let height: Int?
    /// What was actually rendered, when it differs from the delivered size --
    /// an upscaled print's `width` is the final one, not the one to reuse.
    public let generationWidth: Int?
    public let generationHeight: Int?
    public let strength: Double?
    public let scheduler: String?
    public let cfgPlus: Bool?
    /// Wan's flow shift and Lightning distill strengths. Absent means the
    /// tier's own (`types.rs:3372-3384`).
    public let sampleShift: Double?
    public let distillStrengthHigh: Double?
    public let distillStrengthLow: Double?
    public let guidanceOverrides: Ltx2GuidanceOverrides?

    // Conditioning, as provenance: names, digests and shapes, never bytes.
    public let sourceImageName: String?
    public let sourceImageSha256: String?
    /// The opaque client-shaped crop/pad policy, kept verbatim so Reuse
    /// restores the crop controls exactly (`types.rs:3268-3273`).
    public let sourceFit: SourceFit?
    /// `edit_image_sha256s`. Spelled with the trailing capital because
    /// `convertFromSnakeCase` runs `"sha256s".capitalized`, and `256` is a
    /// word boundary to Foundation -- so the key arrives as `Sha256S` and a
    /// property spelled the way a person would spell it decodes as `nil`
    /// forever. Read it through `editImageDigests`, never by this name.
    public let editImageSha256S: [String]?
    public let idImageName: String?
    public let idImageSha256: String?
    public let idImageNames: [String]?
    /// The same trap as `editImageSha256S`. Read `identityDigests`.
    public let idImageSha256S: [String]?
    public let idWeight: Double?
    public let idStartStep: Int?
    public let references: [ReferenceProvenance]?
    public let keyframes: [KeyframeProvenance]?
    public let audioFilePath: String?
    public let sourceVideoPath: String?
    public let extendVideoPath: String?
    public let extendOverlapFrames: Int?
    public let loras: [MetadataLora]?
    /// The legacy singular pair, which is all a pre-stack print carries.
    public let lora: String?
    public let loraScale: Double?
    public let controlModel: String?
    public let controlScale: Double?
    public let mesh: MeshProvenance?

    // Clip and output shape.
    public let frames: Int?
    public let fps: Double?
    public let enableAudio: Bool?
    public let videoOnly: Bool?
    public let pipeline: String?
    /// Whether the authored request NAMED a pipeline. `pipeline` records what
    /// ran, so without this a default would be promoted into an override
    /// (`types.rs:3329-3334`).
    public let pipelineRequested: Bool?
    public let durationPredictionRequested: Bool?
    public let icLoraControl: String?
    public let retakeRange: TimeRangeProvenance?
    public let spatialUpscale: String?
    public let temporalUpscale: String?
    public let outputFormat: String?
    public let upscaleModel: String?

    // Where it came from.
    public let generationTimeMs: Int?
    public let jobId: String?
    /// `one-shot` or `sequence`. A sequence's `prompt` is every stage joined
    /// by newlines, which is why reuse must never restore it wholesale.
    public let outputMode: String?
    /// Both an authored sequence and an auto-chained one-shot carry this;
    /// only the durable job carries the id beside it.
    public let chain: ChainProvenance?
    public let chainJobId: String?
    public let version: String?
}

/// One adapter as metadata records it: `LoraWeight` has a path and a scale
/// and no name (`types.rs:2648-2667`), so this is not `LoraChoice` -- the
/// name a draft shows is derived from the path at restore time.
public struct MetadataLora: Codable, Hashable, Sendable {
    public let path: String
    public let scale: Double
}

/// The 3-D controls that actually ran (`MeshRequestOptions`,
/// `types.rs:2929-2960`). Recorded only on a mesh print.
public struct MeshProvenance: Codable, Hashable, Sendable {
    public let octreeResolution: Int?
    public let threshold: Double?
    public let targetFaces: Int?
    public let texture: Bool?
}

/// `TimeRange` (`types.rs:2380-2385`): LTX-2's retake window.
public struct TimeRangeProvenance: Codable, Hashable, Sendable {
    public let startSeconds: Double?
    public let endSeconds: Double?
}
