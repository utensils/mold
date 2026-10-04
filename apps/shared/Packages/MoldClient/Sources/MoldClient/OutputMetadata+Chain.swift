import Foundation

// The multi-clip and ordered-conditioning provenance a print can carry.
// Separate from the metadata's own shape because all three are read for ONE
// question apiece -- how many stages, how many references, how many
// keyframes -- and modelling them in full would be mirroring three server
// types nothing here consumes.

/// How a stitched clip was split (`chain.rs:66-70`).
///
/// **A print made by a sequence is provenance, not a door back**: this is
/// what makes the first stage's prompt findable, and it is never an
/// authoring surface.
public struct ChainProvenance: Codable, Hashable, Sendable {
    public let stageCount: Int?
    public let motionTailFrames: Int?
    public let stages: [ChainStageProvenance]?
}

/// One clip of a stitched render. `seed` is a STRING on the wire -- a u64
/// seed does not survive JSON's number type, so mold writes it as text and a
/// client that typed it as a number here would fail the whole print.
public struct ChainStageProvenance: Codable, Hashable, Sendable {
    public let prompt: String?
    public let frames: Int?
    public let transition: String?
    public let seed: String?
}

/// One ordered reference, redacted: display-safe labels, digests and probed
/// media facts, never bytes or handles (`types.rs:3279-3283`).
public struct ReferenceProvenance: Codable, Hashable, Sendable {
    public let kind: String?
    public let index: Int?
    public let name: String?
    public let sha256: String?
    public let mimeType: String?
    public let width: Int?
    public let height: Int?
    public var frameCount: Int? = nil
    public var durationMs: Int? = nil
    public var fps: Double? = nil
    public var hasAudio: Bool? = nil
    public var audioDurationMs: Int? = nil
    public var audioSampleCount: Int? = nil
    public var audioSampleRate: Int? = nil
    public var audioChannels: Int? = nil
    public var sampleRate: Int? = nil
    public var channels: Int? = nil
    public var sampleCount: Int? = nil
    public var crop: GenerationReferenceCrop? = nil
    public var imageRole: GenerationImageReferenceRole? = nil

    public var generationReference: GenerationReference? {
        guard let kind, let mimeType else { return nil }
        var ref = GenerationReference(kind: imageRole == nil ? kind : "named_image",
            media: .init(authority: "descriptor"), mimeType: mimeType,
            provenance: .init(name: name, sha256: sha256, crop: crop), width: width, height: height, role: imageRole)
        ref.frameCount = frameCount; ref.durationMs = durationMs; ref.fps = fps
        ref.hasAudio = hasAudio; ref.audioDurationMs = audioDurationMs
        ref.audioSampleCount = audioSampleCount; ref.audioSampleRate = audioSampleRate
        ref.audioChannels = audioChannels; ref.sampleRate = sampleRate
        ref.channels = channels; ref.sampleCount = sampleCount
        return ref
    }
}

/// One pinned still, byte-free (`types.rs:3284-3287`).
public struct KeyframeProvenance: Codable, Hashable, Sendable {
    public let frame: Int?
    public let name: String?
    public let sha256: String?
}
