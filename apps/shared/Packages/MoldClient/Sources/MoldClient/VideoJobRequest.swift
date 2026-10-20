import Foundation

/// One prompt and requested time; the host owns clip boundaries and joins.
/// This new protocol uses literal JSON prompts, without chain escape rewriting.
public struct VideoJobRequest: Codable, Hashable, Sendable {
    public var generation: GenerateRequest
    @DecimalUInt64 public var durationMs: UInt64
    public var source: VideoJobSource?
    public var scenes: [VideoJobScene]?
    public var validationRevision: String?
    /// Ephemeral header authority; excluded from JSON and persisted drafts.
    public var retainedMediaSession: String?
    private enum CodingKeys: String, CodingKey {
        case generation, durationMs, source, scenes, validationRevision
    }

    public init(generation: GenerateRequest, durationMs: UInt64, source: VideoJobSource? = nil,
                scenes: [VideoJobScene]? = nil, validationRevision: String? = nil, retainedMediaSession: String? = nil) {
        self.generation = generation; self.durationMs = durationMs
        self.source = source; self.scenes = scenes; self.validationRevision = validationRevision
        self.retainedMediaSession = retainedMediaSession
    }
}

/// Exact Library identity on the captured owner; never a local path or chain id.
public struct VideoJobSource: Codable, Hashable, Sendable {
    public var filename: String
    public var expectedArchiveIdentitySha256: String?
    public init(filename: String, expectedArchiveIdentitySha256: String? = nil) {
        self.filename = filename; self.expectedArchiveIdentitySha256 = expectedArchiveIdentitySha256
    }
}

public struct VideoJobScene: Codable, Hashable, Sendable {
    public var prompt: String
    @DecimalUInt64 public var durationMs: UInt64
    public init(prompt: String, durationMs: UInt64) { self.prompt = prompt; self.durationMs = durationMs }
}
