import Foundation

public enum GenerationImageReferenceRole: String, OpenWireEnum, CaseIterable { case front, left, back, right, unknown }

/// One explicit authority. Scoped handles and local paths are never persisted as provenance.
public struct GenerationReferenceMedia: Codable, Hashable, Sendable {
    public var authority: String
    public var data: String?
    public var handle: String?
    public var path: String?
    public init(authority: String, data: String? = nil, handle: String? = nil, path: String? = nil) {
        self.authority = authority; self.data = data; self.handle = handle; self.path = path
    }
}
public struct GenerationReferenceCrop: Codable, Hashable, Sendable {
    public var x: Int; public var y: Int; public var width: Int; public var height: Int
    public var sourceWidth: Int; public var sourceHeight: Int; public var sourceSha256: String
}
public struct GenerationReferenceProvenance: Codable, Hashable, Sendable {
    public var name: String?
    public var sha256: String?
    public var crop: GenerationReferenceCrop?
    public init(name: String? = nil, sha256: String? = nil, crop: GenerationReferenceCrop? = nil) {
        self.name = name; self.sha256 = sha256; self.crop = crop
    }
}
/// Ordered heterogeneous reference. Optional facts belong only to their relevant kind.
public struct GenerationReference: Codable, Hashable, Sendable {
    public var kind: String
    public var media: GenerationReferenceMedia
    public var provenance: GenerationReferenceProvenance?
    public var mimeType: String
    public var width: Int?; public var height: Int?
    public var role: GenerationImageReferenceRole?
    public var frameCount: Int?; public var durationMs: Int?; public var fps: Double?
    public var hasAudio: Bool?; public var audioDurationMs: Int?; public var audioSampleCount: Int?
    public var audioSampleRate: Int?; public var audioChannels: Int?
    public var sampleRate: Int?; public var channels: Int?; public var sampleCount: Int?
    public init(kind: String, media: GenerationReferenceMedia, mimeType: String,
                provenance: GenerationReferenceProvenance? = nil, width: Int? = nil,
                height: Int? = nil, role: GenerationImageReferenceRole? = nil) {
        self.kind = kind; self.media = media; self.mimeType = mimeType
        self.provenance = provenance; self.width = width; self.height = height; self.role = role
    }
    public var name: String { provenance?.name ?? kind.capitalized }
    public func redactedForPlacement() -> Self {
        var copy = self
        copy.media = .init(authority: "descriptor")
        copy.provenance?.name = nil
        return copy
    }
}
