import Foundation

/// Original fresh-video bytes awaiting a negotiated host probe/normalization.
/// No dimensions, timing, frame counts or generation authority are invented here.
public struct FreshVideoReference: Codable, Hashable, Sendable {
    public let data: Data
    public let mimeType: String
    public let name: String
    public let sha256: String
    public init(data: Data, mimeType: String, name: String) {
        // Copy mapped/file-backed bytes into an immutable authoring snapshot.
        self.data = data.withUnsafeBytes { Data($0) }
        self.mimeType = mimeType; self.name = name
        sha256 = TransportDigest.sha256(self.data)
    }
}

/// The ordered draft boundary. Only canonical references may enter GenerateRequest.
public enum AuthoringReference: Hashable, Sendable, Codable {
    case canonical(GenerationReference)
    case freshVideo(FreshVideoReference)

    public init(kind: String, media: GenerationReferenceMedia, mimeType: String,
                provenance: GenerationReferenceProvenance? = nil, width: Int? = nil,
                height: Int? = nil, role: GenerationImageReferenceRole? = nil) {
        self = .canonical(.init(kind: kind, media: media, mimeType: mimeType,
            provenance: provenance, width: width, height: height, role: role))
    }
    public var canonicalReference: GenerationReference? {
        if case let .canonical(value) = self { return value }
        return nil
    }
    public var freshVideo: FreshVideoReference? {
        if case let .freshVideo(value) = self { return value }
        return nil
    }
    public var kind: String { canonicalReference?.kind ?? "video" }
    public var name: String { canonicalReference?.name ?? freshVideo!.name }
    public var mimeType: String { canonicalReference?.mimeType ?? freshVideo!.mimeType }
    public var provenance: GenerationReferenceProvenance? {
        canonicalReference?.provenance ?? freshVideo.map { .init(name: $0.name, sha256: $0.sha256) }
    }
    public var media: GenerationReferenceMedia { canonicalReference?.media ?? .init(authority: "fresh_video") }
    public var role: GenerationImageReferenceRole? { canonicalReference?.role }
    public var width: Int? { canonicalReference?.width }
    public var height: Int? { canonicalReference?.height }
    public var durationMs: Int? { canonicalReference?.durationMs }
    public var frameCount: Int? { canonicalReference?.frameCount }
    public var fps: Double? { canonicalReference?.fps }
    public var sampleCount: Int? { canonicalReference?.sampleCount }
    public var audioDurationMs: Int? { canonicalReference?.audioDurationMs }
    public var hasOriginalBytes: Bool { freshVideo?.data.isEmpty == false || canonicalReference?.media.data != nil }

    public func mappingCanonical(_ transform: (GenerationReference) -> GenerationReference) -> Self {
        if case let .canonical(value) = self { return .canonical(transform(value)) }
        return self
    }

    private enum CodingKeys: String, CodingKey { case authoringType, original }
    public init(from decoder: Decoder) throws {
        let keyed = try decoder.container(keyedBy: CodingKeys.self)
        if keyed.contains(.authoringType) {
            guard try keyed.decode(String.self, forKey: .authoringType) == "fresh_video" else {
                throw DecodingError.dataCorruptedError(forKey: .authoringType, in: keyed, debugDescription: "Unknown authoring reference")
            }
            let original = try keyed.decode(FreshVideoReference.self, forKey: .original)
            guard original.sha256 == TransportDigest.sha256(original.data) else {
                throw DecodingError.dataCorruptedError(forKey: .original, in: keyed, debugDescription: "Original video digest mismatch")
            }
            self = .freshVideo(original)
        } else { self = .canonical(try GenerationReference(from: decoder)) }
    }
    public func encode(to encoder: Encoder) throws {
        switch self {
        case let .canonical(value): try value.encode(to: encoder)
        case let .freshVideo(value):
            var keyed = encoder.container(keyedBy: CodingKeys.self)
            try keyed.encode("fresh_video", forKey: .authoringType)
            try keyed.encode(value, forKey: .original)
        }
    }
}

public extension Array where Element == AuthoringReference {
    /// All-or-nothing: a pending video can never disappear from a mixed request.
    var canonicalReferences: [GenerationReference]? {
        let values = compactMap(\.canonicalReference)
        return values.count == count ? values : nil
    }
}
