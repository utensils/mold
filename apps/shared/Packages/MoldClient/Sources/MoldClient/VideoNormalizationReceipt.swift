import Foundation

struct VideoNormalizationReceipt: Decodable {
    let version: Int
    let kind: String
    let originalSha256: String
    let originalByteLength: Int
    let originalMimeType: String
    let canonicalSha256: String
    let canonicalByteLength: Int
    let canonicalMimeType: String
    let converted: Bool

    func canonical(metadata: ReferenceUploadMetadata, entry: VideoNormalizationCapture.Entry,
                   capabilities: VideoNormalizationCapabilities) throws -> GenerationReference {
        let refusal = ReferenceUploadPolicy.refusal("REFERENCE_NORMALIZATION_RESPONSE_MISMATCH", "The host returned mismatched video normalization facts.")
        guard let bytes = entry.bytes, version == 1, kind == entry.original.kind,
              originalSha256 == entry.digest, originalByteLength == bytes.count,
              originalMimeType == entry.mimeType, ReferenceUploadPolicy.digest(canonicalSha256),
              canonicalByteLength > 0, canonicalByteLength <= capabilities.maxCanonicalFileBytes!,
              metadata.sha256 == canonicalSha256, metadata.mimeType == canonicalMimeType else { throw refusal }
        if converted {
            guard entry.original.freshVideo != nil, kind == "video", canonicalMimeType == capabilities.canonicalMimeType else { throw refusal }
        } else {
            guard canonicalSha256 == originalSha256, canonicalByteLength == originalByteLength,
                  canonicalMimeType == originalMimeType else { throw refusal }
        }
        // Metadata validation already enforces positive exact frame/audio facts and a complete soundtrack tuple.
        var original = entry.original.canonicalReference ?? GenerationReference(kind: "video",
            media: .init(authority: "descriptor"), mimeType: canonicalMimeType,
            provenance: .init(name: entry.original.name, sha256: canonicalSha256))
        original.mimeType = canonicalMimeType
        if original.provenance == nil { original.provenance = .init() }
        original.provenance?.sha256 = canonicalSha256
        return try metadata.canonical(original: original, index: entry.index)
    }
}

struct VideoNormalizationFinalizeReceipt: Codable, Equatable {
    let reference: Int
    let kind: String
    let originalSha256: String
    let canonicalSha256: String
}
struct VideoNormalizationSession: Decodable {
    let protocolVersion: Int
    let instanceId: String
    let sessionId: String
    let candidateScopeSha256: String
    let expiresAtMs: Int64
    let sessionHandle: String
    let sessionComplete: Bool
    let uploads: [ReferenceUploadSessionResponse.Slot]
}
struct VideoNormalizationUpload: Decodable {
    let protocolVersion: Int
    let instanceId: String
    let sessionId: String
    let candidateScopeSha256: String
    let reference: Int
    let sessionComplete: Bool
    let receipt: VideoNormalizationReceipt
    let metadata: ReferenceUploadMetadata
}
struct VideoNormalizationFinal: Decodable {
    let protocolVersion: Int
    let instanceId: String
    let sessionId: String
    let candidateScopeSha256: String
    let requestScopeSha256: String
    let sessionComplete: Bool
    let references: [GenerationReference]
    let receipts: [VideoNormalizationFinalizeReceipt]
}
