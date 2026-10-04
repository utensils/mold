import Foundation

struct ReferenceUploadCapture {
    let index: Int
    let original: GenerationReference
    let descriptor: GenerationReference
    let bytes: Data?

    static func capture(_ request: GenerateRequest, capabilities: ReferenceUploadCapabilities) throws -> [Self] {
        guard request.batchSize == 1, let references = request.references, !references.isEmpty else {
            throw ReferenceUploadPolicy.refusal("REFERENCE_UPLOAD_REQUEST_INVALID", "Reference uploads require one output and ordered references.")
        }
        var sessionBytes = 0
        let captured = try references.enumerated().map { offset, original -> Self in
            var original = original
            let index = offset + 1
            guard ["image", "video", "audio"].contains(original.kind), !original.mimeType.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw ReferenceUploadPolicy.refusal("REFERENCE_UPLOAD_REQUEST_INVALID", "Unsupported reference upload descriptor.")
            }
            if let raw = original.provenance?.name {
                let name = raw.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !name.isEmpty, name.utf8.count <= 255, name != ".", name != "..",
                      !name.contains("/"), !name.contains("\\"),
                      name.unicodeScalars.allSatisfy({ !CharacterSet.controlCharacters.contains($0) }) else {
                    throw ReferenceUploadPolicy.refusal("REFERENCE_UPLOAD_REQUEST_INVALID", "Invalid display-only reference filename.")
                }
                original.provenance?.name = name
            }
            var bytes: Data?
            switch original.media.authority {
            case "inline":
                guard let encoded = original.media.data,
                      let maximum = capabilities.maxFileBytes,
                      encoded.utf8.count <= ((maximum / 3) + 1) * 4,
                      let decoded = Data(base64Encoded: encoded), !decoded.isEmpty,
                      decoded.base64EncodedString() == encoded else {
                    throw ReferenceUploadPolicy.refusal("REFERENCE_UPLOAD_MEDIA_INVALID", "Reference media is not canonical padded base64 or exceeds the file limit.")
                }
                guard decoded.count <= maximum, let limit = capabilities.maxSessionBytes,
                      decoded.count <= limit - sessionBytes else {
                    throw ReferenceUploadPolicy.refusal("REFERENCE_UPLOAD_TOO_LARGE", "References exceed the host upload limit.")
                }
                sessionBytes += decoded.count
                let digest = RelayTransport.sha256(decoded)
                if let declared = original.provenance?.sha256,
                   declared.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() != digest {
                    throw ReferenceUploadPolicy.refusal("REFERENCE_UPLOAD_DIGEST_MISMATCH", "Reference media does not match its digest.")
                }
                if original.provenance == nil { original.provenance = .init() }
                original.provenance?.sha256 = digest
                bytes = decoded
            case "server_path":
                guard let path = original.media.path, !path.isEmpty,
                      let digest = original.provenance?.sha256, ReferenceUploadPolicy.digest(digest) else {
                    throw ReferenceUploadPolicy.refusal("REFERENCE_UPLOAD_REQUEST_INVALID", "A host reference needs a path and exact digest.")
                }
                original.provenance?.sha256 = digest.lowercased()
            default:
                throw ReferenceUploadPolicy.refusal("REFERENCE_UPLOAD_REQUEST_INVALID", "Prepare uploads from original media, never a previous handle.")
            }
            var descriptor = original
            descriptor.media = .init(authority: "descriptor")
            return Self(index: index, original: original, descriptor: descriptor, bytes: bytes)
        }
        guard captured.contains(where: { $0.bytes != nil }) else {
            throw ReferenceUploadPolicy.refusal("REFERENCE_UPLOAD_REQUEST_INVALID", "The request has no reference media to upload.")
        }
        return captured
    }
}
