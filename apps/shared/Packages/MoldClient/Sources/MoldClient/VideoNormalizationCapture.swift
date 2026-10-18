import Foundation

struct VideoNormalizationCapture {
    struct Entry {
        let index: Int
        let original: AuthoringReference
        let descriptor: GenerationReference?
        let bytes: Data?
        var digest: String { original.freshVideo?.sha256 ?? descriptor!.provenance!.sha256! }
        var mimeType: String { original.mimeType }
    }
    struct Candidate: Encodable {
        let type: String
        let index: Int
        var upload: Bool?
        var reference: GenerationReference?
        var byteLength: Int?
        var kind: String?
        var original: Original?
    }
    struct Original: Encodable { let mimeType: String; let name: String; let byteLength: Int; let sha256: String }
    struct Body: Encodable { let protocolVersion = 1; let request: GenerateRequest; let referenceCandidates: [Candidate] }
    let entries: [Entry]
    let body: Data
    let fingerprint: String

    static func capture(_ request: GenerateRequest, capabilities uploads: ReferenceUploadCapabilities, apiKey: String?) throws -> Self {
        let refusal = ReferenceUploadPolicy.refusal("REFERENCE_NORMALIZATION_INVALID", "Prepare fresh video references from the original files on a compatible keyed host.")
        guard let caps = VideoNormalizationPolicy.capability(uploads, apiKey: apiKey), request.batchSize == 1,
              request.references == nil, let inputs = request.pendingReferenceInputs, !inputs.isEmpty,
              inputs.contains(where: { $0.freshVideo != nil }) else { throw refusal }
        var total = 0
        var candidates: [Candidate] = []
        var entries: [Entry] = []
        for (offset, input) in inputs.enumerated() {
            let index = offset + 1
            switch input {
            case let .freshVideo(original):
                guard !original.data.isEmpty, original.data.count <= caps.maxOriginalFileBytes!,
                      original.sha256 == TransportDigest.sha256(original.data),
                      caps.acceptedMimeTypes?.contains(original.mimeType) == true,
                      validName(original.name) else { throw refusal }
                candidates.append(.init(type: "fresh_video", index: index, kind: "video",
                    original: .init(mimeType: original.mimeType, name: original.name,
                        byteLength: original.data.count, sha256: original.sha256)))
                entries.append(.init(index: index, original: input, descriptor: nil, bytes: original.data))
            case let .canonical(reference):
                var single = request; single.pendingReferenceInputs = nil; single.references = [reference]
                let entry = try ReferenceUploadCapture.capture(single, capabilities: uploads, requireUpload: false)[0]
                candidates.append(.init(type: "canonical", index: index, upload: entry.bytes != nil,
                    reference: entry.bytes == nil ? entry.original : entry.descriptor, byteLength: entry.bytes?.count))
                entries.append(.init(index: index, original: .canonical(entry.original), descriptor: entry.descriptor, bytes: entry.bytes))
            }
            if let bytes = entries.last?.bytes {
                guard bytes.count <= caps.maxSessionBytes! - total else { throw refusal }
                total += bytes.count
            }
        }
        var scoped = request; scoped.pendingReferenceInputs = nil; scoped.references = nil
        let body = try MoldJSON.encoder.encode(Body(request: scoped, referenceCandidates: candidates))
        var domain = Data("mold-reference-normalization-candidate-v1\0".utf8); domain.append(body)
        return Self(entries: entries, body: body, fingerprint: TransportDigest.sha256(domain))
    }
    private static func validName(_ name: String) -> Bool {
        !name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && name.utf8.count <= 255
            && name != "." && name != ".." && !name.contains("/") && !name.contains("\\")
            && name.unicodeScalars.allSatisfy { !CharacterSet.controlCharacters.contains($0) }
    }
}
