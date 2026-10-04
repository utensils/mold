import Foundation

/// Ephemeral one-use transport authority. Persist only the original inline request.
public struct ReferenceUploadLease: Sendable {
    public let request: GenerateRequest
    public let expiresAtMs: Int64
    public let requestScopeSha256: String
    let cancellation: ReferenceUploadCancellation
    public func cancel() async { await cancellation.cancel() }
}

actor ReferenceUploadCancellation {
    let backend: HTTPBackend
    let path: String
    let header: String
    let handle: String
    var cleanup: Task<Void, Never>?
    init(backend: HTTPBackend, path: String, header: String, handle: String) {
        self.backend = backend; self.path = path; self.header = header; self.handle = handle
    }
    func cancel() async {
        if let cleanup { await cleanup.value; return }
        // Parent cancellation must not prevent releasing the server's session slot.
        let task = Task.detached { [backend, path, header, handle] in
            var request = backend.request(path)
            request.httpMethod = "DELETE"
            request.setValue(handle, forHTTPHeaderField: header)
            _ = try? await backend.referenceUploadBytes(request)
        }
        cleanup = task
        await task.value
    }
}

struct ReferenceUploadSessionResponse: Decodable {
    struct Slot: Decodable { let reference: Int; let handle: String }
    let instanceId: String
    let expiresAtMs: Int64
    let requestScopeSha256: String
    let sessionHandle: String
    let uploads: [Slot]

    func validate(instance: String, indices: [Int], now: Int64) throws {
        guard instanceId == instance else {
            throw ReferenceUploadPolicy.refusal("REFERENCE_UPLOAD_INSTANCE_MISMATCH", "The upload session came from a different Mold instance.")
        }
        guard expiresAtMs > now, ReferenceUploadPolicy.digest(requestScopeSha256),
              ReferenceUploadPolicy.secret(sessionHandle),
              Set(uploads.map(\.reference)) == Set(indices), uploads.count == indices.count,
              Set(uploads.map(\.handle)).count == uploads.count,
              uploads.allSatisfy({ ReferenceUploadPolicy.secret($0.handle) }) else {
            throw ReferenceUploadPolicy.refusal("REFERENCE_UPLOAD_SESSION_INVALID", "The host returned invalid upload authority.")
        }
    }
}

struct ReferenceUploadCompletion: Decodable {
    let instanceId: String
    let reference: Int
    let requestScopeSha256: String
    let sessionComplete: Bool
    let metadata: ReferenceUploadMetadata
}

struct ReferenceUploadMetadata: Decodable {
    let index: Int
    let kind: String
    let mimeType: String
    let sha256: String
    let name: String?
    let width: Int?
    let height: Int?
    let durationMs: Int?
    let sampleRate: Int?
    let sampleCount: Int?
    let channels: Int?
    let fps: Double?
    let frameCount: Int?
    let hasAudio: Bool?
    let audioDurationMs: Int?
    let audioSampleCount: Int?
    let audioSampleRate: Int?
    let audioChannels: Int?
}
