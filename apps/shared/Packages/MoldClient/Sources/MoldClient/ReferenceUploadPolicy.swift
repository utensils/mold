import Foundation

/// Validation of capability-provided routes and credentials before any bytes leave the device.
public enum ReferenceUploadPolicy {
    static func refusal(_ code: String, _ message: String) -> MoldClientError {
        .http(status: 422, code: code, message: message)
    }

    static func matches(_ value: String, _ pattern: String) -> Bool {
        value.range(of: pattern, options: .regularExpression) != nil
    }

    static func digest(_ value: String) -> Bool { matches(value, "^[0-9a-fA-F]{64}$") }
    static func secret(_ value: String) -> Bool {
        !value.isEmpty && value.utf8.count <= 1024 && matches(value, "^[A-Za-z0-9._~-]+$")
    }

    public static func validate(_ caps: ReferenceUploadCapabilities) throws {
        guard caps.available, caps.protocolVersion == 2, caps.requiresApiKey == true else {
            throw refusal("REFERENCE_UPLOAD_UNAVAILABLE", "The host does not offer authenticated reference uploads.")
        }
        let reserved = Set(["authorization", "connection", "content-length", "content-type", "cookie", "host", "origin", "proxy-authorization", "transfer-encoding", "x-api-key"])
        guard let session = caps.sessionPath, let upload = caps.uploadPath,
              matches(session, "^/api/[A-Za-z0-9/_-]+$"), matches(upload, "^/api/[A-Za-z0-9/_-]+$"), session != upload,
              let sessionHeader = caps.sessionHandleHeader, let uploadHeader = caps.uploadHandleHeader,
              !sessionHeader.isEmpty, !uploadHeader.isEmpty,
              sessionHeader.utf8.count <= 128, uploadHeader.utf8.count <= 128,
              matches(sessionHeader, "^[!#$%&'*+.^_`|~0-9A-Za-z-]+$"),
              matches(uploadHeader, "^[!#$%&'*+.^_`|~0-9A-Za-z-]+$"),
              !reserved.contains(sessionHeader.lowercased()), !reserved.contains(uploadHeader.lowercased()),
              sessionHeader.lowercased() != uploadHeader.lowercased(),
              let file = caps.maxFileBytes, let total = caps.maxSessionBytes,
              file > 0, total >= file, total <= 9_007_199_254_740_991,
              let active = caps.maxActiveSessions, active > 0, active <= 9_007_199_254_740_991,
              let ttl = caps.sessionTtlMs, ttl > 0, ttl <= 9_007_199_254_740_991 else {
            throw refusal("REFERENCE_UPLOAD_CAPABILITY_INVALID", "The host advertised unsafe reference-upload capabilities.")
        }
    }

    public static func shouldUpload(_ request: GenerateRequest, apiKey: String?, capabilities: ReferenceUploadCapabilities?) -> Bool {
        capabilities?.available == true && !(apiKey?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ?? true)
            && request.references?.contains(where: { $0.media.authority == "inline" && ["image", "audio", "video"].contains($0.kind) }) == true
    }

    /// Apple trims AAC padding that the server counts. Upload V2 returns the server's exact facts.
    public static func validateInlineReferences(_ references: [GenerationReference]) throws {
        guard !references.contains(where: { $0.kind == "video" && $0.hasAudio == true && $0.media.authority == "inline" }) else {
            throw refusal("REFERENCE_SOUNDTRACK_REQUIRES_UPLOAD", "Video references with sound need a host with authenticated reference uploads. Connect with an API key, or choose a silent MP4 and a separate PCM WAV audio reference.")
        }
    }

    public static func batchLimit(requests: [GenerateRequest], apiKey: String?, capabilities: ReferenceUploadCapabilities?, batchLimit: Int) -> Int {
        guard requests.contains(where: { shouldUpload($0, apiKey: apiKey, capabilities: capabilities) }),
              let active = capabilities?.maxActiveSessions, active > 0 else { return batchLimit }
        return min(batchLimit, active)
    }
}
