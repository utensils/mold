import Foundation

/// Additive V1 fresh typed-video ingress. Base authenticated upload V2 stays unchanged.
public struct VideoNormalizationCapabilities: Codable, Hashable, Sendable {
    public let available: Bool
    public let protocolVersion: Int?
    public let inputRoles: [String]?
    public let candidateSessionPath: String?
    public let finalizePath: String?
    public let candidateDigestHeader: String?
    public let maxOriginalFileBytes: Int?
    public let maxCanonicalFileBytes: Int?
    public let maxSessionBytes: Int?
    public let acceptedMimeTypes: [String]?
    public let acceptedExtensions: [String]?
    public let acceptedVideoCodecs: [String]?
    public let canonicalMimeType: String?
}

extension ReferenceUploadCapabilities {
    enum CodingKeys: String, CodingKey {
        case available, protocolVersion, requiresApiKey, sessionPath, uploadPath,
             sessionHandleHeader, uploadHandleHeader, maxFileBytes, maxSessionBytes,
             maxActiveSessions, sessionTtlMs, videoNormalization
    }
    public init(from decoder: Decoder) throws {
        let keyed = try decoder.container(keyedBy: CodingKeys.self)
        available = try keyed.decode(Bool.self, forKey: .available)
        protocolVersion = try keyed.decodeIfPresent(Int.self, forKey: .protocolVersion)
        requiresApiKey = try keyed.decodeIfPresent(Bool.self, forKey: .requiresApiKey)
        sessionPath = try keyed.decodeIfPresent(String.self, forKey: .sessionPath)
        uploadPath = try keyed.decodeIfPresent(String.self, forKey: .uploadPath)
        sessionHandleHeader = try keyed.decodeIfPresent(String.self, forKey: .sessionHandleHeader)
        uploadHandleHeader = try keyed.decodeIfPresent(String.self, forKey: .uploadHandleHeader)
        maxFileBytes = try keyed.decodeIfPresent(Int.self, forKey: .maxFileBytes)
        maxSessionBytes = try keyed.decodeIfPresent(Int.self, forKey: .maxSessionBytes)
        maxActiveSessions = try keyed.decodeIfPresent(Int.self, forKey: .maxActiveSessions)
        sessionTtlMs = try keyed.decodeIfPresent(Int.self, forKey: .sessionTtlMs)
        // A malformed additive advertisement never disables the ordinary V2 path.
        videoNormalization = try? keyed.decodeIfPresent(VideoNormalizationCapabilities.self, forKey: .videoNormalization)
    }
}

public enum VideoNormalizationPolicy {
    public static func capability(_ uploads: ReferenceUploadCapabilities?, apiKey: String?) -> VideoNormalizationCapabilities? {
        guard let uploads, (try? ReferenceUploadPolicy.validate(uploads)) != nil,
              !(apiKey?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ?? true),
              let cap = uploads.videoNormalization, cap.available, cap.protocolVersion == 1,
              cap.inputRoles?.contains("reference_video") == true,
              let candidate = cap.candidateSessionPath, let finalize = cap.finalizePath,
              ReferenceUploadPolicy.matches(candidate, "^/api/[A-Za-z0-9/_-]+$"),
              ReferenceUploadPolicy.matches(finalize, "^/api/[A-Za-z0-9/_-]+$"),
              Set([candidate, finalize, uploads.sessionPath!, uploads.uploadPath!]).count == 4,
              let header = cap.candidateDigestHeader, !header.isEmpty, header.utf8.count <= 128,
              ReferenceUploadPolicy.matches(header, "^[!#$%&'*+.^_`|~0-9A-Za-z-]+$"),
              !Set(["authorization", "connection", "content-length", "content-type", "cookie", "host", "origin", "proxy-authorization", "transfer-encoding", "x-api-key", uploads.sessionHandleHeader!.lowercased(), uploads.uploadHandleHeader!.lowercased()]).contains(header.lowercased()),
              let original = cap.maxOriginalFileBytes, original > 0, original <= uploads.maxFileBytes!,
              let canonical = cap.maxCanonicalFileBytes, canonical > 0, canonical <= uploads.maxFileBytes!,
              let total = cap.maxSessionBytes, total >= max(original, canonical), total <= uploads.maxSessionBytes!,
              let mimes = cap.acceptedMimeTypes, !mimes.isEmpty,
              mimes.allSatisfy({ ReferenceUploadPolicy.matches($0, "^video/[a-z0-9.+-]+$") }),
              let extensions = cap.acceptedExtensions, !extensions.isEmpty,
              extensions.allSatisfy({ ReferenceUploadPolicy.matches($0, "^[a-z0-9]{1,16}$") }),
              let codecs = cap.acceptedVideoCodecs, !codecs.isEmpty, codecs.allSatisfy({ !$0.isEmpty }),
              cap.canonicalMimeType == "video/mp4" else { return nil }
        return cap
    }
}
