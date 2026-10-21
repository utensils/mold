import Foundation

/// Destination-issued, completed prefix authority. Keep only in memory.
public struct VideoSourceUploadRef: Codable, Hashable, Sendable {
    public let handle: String
    public let sha256: String
    public let sizeBytes: UInt64
    public init(handle: String, sha256: String, sizeBytes: UInt64) {
        self.handle = handle
        self.sha256 = sha256
        self.sizeBytes = sizeBytes
    }
    private enum CodingKeys: String, CodingKey { case handle, sha256, sizeBytes }
    public init(from decoder: any Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        handle = try values.decode(String.self, forKey: .handle)
        sha256 = try values.decode(String.self, forKey: .sha256)
        sizeBytes = try values.decode(DecimalUInt64.self, forKey: .sizeBytes).wrappedValue
        guard !handle.isEmpty, sizeBytes > 0, RetainedSourceMedia.validTransferDigest(sha256) else {
            throw MoldClientError.malformedResponse
        }
    }
    public func encode(to encoder: any Encoder) throws {
        var values = encoder.container(keyedBy: CodingKeys.self)
        try values.encode(handle, forKey: .handle)
        try values.encode(sha256, forKey: .sha256)
        try values.encode(DecimalUInt64(wrappedValue: sizeBytes), forKey: .sizeBytes)
    }
}

public struct VideoSourceUploadStatus: Decodable, Hashable, Sendable {
    public let handle: String
    @DecimalUInt64 public var offsetBytes: UInt64
    public let completed: Bool
}

public struct VideoRestartPrefix: Decodable, Hashable, Sendable {
    public let filename: String
    public let archiveIdentitySha256: String
    public let sha256: String
    @DecimalUInt64 public var sizeBytes: UInt64
    public let sourceRef: String
    /// Opaque canonical provenance; never parse or rewrite full-u64 metadata.
    public let uploadSpecJSON: String
    public let retainedUpload: VideoSourceUploadRef?
    private enum CodingKeys: String, CodingKey {
        case filename, archiveIdentitySha256, sha256, sizeBytes, sourceRef, retainedUpload
        case uploadSpecJSON = "uploadSpecJson"
    }
}

/// Parent protocol has literal prompts. It must never use the old chain route.
public struct VideoParentRestartExport: Decodable, Hashable, Sendable {
    public let sourceInstanceId: String
    public let sourceRevision: String
    public let destinationInstanceId: String
    public let requestJSON: String
    public let prefix: VideoRestartPrefix?
    private enum CodingKeys: String, CodingKey {
        case sourceInstanceId, sourceRevision, destinationInstanceId, prefix
        case requestJSON = "requestJson"
    }
    public func validated(for request: ChainRestartExportRequest) throws -> String {
        guard sourceInstanceId == request.sourceInstanceId, sourceRevision == request.sourceRevision,
            destinationInstanceId == request.destinationInstanceId, !requestJSON.isEmpty,
            requestJSON.utf8.count <= 64 * 1024 * 1024
        else { throw MoldClientError.malformedResponse }
        if let prefix {
            guard !prefix.filename.isEmpty, !prefix.sourceRef.isEmpty, prefix.sizeBytes > 0,
                RetainedSourceMedia.validTransferDigest(prefix.sha256),
                RetainedSourceMedia.validTransferDigest(prefix.archiveIdentitySha256),
                !prefix.uploadSpecJSON.isEmpty, prefix.uploadSpecJSON.utf8.count <= 8 * 1024 * 1024
            else {
                throw MoldClientError.malformedResponse
            }
        }
        return requestJSON
    }
    public func localPrefixUpload(destinationInstanceId captured: String) throws -> VideoSourceUploadRef? {
        guard let prefix, let upload = prefix.retainedUpload else { return nil }
        guard sourceInstanceId == captured, destinationInstanceId == captured,
            upload.sha256 == prefix.sha256, upload.sizeBytes == prefix.sizeBytes
        else {
            throw SequenceDraftError.invalid(
                "This video's retained prefix belongs to another machine or changed. Reopen Start again.")
        }
        return upload
    }
}
