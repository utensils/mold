import Foundation

/// Portable publication assets. The bundle identity is opaque, never a path.
public struct VideoJobArtifacts: Codable, Hashable, Sendable {
    public let version: Int
    public let bundleId: String
    public let provenance: VideoArtifactFile
    public let pcm: VideoPcmArtifact?

    func validate() throws {
        guard version == 1, RetainedSourceMedia.validTransferDigest(bundleId) else { throw MoldClientError.malformedResponse }
        try provenance.validate()
        if let pcm {
            try pcm.file.validate()
            let samples = pcm.frames.multipliedReportingOverflow(by: UInt64(pcm.channels))
            let bytes = samples.partialValue.multipliedReportingOverflow(by: 4)
            guard pcm.sampleRate > 0, pcm.sampleRate <= 192000, (1...2).contains(pcm.channels), pcm.frames > 0,
                  !samples.overflow, !bytes.overflow, bytes.partialValue == pcm.file.sizeBytes else {
                throw MoldClientError.malformedResponse
            }
        }
    }
}

public struct VideoArtifactFile: Codable, Hashable, Sendable {
    public let sha256: String
    @DecimalUInt64 public var sizeBytes: UInt64
    func validate() throws {
        guard sizeBytes > 0, RetainedSourceMedia.validTransferDigest(sha256) else { throw MoldClientError.malformedResponse }
    }
}

public struct VideoPcmArtifact: Codable, Hashable, Sendable {
    public let file: VideoArtifactFile
    public let sampleRate: UInt32
    public let channels: UInt16
    @DecimalUInt64 public var frames: UInt64
}

public enum VideoArtifactRole: String, Codable, Sendable { case provenance, pcm }

public struct VideoArtifactTransferStatus: Codable, Hashable, Sendable {
    public let handle: String
    public let archiveIdentitySha256: String
    public let offsets: VideoArtifactOffsets
    public let complete: Bool
}

public struct VideoArtifactOffsets: Codable, Hashable, Sendable {
    @DecimalUInt64 public var provenance: UInt64
    public let pcm: UInt64?
    private enum CodingKeys: String, CodingKey { case provenance, pcm }
    public init(from decoder: any Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        _provenance = try values.decode(DecimalUInt64.self, forKey: .provenance)
        pcm = try values.decodeIfPresent(DecimalUInt64.self, forKey: .pcm)?.wrappedValue
    }
    public func encode(to encoder: any Encoder) throws {
        var values = encoder.container(keyedBy: CodingKeys.self)
        try values.encode(_provenance, forKey: .provenance)
        try values.encodeIfPresent(pcm.map(String.init), forKey: .pcm)
    }
}
