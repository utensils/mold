import Foundation

/// Published authoring facts, separate from delivered frames and private job
/// state. Counts remain decimal integers, including beyond JavaScript's range.
public struct VideoJobPrintProvenance: Codable, Hashable, Sendable {
    public let id: String
    @DecimalUInt64 public var sourceFrames: UInt64
    @DecimalUInt64 public var totalFrames: UInt64
    public let authoring: VideoJobPrintAuthoring?
}

public struct VideoJobPrintAuthoring: Codable, Hashable, Sendable {
    @DecimalUInt64 public var durationMs: UInt64
    public let scenes: [VideoJobScene]?
    public let boundaryFrames: Int
    public let inputFrames: Int?
    /// The explicit decimal seed is additive; ordinary metadata also decodes
    /// UInt64 directly without a floating point conversion.
    public let seed: String?
}
