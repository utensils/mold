import Foundation

public enum VideoJoinMode: String, Codable, Hashable, Sendable { case continuous, blendedShots = "blended_shots" }

public struct VideoJobPlan: Codable, Hashable, Sendable {
    public let validationRevision: String
    public let model: String
    public let width: Int
    public let height: Int
    public let fps: Int
    public let singleClipMaxFrames: Int
    public let overlapFrames: Int
    public let joinMode: VideoJoinMode
    @DecimalUInt64 public var clipCount: UInt64
    @DecimalUInt64 public var partCount: UInt64
    @DecimalUInt64 public var addedFrames: UInt64
    @DecimalUInt64 public var sourceFrames: UInt64
    @DecimalUInt64 public var totalFrames: UInt64
    @DecimalUInt64 public var addedDurationMs: UInt64
    @DecimalUInt64 public var totalDurationMs: UInt64
    public let warnings: [String]
}

/// An unknown future state stays read-only; lifecycle controls use server flags.
public enum VideoJobState: String, Codable, Hashable, Sendable {
    case queued, running, paused, completed, failed, interrupted, cancelled, unknown
    /// Eligibility to request a fresh-start offer, never permission to mutate.
    /// Capabilities and the current server offer remain authoritative.
    public var canRequestRestart: Bool { [.failed, .interrupted, .paused, .cancelled].contains(self) }
    public init(from decoder: any Decoder) throws {
        self = Self(rawValue: try decoder.singleValueContainer().decode(String.self)) ?? .unknown
    }
}

/// Public status excludes retained inputs and private manifests.
public struct VideoJob: Codable, Hashable, Sendable, Identifiable {
    public let id: String
    public let state: VideoJobState
    public let plan: VideoJobPlan
    @DecimalUInt64 public var completedClips: UInt64
    public let activeChildId: String?
    public let outputFilename: String?
    public let error: String?
    public let errorDetail: String?
    public let createdAtMs: Int64
    public let updatedAtMs: Int64
    public let canCancel: Bool
    public let canResume: Bool
    public let canDelete: Bool
    public var isSettled: Bool { [.completed, .failed, .interrupted, .cancelled].contains(state) }
}

public struct VideoJobCapabilities: Codable, Hashable, Sendable {
    public let supported: Bool
    public let literalPrompts: Bool
    public let libraryExtend: Bool
    public let sourceTransfer: Bool?
    public let restart: Bool?
}

public extension Capabilities {
    var canCreateVideoJob: Bool { videoJobs?.supported == true && videoJobs?.literalPrompts == true }
    var canRestartVideoJob: Bool { canCreateVideoJob && videoJobs?.restart == true }
    var canTransferVideoSource: Bool { canCreateVideoJob && videoJobs?.sourceTransfer == true }
    var canExtendLibraryVideo: Bool { canCreateVideoJob && videoJobs?.libraryExtend == true }
}
