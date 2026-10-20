import Foundation

public struct LinkedPartLimits: Codable, Hashable, Sendable {
    public let supported: Bool
    public let maxStagesPerPart: Int
    /// nil is unlimited successor parts; each submitted part stays bounded.
    public let maxParts: UInt64?
}
public struct ChainAuthoringLimits: Codable, Hashable, Sendable {
    public let motionTailFrames: Int
    public let carryMode: String
    public let minFrames: Int
    public let frameStep: Int
    public let frameOffset: Int
}
