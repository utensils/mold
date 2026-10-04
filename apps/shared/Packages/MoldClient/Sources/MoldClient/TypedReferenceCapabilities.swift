import Foundation

/// Additive host authority for ordered image/video/audio conditioning.
public struct GenerationReferencesCapability: Codable, Hashable, Sendable {
    public var mode: ControlMode
    public var required: Bool
    public var kinds: [String]
    public var maxCount: Int
    public var maxImages: Int; public var maxVideos: Int; public var maxAudios: Int
    public var minDurationMs: Int; public var maxDurationMs: Int
    public var maxVideoDurationMs: Int; public var maxAudioDurationMs: Int
    public var maxInlineBytes: Int
    public var requiresVisual: Bool
}
public struct NamedViewsCapability: Codable, Hashable, Sendable {
    public var mode: ControlMode
    public var roles: [GenerationImageReferenceRole]
    public var minCount: Int; public var maxCount: Int
    public var reason: String?
}
public struct RecipeMeshCapability: Codable, Hashable, Sendable {
    public var namedViews: NamedViewsCapability?
}
public struct BoundaryFramesCapability: Codable, Hashable, Sendable {
    public var mode: ControlMode
    public var firstRequired: Bool; public var lastRequired: Bool
    public var minFrames: Int
    public var wire: String
}

public extension NamedViewsCapability {
    init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        mode = try values.decode(ControlMode.self, forKey: .mode)
        roles = try values.decode([GenerationImageReferenceRole].self, forKey: .roles).filter { $0 != .unknown }
        minCount = try values.decode(Int.self, forKey: .minCount)
        maxCount = try values.decode(Int.self, forKey: .maxCount)
        reason = try values.decodeIfPresent(String.self, forKey: .reason)
    }
}
