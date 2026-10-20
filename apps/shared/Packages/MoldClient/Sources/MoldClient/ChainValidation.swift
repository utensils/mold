import Foundation

public struct ChainValidationStage: Codable, Hashable, Sendable {
    public let prompt: String
    public let frames: Int
    public let outputFrames: Int
    public let transition: String
    public let fadeFrames: Int?
    public let hasSourceImage: Bool
    public let hasNegativePrompt: Bool
}

/// Delivered duration is the host's normalized result, including carry and fade trims.
public struct ChainValidationResponse: Decodable, Hashable, Sendable {
    public let model: String
    public let width: Int
    public let height: Int
    public let fps: Int
    public let motionTailFrames: Int
    public let stageCount: Int
    public let estimatedTotalFrames: Int
    public let estimatedDurationMs: UInt64
    public let stages: [ChainValidationStage]
    public let warnings: [String]
}

public struct ChainScriptSettings: Codable, Hashable, Sendable {
    public var continueFrom: ChainContinuation?
    public var model: String
    public var width: Int
    public var height: Int
    public var fps: Int
    public var seed: UInt64?
    public var steps: Int
    public var guidance: Double
    public var strength: Double
    public var motionTailFrames: Int
    public var outputFormat: String
    public var enableAudio: Bool?
}
public struct ChainScript: Codable, Hashable, Sendable {
    public var schema: String
    public var chain: ChainScriptSettings
    public var stage: [ChainStageInput]
}
