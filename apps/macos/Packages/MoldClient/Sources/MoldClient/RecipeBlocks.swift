import Foundation

// The wire blocks a `GenerationRecipe` carries, split from `Recipe.swift` for
// size -- the precedent being `CapabilityBlocks.swift` beside
// `Capabilities.swift`. Every one of these is read through
// `RecipeCapabilities+Reading`, never directly.

public struct PromptCapability: Codable, Hashable, Sendable {
    public let mode: PromptRequirement
    public let reason: String?

    /// Absence means `required` -- that is the server's own default, and it is
    /// the safe reading: asking for a prompt that turns out to be optional
    /// costs nothing, omitting one that was required fails the request.
    public static let assumedRequired = PromptCapability(mode: .required, reason: nil)
}

public struct OutputCapabilities: Codable, Hashable, Sendable {
    public let defaultFormat: String
    public let formats: [String]
    public let audioRequiresMp4: Bool?
    /// Real, and the only thing that explains a one-entry `formats` list: a
    /// mesh recipe delivers only GLB, an audio-only recipe only WAV.
    /// `generation_profile.rs:495-502`.
    public let deliveryReason: String?

    /// A recipe with one deliverable container has nothing to pick between.
    public var isFixed: Bool { formats.count <= 1 }
}

/// How reference images relate to a source image on this recipe.
public enum ReferenceSourceRelation: String, OpenWireEnum {
    /// The references ARE the conditioning: no strength, no mask, no source.
    case replaces
    /// The recipe keeps its source paths, but one render carries a source
    /// image OR references, never both.
    case exclusive
    /// The references ride alongside img2img, inpaint and a LoRA.
    case combines
    case unknown
}

public struct ReferenceImagesCapability: Codable, Hashable, Sendable {
    public let mode: ControlMode
    public let required: Bool
    public let maxCount: Int?
    public let primaryIsTarget: Bool
    public let sourceRelation: ReferenceSourceRelation
    public let reason: String?
    public let weight: FloatControl?
}

public struct GenerationDefaults: Codable, Hashable, Sendable {
    public let width: Int
    public let height: Int
    public let steps: Int
    public let guidance: Double
    public let frames: Int?
    public let fps: Int?
    public let negativePrompt: String?
}
