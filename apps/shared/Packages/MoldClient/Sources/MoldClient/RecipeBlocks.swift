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

/// Where a recipe's DEFAULT canvas comes from once references are staged
/// (`ReferenceCanvasRule`, `generation_profile.rs`).
public enum ReferenceCanvasRule: String, OpenWireEnum {
    /// Qwen Image 2.1: the last reference's aspect at upstream's fixed area
    /// (`ReferenceCanvas.lastReference`).
    case lastReference = "last-reference"
    /// A rule added after this build. Treated as no rule: resizing a canvas
    /// by a rule this build cannot read would be a guess.
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
    /// ADDITIVE: absent on every recipe but Qwen Image 2.1, and on an older
    /// host. Absence means the canvas never follows the references.
    public var canvas: ReferenceCanvasRule? = nil
    /// The containers a reference may arrive in (`ImageInputFormat`). Absent
    /// is the server's legacy PNG-and-JPEG set; read it through
    /// `acceptedFormats`, never directly.
    public var formats: [String]? = nil

    /// `ImageInputFormat::LEGACY`: what a recipe that advertises no list
    /// accepts.
    public static let legacyFormats = ["png", "jpeg"]

    public var acceptedFormats: [String] { formats ?? Self.legacyFormats }
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
