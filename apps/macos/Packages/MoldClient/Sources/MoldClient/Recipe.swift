import Foundation

/// Whether this recipe reads the prompt at all.
///
/// `ignored` is real: the Hunyuan3D family has no text encoder anywhere, so a
/// prompt box there is furniture. The app hides it rather than collecting text
/// nothing will read.
public enum PromptRequirement: String, OpenWireEnum {
    case required
    case optional
    case ignored
    case unknown
}

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

public struct RecipeCapabilities: Codable, Hashable, Sendable {
    /// Absent means required -- see `PromptCapability.assumedRequired`.
    public let prompt: PromptCapability?
    public let negativePrompt: FeatureControl?
    public let output: OutputCapabilities?
    public let referenceImages: ReferenceImagesCapability?
    /// Absent is deliberately NOT a `true` nobody wrote: it means an older
    /// host, and the caller falls back to its own legacy predicate.
    public let supportsStrength: Bool?
    public let supportsLora: Bool?
    public let supportsIdentity: Bool?
    public let supportsSequence: Bool?
    public let supportsExtend: Bool?
    public let supportsAudio: Bool?

    public var promptRequirement: PromptRequirement {
        (prompt ?? .assumedRequired).mode
    }
}

/// One way of running a model. Most models have exactly one; LTX-2 has several
/// and picks by pipeline.
public struct GenerationRecipe: Codable, Hashable, Sendable, Identifiable {
    public let id: String
    public let label: String
    public let defaults: GenerationDefaults
    public let resolution: ResolutionProfile
    public let steps: IntegerControl
    public let guidance: FloatControl
    public let capabilities: RecipeCapabilities
}

/// The set of recipes a model advertises, and which one is the default.
public struct GenerationProfileSet: Codable, Hashable, Sendable {
    public let schemaVersion: Int
    public let profileId: String
    public let profileHash: String
    public let defaultRecipeId: String
    public let recipes: [GenerationRecipe]

    /// The recipe a plain request runs. Falls back to the first rather than
    /// returning nil: a profile with recipes always has one that works, and a
    /// mismatched id is the server's bug, not a reason to show no controls.
    public var defaultRecipe: GenerationRecipe? {
        recipes.first { $0.id == defaultRecipeId } ?? recipes.first
    }
}
