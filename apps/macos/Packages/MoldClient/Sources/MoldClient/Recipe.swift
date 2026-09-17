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

/// A repeatable adapter input and its immutable stack limit.
/// `generation_profile.rs:389-395`.
public struct AdapterControl: Codable, Hashable, Sendable {
    public let mode: ControlMode
    public let maxCount: Int
    public let reason: String?
}

/// Which pipeline a recipe asks the server to run.
///
/// `nil` on the `auto` recipe: the server picks. Never spelled `"auto"` --
/// that string is a display key, not a wire value, and the app must not send
/// it as though it were one.
public struct RecipeSelector: Codable, Hashable, Sendable {
    public let pipeline: String?
}

/// Wan's own sampler controls. Decoded so the block round-trips; nothing in
/// M4 reads it. A later milestone that offers the distill-strength slider or
/// the first/last-frame toggle reads this rather than adding a second copy.
public struct WanRecipeCapabilities: Codable, Hashable, Sendable {
    public let mode: ControlMode
    public let supportsDistillStrength: Bool
    public let supportsFirstLastFrame: Bool
    public let firstLastFrameMinFrames: Int?
    public let reason: String?
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
    public let supportsControlnet: Bool?
    public let supportsIdentity: Bool?
    public let supportsSequence: Bool?
    public let supportsExtend: Bool?
    public let supportsAudio: Bool?
    /// Absent means this recipe reads a source image -- the field is omitted
    /// for image families. See `RecipeCapabilities.readsSourceImage`.
    public let sourceImage: SourceImageCapability?
    public let lora: AdapterControl?
    public let controlnet: AdapterControl?
    public let mask: FeatureControl?
    public let keyframes: FeatureControl?
    public let audio: FeatureControl?
    public let sourceVideo: FeatureControl?
    /// Absent, never `[]`, on a recipe with no scheduler choice
    /// (`skip_serializing_if = "Vec::is_empty"`).
    public let schedulers: [String]?
    /// Wan's sampler controls. Nothing in M4 reads this -- see the type's own
    /// doc comment.
    public let wanRecipe: WanRecipeCapabilities?

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
    /// Present only for the families that make a clip.
    public let temporal: TemporalProfile?
    public let capabilities: RecipeCapabilities
    /// What to send the server to pick this recipe. `nil` on `auto`.
    public let requestSelector: RecipeSelector?
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

    /// One named recipe, or nil when this profile does not advertise it.
    public func recipe(named id: String) -> GenerationRecipe? {
        recipes.first { $0.id == id }
    }
}
