import Foundation

/// What kind of render a prompt is being written for. The wire spelling is
/// KEBAB-case (`text-to-image`), not snake_case: `#[serde(rename_all =
/// "kebab-case")]` at `crates/mold-core/src/types.rs:204`. `MoldJSON`'s key
/// strategy converts KEYS, never values, so an underscored raw value here
/// would degrade every task to `.unknown` in silence.
public enum ExpandTask: String, OpenWireEnum {
    case textToImage = "text-to-image"
    case textToVideo = "text-to-video"
    case imageToVideo = "image-to-video"
    case videoToVideo = "video-to-video"
    case retake
    case keyframeInterpolation = "keyframe-interpolation"
    case audioDrivenVideo = "audio-driven-video"
    case referenceToAudioVideo = "reference-to-audio-video"
    case textToAudio = "text-to-audio"
    case unknown
}

/// A dimension a remix may vary. `types.rs:648-656`, kebab-case, all single
/// words so the spelling is its own name.
public enum RemixDimension: String, OpenWireEnum {
    case composition, camera, lighting, setting, mood, movement, style
    case unknown
}

/// Which prompt the remix started from. `types.rs:694-699`; `direct` is the
/// server's `#[default]`.
public enum RemixSourceKind: String, OpenWireEnum {
    case original, current, direct
    case unknown
}

/// `types.rs:702-707`.
public enum PromptTransformOperation: String, OpenWireEnum {
    case expand, remix
    case unknown
}

/// Provenance for a prompt that a transform produced, carried on the render
/// that uses it. `types.rs:709-728`.
public struct PromptTransformProvenance: Codable, Hashable, Sendable {
    public let operation: PromptTransformOperation
    public let rootPrompt: String?
    public let sourcePrompt: String
    public let sourceKind: RemixSourceKind
    public let task: ExpandTask
    public let dimensions: [RemixDimension]

    public init(
        operation: PromptTransformOperation,
        rootPrompt: String? = nil,
        sourcePrompt: String,
        sourceKind: RemixSourceKind = .direct,
        task: ExpandTask,
        dimensions: [RemixDimension] = []
    ) {
        self.operation = operation
        self.rootPrompt = rootPrompt
        self.sourcePrompt = sourcePrompt
        self.sourceKind = sourceKind
        self.task = task
        self.dimensions = dimensions
    }
}

/// Request to rewrite a short prompt into a generation-aware one.
/// `types.rs:730-759`.
///
/// Two rules this file carries: an `OpenWireEnum` may never be ENCODED as
/// `.unknown` -- `task` is the first open enum here riding a REQUEST body,
/// so it is `ExpandTask?` and the caller omits it rather than sending
/// `"unknown"`, which is a 422 on a route whose whole job is optional. And
/// `style`/`context` are not modelled: `style` is a preset label this app has
/// no presets for, and `ExpandContext` is identity/canvas/frames/references
/// for M4's controls; the server infers the task from the family when both
/// are absent (`routes.rs:4036-4045`), which is the answer M3 wants.
public struct ExpandRequest: Codable, Sendable {
    public var prompt: String
    public var modelFamily: String
    public var variations: Int
    public var task: ExpandTask?

    public init(
        prompt: String,
        modelFamily: String = "flux",
        variations: Int = 1,
        task: ExpandTask? = nil
    ) {
        self.prompt = prompt
        self.modelFamily = modelFamily
        self.variations = variations
        self.task = task
    }
}

/// Request for subject-preserving prompt alternatives. A SEPARATE endpoint
/// from Expand so an older host fails closed rather than silently expanding.
/// `types.rs:782-813`. `style`/`context` are not modelled, same reasoning as
/// `ExpandRequest`.
public struct RemixRequest: Codable, Sendable {
    public var sourcePrompt: String
    public var rootPrompt: String?
    public var sourceKind: RemixSourceKind
    public var modelFamily: String
    public var variations: Int
    public var task: ExpandTask?
    /// Empty means the server's task-aware default set.
    public var dimensions: [RemixDimension]

    public init(
        sourcePrompt: String,
        rootPrompt: String? = nil,
        sourceKind: RemixSourceKind = .direct,
        modelFamily: String = "flux",
        variations: Int = 3,
        task: ExpandTask? = nil,
        dimensions: [RemixDimension] = []
    ) {
        self.sourcePrompt = sourcePrompt
        self.rootPrompt = rootPrompt
        self.sourceKind = sourceKind
        self.modelFamily = modelFamily
        self.variations = variations
        self.task = task
        self.dimensions = dimensions
    }
}
