import Foundation

/// What the Generate pane is holding before anything is submitted.
///
/// The draft is always reconciled against the chosen model's recipe: the
/// server owns what each control may be, and a value carried over from a
/// different model is only kept when the new recipe would accept it.
public struct RenderDraft: Hashable, Sendable {
    public var prompt: String = ""
    public var negativePrompt: String = ""
    public var width: Int = 1024
    public var height: Int = 1024
    /// `didSet` clamps a staged identity's `startStep` below the new count --
    /// the identity and steps controls live in different places on screen,
    /// so dragging Steps down after Start step was set must not silently
    /// arm a 422 (`identity.rs:560-566`). `applyIdentity` clamps again at
    /// request time as a belt; this is what keeps the ON-SCREEN bound in
    /// sync as it happens rather than only at submit.
    public var steps: Int = 20 {
        didSet {
            guard var conditioning = media.identity else { return }
            let range = Identity.startStepRange(steps: steps)
            let clamped = Swift.min(Swift.max(conditioning.startStep, range.lowerBound), range.upperBound)
            guard clamped != conditioning.startStep else { return }
            conditioning.startStep = clamped
            media.identity = conditioning
        }
    }
    public var guidance: Double = 3.5
    public var batchSize: Int = 1
    /// nil means "let the host pick", which is the default and what makes
    /// repeated renders differ.
    public var seed: UInt64?
    public var locksSeed: Bool = false
    /// Clip length, for the families that make one.
    public var frames: Int?
    public var fps: Int?
    /// LTX-2's chosen way of running, echoed straight from the adopted
    /// recipe's own `request_selector.pipeline` -- `nil` on `auto`, which
    /// means "let the server pick" and must never be spelled as the string
    /// `"auto"` (`RenderDraft+Recipe.swift`'s `adopting`).
    public var pipeline: String?
    /// The person's audio choice, apart from whether the current recipe can
    /// honour it. `nil` is untouched, whose default is sound ON for a capable
    /// video recipe. Moving through a still or video-only recipe must not turn
    /// an authored OFF into ON (or an authored ON into OFF), so availability
    /// is kept separately below.
    public var preferredAudio: Bool?
    /// The recipe answer last reconciled by `adopting`. This is draft state so
    /// every request path -- one render, a batch, placement and an auto-chain
    /// -- reads the same effective capability without accepting a recipe as a
    /// second argument.
    public var supportsAudio: Bool = false
    /// H3 and text-to-audio always render audio. Their fixed capability keeps
    /// a parked preference but never turns the effective request off.
    public var requiresAudio: Bool = false
    /// Whether the current family exposes the two LTX audio choices in the
    /// Clip group. A fixed-audio recipe can support sound without a switch.
    public var offersAudioControl: Bool = false
    /// The recipe supports sound but this checkpoint row explicitly reports
    /// missing audio assets.
    public var audioUnavailableForModel: Bool = false
    /// LTX video resolves an omitted audio flag from the output format, so
    /// even an unavailable checkpoint must send an explicit false.
    public var usesOptionalAudioBranch: Bool = false
    /// What the Sound switch shows. Writing it records a real preference;
    /// recipe reconciliation changes only `supportsAudio` and therefore parks
    /// that preference while sound is unavailable.
    public var enableAudio: Bool {
        get { supportsAudio && (requiresAudio || (preferredAudio ?? true)) }
        set { preferredAudio = newValue }
    }
    /// Skips the audio branch on a video render. Never sent while
    /// `VideoOnlyPolicy` finds a conflict -- see `RenderDraft.videoOnlyInputs`.
    public var videoOnly: Bool = false
    /// Only meaningful with a source image to apply it to
    /// (`RenderDraft+Request.swift` reads `media.sourceImage` to decide).
    /// Stays here rather than on `DraftMedia` -- it is a numeric control like
    /// `guidance`, not a conditioning input, even though it rides with one.
    public var strength: Double = 0.75
    /// The recipe's advertised answer for denoise strength. `nil` means an
    /// older host, where the pre-profile behaviour remains compatible; false
    /// fixes the WIRE value to 1 without overwriting the authored slider value
    /// parked in `strength`.
    public var supportsStrength: Bool?
    /// What this render is conditioned on besides its prompt and numbers --
    /// the still, references, mask, identity, adapters, ControlNet,
    /// keyframes, extend continuation, audio file and source video, plus the
    /// parking rule that protects all of them across a recipe switch
    /// (`DraftMedia.swift`).
    public var media = DraftMedia()
    /// The sampler controls a recipe advertises -- the solver, CFG++, wan's
    /// flow shift and distill strengths, LTX-2's guidance overrides -- and
    /// the parking rule that protects them across a recipe switch
    /// (`AdvancedControls.swift`). Every one is absent from the request until
    /// somebody moves it.
    public var advanced = AdvancedControls()
    /// WHY the canvas holds the size it holds -- recorded when somebody acts,
    /// never inferred from the size afterwards (#1166, `CanvasIntent`).
    public var canvasIntent: CanvasIntent = .modelDefault

    /// Filing: title, tags and a collection to file the finished print
    /// under, gated on `canOrganize` at the call site.
    public var title: String = ""
    public var tags: [String] = []
    public var collectionName: String?
    /// Whether the title's own words are folded into tags too. Mirrors
    /// `mold_core::organization::compose_client_tags`; the rule itself is
    /// applied at request time, not here (`ClientTags`, M3 S4).
    public var autoTagTitle: Bool = true
    public var outputFormat: String?
    public var upscaleModel: String?
    /// `false` publishes the print and moves it straight to the trash. `true`
    /// is the server's own default, so a request never has to say so.
    public var savesToGallery: Bool = true
    /// Provenance for a prompt an expand/remix wand produced.
    public var originalPrompt: String?
    public var promptTransform: PromptTransformProvenance?

    public init() {}
}

public extension IntegerControl {
    func clamp(_ value: Int) -> Int { Swift.min(Swift.max(value, min), max) }
}

public extension FloatControl {
    func clamp(_ value: Double) -> Double { Swift.min(Swift.max(value, min), max) }
}

// `RenderDraft(reusing:)` lives in `RenderDraft+Reuse.swift`.
