import Foundation

/// What a render is conditioned on besides its prompt and numbers, and
/// what the current recipe cannot take, held so it comes back.
///
/// Split out of `RenderDraft` (S6c, M4 design): the prompt and the numeric
/// controls are one concern, and the still/references/mask/identity/
/// adapters/ControlNet/keyframes/extend/audio/source-video inputs -- plus
/// the parking rule that protects all of them across a recipe switch -- are
/// another. `RenderDraft` always reaches this through its own `media`
/// field; nothing here reads the prompt or the numeric controls.
public struct DraftMedia: Hashable, Sendable {
    /// A still to condition on, already base64-encoded, with the name the host
    /// should record for it.
    public var sourceImage: String?
    public var sourceImageName: String?
    /// The picture as it was PICKED, before any fit -- `sourceImage` above is
    /// what will actually ship, and is regenerated from this whenever the
    /// canvas or the policy moves. Held so a re-fit never compounds: fitting
    /// an already-fitted picture crops a crop.
    public var sourceImageOriginal: String?
    public var sourceImageOriginalName: String?
    /// How a source whose shape differs from the canvas is mapped onto it.
    /// Recorded on the request as provenance -- the engine never reads it,
    /// the fitting happens here (`types.rs:3268-3273`).
    public var sourceFit: SourceFit = .default
    /// Ordered reference images, base64. For a recipe whose first image is the
    /// Target, index 0 is that one.
    public var editImages: [String] = []
    public var referenceWeight: Double?
    /// A repaint mask over `sourceImage`, base64 PNG. Meaningless without a
    /// source, and dropped at request time when there is none
    /// (`validation.rs:3101-3107`).
    public var maskImage: String?
    /// Face-identity conditioning. One value whether it carries one
    /// photograph or four -- the wire shape is chosen at request time from
    /// the host's `multi_photo`, so `id_image` and `id_images` can never
    /// both be set (`IdentityConditioning.wire(maxPhotos:)`).
    public var identity: IdentityConditioning?
    /// The adapter stack, in the order it was added. Never written into the
    /// legacy singular `lora` field (`types.rs:3419-3444`) -- there is no
    /// Swift equivalent of it and there never will be.
    public var loras: [LoraChoice] = []
    /// ControlNet conditioning. Parked/restored the same way as every other
    /// conditioning input (`DraftMedia+Park.swift`) -- see
    /// `ControlConditioning`'s own doc comment for why both its halves are
    /// optional.
    public var control: ControlConditioning?
    // LTX-2 keyframe interpolation and continuation ("extend"). The
    // invariants -- mutually exclusive, an extend parks the source image,
    // overlap snapped to the recipe's own temporal grid -- live in
    // `DraftMedia+Clip.swift`, beside the pure functions that enforce them.
    public var keyframes: [KeyframeCondition] = []
    /// An existing clip to continue, base64. `extendVideoName` is display
    /// only -- there is no `extend_video_name` on the wire (`types.rs:2098-2102`).
    public var extendVideo: String?
    public var extendVideoName: String?
    /// Carryover pixel frames for the continuation. `nil` sends nothing, so
    /// the server fills in the family's own default (decision 12, M4
    /// design) -- see `RenderDraft.snappedOverlap`, which stays on
    /// `RenderDraft` because it reads the draft's own frame count.
    public var extendOverlapFrames: Int?
    /// Conditioning audio for LTX-2 audio-to-video, base64. `audioFileName`
    /// is display only -- no `audio_file_name` on the wire.
    public var audioFile: String?
    public var audioFileName: String?
    /// Reference video conditioning, base64. `sourceVideoName` is display
    /// only -- no `source_video_name` on the wire.
    public var sourceVideo: String?
    public var sourceVideoName: String?

    /// The image-conditioning layout the ADOPTED recipe projects, written by
    /// `reconcile(for:family:model:)` and read by the request builder and by
    /// every well. Kept on the draft rather than re-derived at each call site
    /// so the request that ships and the wells on screen can never disagree
    /// about which relation is in force.
    public var sourceMode: SourceImageMode = .single
    /// Which of the two EXCLUSIVE wells was written most recently. Only
    /// `sourceMode == .singleOrReferences` reads it; `nil` reads as the
    /// source well, which is what a restored print with both carries
    /// (`ExclusiveWells.resolve`).
    public var lastExclusiveWrite: ExclusiveWell?

    /// What the CURRENT recipe cannot take, held so it comes back
    /// (`DraftMedia+Park.swift`).
    public var parked = ParkedConditioning()

    public init() {}

    /// Which conditioning a request built from this media carries. The one
    /// question the request builder, the Strength control and the Mask row
    /// all ask -- never `sourceImage != nil`, which on an exclusive recipe
    /// answers for a well that is parked.
    public var requestConditioning: RequestConditioning {
        RequestConditioning.resolve(
            mode: sourceMode, hasSource: sourceImage != nil,
            referenceCount: editImages.count, lastWrite: lastExclusiveWrite)
    }

    /// Which exclusive well is parked right now, and the sentence it renders.
    /// `nil` on every other relation -- nothing parks there.
    public var exclusiveWells: ExclusiveWells? {
        guard sourceMode == .singleOrReferences else { return nil }
        return ExclusiveWells.resolve(
            hasSource: sourceImage != nil, referenceCount: editImages.count,
            lastWrite: lastExclusiveWrite)
    }
}
