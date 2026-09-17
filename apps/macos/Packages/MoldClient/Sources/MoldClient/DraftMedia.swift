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

    /// What the CURRENT recipe cannot take, held so it comes back
    /// (`DraftMedia+Park.swift`).
    public var parked = ParkedConditioning()

    public init() {}
}
