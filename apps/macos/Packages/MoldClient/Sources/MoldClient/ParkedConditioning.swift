import Foundation

/// Conditioning a draft is holding for a recipe that cannot read it.
///
/// Switching to a text-to-video model and back used to DESTROY a picture
/// somebody had dragged in -- `RenderDraft+Recipe.swift`'s `adopting` used to
/// set the field to `nil` outright with no way back. Parking is the studio's
/// own behaviour for a staged identity photo
/// (`studio/lib/identityConditioning.test.ts:196-215`), and there is no
/// reason a source image deserves less care than a face. See
/// `RenderDraft+Park.swift` for the reconciliation rule this struct exists
/// to hold state for.
public struct ParkedConditioning: Hashable, Sendable {
    public var sourceImage: String?
    public var sourceImageName: String?
    public var editImages: [String] = []
    public var maskImage: String?
    public var identity: IdentityConditioning?
    public var control: ControlConditioning?
    public var loras: [LoraChoice] = []

    public init() {}
}
