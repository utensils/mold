import Foundation

/// ControlNet conditioning: a control picture, the installed adapter that
/// reads it, and how strongly it steers the denoise.
///
/// `image` and `model` are BOTH optional, unlike `IdentityConditioning`'s
/// photos -- the Refine group's picker and its picture well are one unit but
/// fill in whichever half was touched first (decision in the S5 report: a
/// person may choose the adapter before dropping a picture, or the other way
/// round), so a draft can be sitting on just one of the two while someone
/// works. `RenderDraft+Request.swift` is where that resolves: sent both or
/// neither, never one alone (`validation.rs:3079-3090`, a symmetric pair).
public struct ControlConditioning: Codable, Hashable, Sendable {
    /// Base64, as mold encodes every byte field on the wire.
    public var image: String?
    /// Display only, like `LoraChoice.name` -- there is no `control_image`
    /// name field on the wire.
    public var name: String?
    /// The installed adapter's model name, e.g. `controlnet-canny-sd15:fp16`.
    public var model: String?
    /// `types.rs:2751` (`default_control_scale`): 1.0. Only `< 0` is refused
    /// (`validation.rs:3090`-area) -- there is no server ceiling.
    public var scale: Double

    public init(image: String? = nil, name: String? = nil, model: String? = nil,
                scale: Double = Control.defaultScale) {
        self.image = image
        self.name = name
        self.model = model
        self.scale = scale
    }
}

/// ControlNet constants shared by the reading extensions and the draft.
public enum Control {
    public static let defaultScale = 1.0
    /// Presentation only -- the server refuses nothing above any value here,
    /// only below zero. Wide enough that a real render is still on-slider.
    public static let scaleRange: ClosedRange<Double> = 0 ... 2
}
