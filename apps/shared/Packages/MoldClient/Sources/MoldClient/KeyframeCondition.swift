import Foundation

/// One pinned still in LTX-2's keyframe interpolation, mirroring
/// `KeyframeCondition` (`types.rs:2367-2377`).
///
/// `image` is base64, like every byte field on the wire. `name` is
/// display-only provenance -- the engine ignores it, and saved metadata
/// retains only a sanitized name and content digest.
///
/// Not `Identifiable`: a freshly added row has no image yet, so an id
/// derived from its content would collide with every other empty row.
/// `KeyframeTable` iterates by array index instead.
public struct KeyframeCondition: Codable, Hashable, Sendable {
    public var frame: Int
    public var image: String
    public var name: String?

    public init(frame: Int, image: String, name: String? = nil) {
        self.frame = frame
        self.image = image
        self.name = name
    }
}
