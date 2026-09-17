import Foundation

/// How a recipe decides what sizes it will accept.
public enum ResolutionDomain: String, OpenWireEnum {
    /// Any size within the bounds, on the alignment grid.
    case dynamic
    /// Only the advertised presets.
    case buckets
    /// The size comes from a supplied image or video, not from the user.
    case sourceDriven = "source-driven"
    /// The recipe has no canvas at all -- a mesh, for instance.
    case none
    case unknown
}

/// What happens to a size that isn't on an advertised bucket.
public enum OffBucketPolicy: String, OpenWireEnum {
    case reject
    case warn
    case unknown
}

public struct SizePreset: Codable, Hashable, Sendable, Identifiable {
    public let id: String
    public let width: Int
    public let height: Int
    /// `recommended`, or another tier the server may add later.
    public let tier: String?

    /// For an off-ladder size a client builds itself -- a Reuse can carry
    /// pixels no advertised preset names, and that size is still shown, not
    /// snapped to the nearest one (`ShapeControl.resolve`).
    public init(id: String, width: Int, height: Int, tier: String? = nil) {
        self.id = id
        self.width = width
        self.height = height
        self.tier = tier
    }

    public var label: String { "\(width) × \(height)" }
}

public struct AspectGroup: Codable, Hashable, Sendable, Identifiable {
    public let id: String
    public let label: String
    public let presets: [SizePreset]
}

public struct ResolutionProfile: Codable, Hashable, Sendable {
    public let domain: ResolutionDomain
    public let alignment: Int?
    public let minWidth: Int?
    public let minHeight: Int?
    public let maxPixels: Int?
    public let maxAxisPixels: Int?
    /// Absent means `reject`. A client that defaulted to `warn` would submit
    /// sizes the host is going to refuse.
    public let offBucket: OffBucketPolicy?
    public let aspectGroups: [AspectGroup]?

    /// True when this recipe puts a picture on a canvas at all.
    public var hasCanvas: Bool { domain != .none && domain != .unknown }

    public var presets: [SizePreset] { (aspectGroups ?? []).flatMap(\.presets) }
}
