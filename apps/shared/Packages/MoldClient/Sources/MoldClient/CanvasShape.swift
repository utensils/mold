import Foundation

/// How a recipe's canvas is offered: aspect and size menus from its own
/// ladder, a fixed size, "from the source", or nothing at all. Shared by the
/// Mac's `ShapeControl` and the iPhone's Shape chip, so the two offer exactly
/// the same sizes for the same model.
public enum CanvasShape: Equatable, Sendable {
    public struct Menus: Equatable, Sendable {
        /// The aspect the current size belongs to ("3:2").
        public let aspect: String
        public let aspects: [AspectGroup]
        /// Sizes offered for that aspect -- plus the current size when it is
        /// off the ladder (a reused print), so it stays visible and chosen.
        public let sizes: [SizePreset]
        public let isOnLadder: Bool
    }

    case menus(Menus)
    /// The one size this recipe renders ("1024 × 1024").
    case fixed(String)
    /// The source picture decides the size.
    case fromSource
    /// No canvas at all (audio, a mesh from a picture).
    case hidden

    public static func resolve(_ resolution: ResolutionProfile, width: Int, height: Int) -> CanvasShape {
        guard resolution.hasCanvas else { return .hidden }
        if resolution.domain == .sourceDriven { return .fromSource }
        guard let groups = resolution.aspectGroups, !groups.isEmpty else {
            return .fixed("\(width) × \(height)")
        }
        if let group = groups.first(where: { $0.presets.contains { $0.width == width && $0.height == height } }) {
            return .menus(Menus(aspect: group.id, aspects: groups, sizes: group.presets, isOnLadder: true))
        }
        let aspect = gcdAspect(width: width, height: height)
        let extra = SizePreset(id: "\(width)x\(height)", width: width, height: height)
        let sizes = (groups.first { $0.id == aspect }?.presets ?? []) + [extra]
        return .menus(Menus(aspect: aspect, aspects: groups, sizes: sizes, isOnLadder: false))
    }

    /// Switching aspect keeps roughly the same pixel count: the size in the
    /// new group closest in area to the current one.
    public static func size(in group: AspectGroup, near width: Int, _ height: Int) -> SizePreset? {
        let target = width * height
        return group.presets.min { abs($0.width * $0.height - target) < abs($1.width * $1.height - target) }
    }

    static func gcdAspect(width: Int, height: Int) -> String {
        func gcd(_ a: Int, _ b: Int) -> Int { b == 0 ? a : gcd(b, a % b) }
        let divisor = max(gcd(width, height), 1)
        return "\(width / divisor):\(height / divisor)"
    }
}
