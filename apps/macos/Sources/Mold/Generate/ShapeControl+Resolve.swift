import Foundation
import MoldClient

/// Aspect + size, drawn as two menus under the prompt instead of one buried
/// "1024 × 1024" menu (M8 design, decision 3). This is the pure resolution
/// behind it; the view (`ShapeControl.swift`, S2) is `ShapeControl`'s primary
/// declaration -- a zero-case enum can never be constructed, so it cannot
/// also conform to `View`, and this file only extends the struct declared
/// there.
extension ShapeControl {
    struct Shape: Equatable {
        /// The aspect the current size belongs to -- the group's own id when
        /// the size is one of its presets, else `width:height` reduced by
        /// gcd, the way the server labels a group (`generation_profile.rs:2623`).
        let aspect: String
        /// Every advertised aspect, in the server's order.
        let aspects: [AspectGroup]
        /// The sizes offered under `aspect`: that group's presets, plus the
        /// current size as an extra last row when it is not one of them (a
        /// Reuse can carry an off-ladder size; it is shown, never snapped).
        let sizes: [SizePreset]
        let isOnLadder: Bool
    }

    enum Presentation: Equatable {
        case menus(Shape)
        case fixed(String)
        case fromSource
        case hidden
    }

    static func resolve(resolution: ResolutionProfile, width: Int, height: Int) -> Presentation {
        guard resolution.hasCanvas else { return .hidden }
        if resolution.domain == .sourceDriven { return .fromSource }
        guard let groups = resolution.aspectGroups, !groups.isEmpty else {
            // Verbatim digits, no thousands separator.
            return .fixed("\(width) × \(height)")
        }
        if let group = groups.first(where: { onLadder(width: width, height: height, group: $0) }) {
            return .menus(Shape(aspect: group.id, aspects: groups, sizes: group.presets, isOnLadder: true))
        }
        let aspect = gcdAspect(width: width, height: height)
        let extra = SizePreset(id: "\(width)x\(height)", width: width, height: height)
        let sizes = (groups.first { $0.id == aspect }?.presets ?? []) + [extra]
        return .menus(Shape(aspect: aspect, aspects: groups, sizes: sizes, isOnLadder: false))
    }

    /// The preset in `group` nearest the current pixel count -- picking 16:9
    /// from 1024x1024 lands on 1024x576, not on the group's smallest.
    static func size(in group: AspectGroup, nearWidth: Int, height: Int) -> SizePreset? {
        let target = nearWidth * height
        return group.presets.min {
            abs($0.width * $0.height - target) < abs($1.width * $1.height - target)
        }
    }

    private static func onLadder(width: Int, height: Int, group: AspectGroup) -> Bool {
        group.presets.contains { $0.width == width && $0.height == height }
    }

    private static func gcdAspect(width: Int, height: Int) -> String {
        func gcd(_ a: Int, _ b: Int) -> Int { b == 0 ? a : gcd(b, a % b) }
        let divisor = max(gcd(width, height), 1)
        return "\(width / divisor):\(height / divisor)"
    }
}
