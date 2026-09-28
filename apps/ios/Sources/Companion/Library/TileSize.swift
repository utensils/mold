import SwiftUI

/// The Library's tile sizes, smallest to largest, the way Photos steps:
/// a pinch walks the ladder live, one size for roughly every third more
/// (or less) of pinch, and never past either end. Minimum widths are at
/// Large text; the grid scales them with Dynamic Type.
enum TileSize: String, CaseIterable, Identifiable {
    case tiny, small, medium, large, huge
    var id: Self { self }

    var title: String {
        switch self {
        case .tiny: String(localized: "Tiny")
        case .small: String(localized: "Small")
        case .medium: String(localized: "Medium")
        case .large: String(localized: "Large")
        case .huge: String(localized: "Largest")
        }
    }

    /// About 7, 5, 3, 2 and 1 columns on an iPhone at Large text.
    var basePoints: CGFloat {
        switch self {
        case .tiny: 52
        case .small: 76
        case .medium: 112
        case .large: 170
        case .huge: 300
        }
    }

    func stepped(bigger: Bool) -> TileSize { offset(by: bigger ? 1 : -1) }

    func offset(by steps: Int) -> TileSize {
        let all = Self.allCases
        let index = all.firstIndex(of: self) ?? 2
        return all[max(0, min(all.count - 1, index + steps))]
    }

    /// Where a pinch that began at `start` has got to: one step per 35% of
    /// spread or squeeze, so a single long pinch can cross several sizes.
    /// `current` is the size on screen: it holds until the fingers move a
    /// tenth of a step past a boundary, so resting on one never flickers.
    static func pinched(from start: TileSize, magnification: CGFloat, current: TileSize? = nil) -> TileSize {
        guard magnification > 0 else { return start }
        let raw = log(magnification) / log(1.35)
        let target = start.offset(by: Int(raw.rounded(.towardZero)))
        guard let current, current != target else { return target }
        let all = allCases
        let shown = Double((all.firstIndex(of: current) ?? 0) - (all.firstIndex(of: start) ?? 0))
        // Within 1.1 steps of the size on screen: keep it.
        return abs(raw - shown) < 1.1 ? current : target
    }
}
