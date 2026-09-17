import CoreGraphics
import Foundation

/// One dab of the brush, in the SOURCE image's own pixel coordinates.
///
/// Storing points in source-pixel space rather than canvas space means the
/// drawing view never has to reconcile a resize or a window change against
/// strokes that were captured at a different scale -- `MaskRender` renders
/// straight from these into a bitmap sized to the source.
public struct MaskStroke: Hashable, Sendable {
    public var points: [CGPoint]
    public var radius: Double
    public var erases: Bool

    public init(points: [CGPoint], radius: Double, erases: Bool) {
        self.points = points
        self.radius = radius
        self.erases = erases
    }
}

/// A bounded stack of strokes, plus whether the whole mask reads inverted.
///
/// Bounded at 32 because a mask is a few dozen dabs and an unbounded stack of
/// point arrays over a 4K source is megabytes of undo nobody asked for. The
/// web and desktop editors cap at 20 (`desktop/src/lib/maskEditor.ts:31-34`);
/// 32 is the same idea with room for a careful hand.
public struct MaskStrokes: Hashable, Sendable {
    public static let limit = 32

    public private(set) var strokes: [MaskStroke]
    public private(set) var isInverted: Bool

    public init(strokes: [MaskStroke] = [], isInverted: Bool = false) {
        self.strokes = strokes
        self.isInverted = isInverted
    }

    /// Appends a stroke, dropping the oldest once the stack is at `limit`.
    public mutating func add(_ stroke: MaskStroke) {
        strokes.append(stroke)
        if strokes.count > Self.limit {
            strokes.removeFirst(strokes.count - Self.limit)
        }
    }

    /// Removes the most recent stroke. `false` on an empty stack, and leaves
    /// it untouched -- there is nothing to undo into.
    @discardableResult
    public mutating func undo() -> Bool {
        guard !strokes.isEmpty else { return false }
        strokes.removeLast()
        return true
    }

    /// Empties the stack. Leaves `isInverted` as it was: clearing the paint
    /// is not the same question as which colour means "repaint".
    public mutating func clear() {
        strokes.removeAll()
    }

    public mutating func invert() {
        isInverted.toggle()
    }

    public var isEmpty: Bool { strokes.isEmpty && !isInverted }
}

public enum MaskBrush {
    public static let sizes: [Double] = [4, 8, 16, 32, 64, 128, 256]
    public static let defaultSize: Double = 32

    /// `[` and `]` step through `sizes`, never a free scalar: a size that is
    /// not on the list cannot be reached twice. `size` need not already be on
    /// the list -- the nearest entry is where stepping starts from.
    public static func stepped(_ size: Double, by delta: Int) -> Double {
        let nearestIndex = sizes.indices.min { lhs, rhs in
            abs(sizes[lhs] - size) < abs(sizes[rhs] - size)
        } ?? 0
        let target = Swift.min(Swift.max(nearestIndex + delta, 0), sizes.count - 1)
        return sizes[target]
    }
}
