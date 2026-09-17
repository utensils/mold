import Foundation

/// Where a source picture lands on the canvas, and how big it is drawn.
///
/// Port of `resolveSourceFitTransform` (`studio/lib/sourceFit.ts:175-227`).
/// Pure: the pixels are pushed elsewhere, so a test asks the geometry a
/// question without a bitmap.
public struct SourceFitTransform: Hashable, Sendable {
    public let outputWidth: Int
    public let outputHeight: Int
    public let drawWidth: Int
    public let drawHeight: Int
    /// May be NEGATIVE on a crop: the picture is drawn larger than the canvas
    /// and its edges fall outside.
    public let offsetX: Int
    public let offsetY: Int
    /// Whether the bands this transform adds are to be PAINTED into the mask,
    /// so the model repaints them. True for `pad-repaint` alone.
    public let maskPaddedPixels: Bool

    /// True when the source already fills the canvas exactly -- there is
    /// nothing to draw and the original bytes ship untouched.
    public var isIdentity: Bool {
        drawWidth == outputWidth && drawHeight == outputHeight && offsetX == 0 && offsetY == 0
    }

    public static func resolve(
        source: (width: Int, height: Int), target: (width: Int, height: Int), policy: SourceFit
    ) -> SourceFitTransform {
        let fit = policy.effective
        let outputWidth = target.width
        let outputHeight = target.height
        guard source.width > 0, source.height > 0, outputWidth > 0, outputHeight > 0,
              fit.mode != .lanczosResize
        else {
            return SourceFitTransform(
                outputWidth: outputWidth, outputHeight: outputHeight,
                drawWidth: outputWidth, drawHeight: outputHeight,
                offsetX: 0, offsetY: 0, maskPaddedPixels: false)
        }

        let sourceRatio = Double(source.width) / Double(source.height)
        let targetRatio = Double(outputWidth) / Double(outputHeight)
        let crop = fit.mode == .cropFill
        // Crop takes the LARGER scale so the canvas is filled; pad takes the
        // smaller so the whole picture is kept.
        let scale = crop
            ? (targetRatio > sourceRatio
                ? Double(outputWidth) / Double(source.width)
                : Double(outputHeight) / Double(source.height))
            : (targetRatio < sourceRatio
                ? Double(outputWidth) / Double(source.width)
                : Double(outputHeight) / Double(source.height))
        let drawWidth = Int((Double(source.width) * scale).rounded())
        let drawHeight = Int((Double(source.height) * scale).rounded())
        let availableX = outputWidth - drawWidth
        let availableY = outputHeight - drawHeight
        var alignX: SourceFitAlignX?
        var alignY: SourceFitAlignY?
        if case let .cropFill(x, y) = fit { alignX = x; alignY = y }
        return SourceFitTransform(
            outputWidth: outputWidth, outputHeight: outputHeight,
            drawWidth: drawWidth, drawHeight: drawHeight,
            offsetX: crop ? -alignOffset(-availableX, alignX?.rawValue)
                          : Int((Double(availableX) / 2).rounded()),
            offsetY: crop ? -alignOffset(-availableY, alignY?.rawValue)
                          : Int((Double(availableY) / 2).rounded()),
            maskPaddedPixels: fit.mode == .padRepaint)
    }

    /// Absent alignment is CENTRED (`sourceFit.ts:166-173`).
    private static func alignOffset(_ available: Int, _ align: String?) -> Int {
        switch align {
        case "left", "top": 0
        case "right", "bottom": available
        default: Int((Double(available) / 2).rounded())
        }
    }

    /// The bands a `pad-repaint` adds, for the mask to cover. Port of
    /// `maskPaddingRectangles` (`sourceFit.ts:229-259`).
    public var maskPadding: [SourceFitRect] {
        guard maskPaddedPixels else { return [] }
        let left = Swift.max(0, offsetX)
        let top = Swift.max(0, offsetY)
        let right = Swift.max(0, outputWidth - (offsetX + drawWidth))
        let bottom = Swift.max(0, outputHeight - (offsetY + drawHeight))
        var rects: [SourceFitRect] = []
        if top > 0 { rects.append(.init(x: 0, y: 0, width: outputWidth, height: top)) }
        if bottom > 0 {
            rects.append(.init(x: 0, y: outputHeight - bottom,
                               width: outputWidth, height: bottom))
        }
        if left > 0 {
            rects.append(.init(x: 0, y: top, width: left, height: outputHeight - top - bottom))
        }
        if right > 0 {
            rects.append(.init(x: outputWidth - right, y: top,
                               width: right, height: outputHeight - top - bottom))
        }
        return rects.filter { $0.width > 0 && $0.height > 0 }
    }
}

public struct SourceFitRect: Hashable, Sendable {
    public let x: Int
    public let y: Int
    public let width: Int
    public let height: Int
}
