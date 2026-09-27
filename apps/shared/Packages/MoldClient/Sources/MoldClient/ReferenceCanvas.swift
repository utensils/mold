import Foundation
import ImageIO

/// The default canvas a `canvas: last-reference` recipe (Qwen Image 2.1)
/// takes from its references.
///
/// `GenerateRequest.width`/`height` are required, so the server cannot tell a
/// chosen size from a default one: the rule is a CLIENT rule and admission
/// stays exact. While the canvas intent is `modelDefault` every surface sizes
/// the canvas to the LAST reference's aspect at upstream's fixed 1024x1024
/// area, on the recipe's grid, clamped into its advertised bounds. A manual
/// canvas never moves.
///
/// This is `mold_core::validation::last_reference_canvas` exactly
/// (`validation.rs:1444-1580`), mirrored the way `studio/lib/referenceCanvas.ts`
/// mirrors it: `fitToTargetAreaTiesEven` is diffusers' `calculate_dimensions`
/// (`pipeline_qwenimage21.py:149-156`), rounded with Python's `round()` --
/// halves to EVEN -- so a 4225x4096 reference is 1024x1024 here as it is in
/// the engine, not 1056x1024. `clamp` is mold's deliberate divergence:
/// upstream caps nothing, so a panorama past about 7.3:1 would derive a
/// width admission refuses.
public enum ReferenceCanvas {
    /// `mold_core::validation::LAST_REFERENCE_CANVAS_AREA`: upstream's 1024².
    public static let area = 1024 * 1024

    /// Python's `round()` for a finite value: halves go to the even neighbour.
    public static func roundHalfToEven(_ value: Double) -> Int {
        Int(value.rounded(.toNearestOrEven))
    }

    /// `fit_to_target_area_ties_even(src_w, src_h, area, align)`, exactly.
    public static func fitToTargetAreaTiesEven(
        width: Int, height: Int, targetArea: Int, alignment: Int
    ) -> SourcePixels {
        let align = max(1, alignment)
        let ratio = Double(max(1, width)) / Double(max(1, height))
        let fittedWidth = (Double(targetArea) * ratio).squareRoot()
        let fittedHeight = fittedWidth / ratio
        // A degenerate aspect that rounds an axis to zero is lifted to one cell.
        func snap(_ value: Double) -> Int { max(1, roundHalfToEven(value / Double(align))) * align }
        return SourcePixels(width: snap(fittedWidth), height: snap(fittedHeight))
    }

    /// `clamp_canvas_to_limits`, exactly: integer cells only. The long side
    /// walks down one grid cell at a time from the axis ceiling, the short
    /// side follows by FLOORED proportion and is lifted to its minimum, until
    /// both ceilings hold. A canvas already inside is returned unchanged.
    public static func clamp(width: Int, height: Int, limits: ResolutionProfile) -> SourcePixels {
        let align = max(1, limits.alignment ?? 1)
        let maxPixels = limits.maxPixels ?? Int.max
        let axis = limits.maxAxisPixels
        func fits(_ w: Int, _ h: Int) -> Bool {
            w * h <= maxPixels && (axis.map { w <= $0 && h <= $0 } ?? true)
        }
        if fits(width, height) { return SourcePixels(width: width, height: height) }
        let landscape = width >= height
        let (long, short) = landscape ? (width, height) : (height, width)
        func minCells(_ pixels: Int?) -> Int { max(1, ((pixels ?? 0) + align - 1) / align) }
        let longMin = minCells(landscape ? limits.minWidth : limits.minHeight)
        let shortMin = minCells(landscape ? limits.minHeight : limits.minWidth)
        var longCells = max(1, long / align)
        if let axis { longCells = min(longCells, max(1, axis / align)) }
        while true {
            let shortCells = max(longCells * short / long, shortMin)
            let (lw, sh) = (longCells * align, shortCells * align)
            if fits(lw, sh) || longCells <= longMin {
                return landscape ? SourcePixels(width: lw, height: sh)
                    : SourcePixels(width: sh, height: lw)
            }
            longCells -= 1
        }
    }

    /// `last_reference_canvas(ref_w, ref_h, limits)`, exactly.
    public static func lastReference(width: Int, height: Int, limits: ResolutionProfile) -> SourcePixels {
        let fitted = fitToTargetAreaTiesEven(
            width: width, height: height, targetArea: area, alignment: limits.alignment ?? 1)
        return clamp(width: fitted.width, height: fitted.height, limits: limits)
    }

    /// The canvas the rule asks for, or nil to leave the canvas alone: no
    /// rule (or one newer than this build), a canvas somebody chose, an
    /// empty strip, or a last reference whose size is not known -- never
    /// guess. An empty strip answers nil rather than the recipe default
    /// because a restored draft hydrates through here too, and emptying the
    /// strip keeps the last reference's shape (`referenceCanvasSize`).
    public static func size(
        rule: ReferenceCanvasRule?, intent: CanvasIntent,
        references: [SourcePixels?], resolution: ResolutionProfile
    ) -> SourcePixels? {
        guard rule == .lastReference, intent == .modelDefault,
              let lastEntry = references.last, let last = lastEntry else { return nil }
        return lastReference(width: last.width, height: last.height, limits: resolution)
    }

    /// A staged reference's UPRIGHT size, read from its header with the EXIF
    /// orientation applied -- how the engine decodes it. A phone portrait is
    /// stored as landscape pixels plus `Orientation = 6`, and a canvas sized
    /// from the stored header would come out sideways.
    public static func uprightPixels(ofBase64 encoded: String) -> SourcePixels? {
        guard !encoded.isEmpty, let data = Data(base64Encoded: encoded),
              let source = CGImageSourceCreateWithData(data as CFData, nil),
              let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil)
                  as? [CFString: Any],
              let width = properties[kCGImagePropertyPixelWidth] as? Int,
              let height = properties[kCGImagePropertyPixelHeight] as? Int
        else { return nil }
        let orientation = properties[kCGImagePropertyOrientation] as? Int ?? 1
        // 5-8 are the four orientations that swap the axes.
        return (5 ... 8).contains(orientation)
            ? SourcePixels(width: height, height: width)
            : SourcePixels(width: width, height: height)
    }
}

public extension RenderDraft {
    /// Re-derives a `last-reference` recipe's default canvas from the strip.
    /// Called on every add, replace, reorder and removal, and at the end of
    /// `adopting` -- a no-op unless the canvas is still the model's
    /// (`canvasIntent == .modelDefault`) and the recipe advertises the rule.
    /// Only the LAST reference is read, so this decodes one header.
    mutating func followLastReference(recipe: GenerationRecipe?) {
        guard let recipe, let references = recipe.capabilities.referenceImages,
              references.canvas == .lastReference, canvasIntent == .modelDefault,
              let last = media.editImages.last else { return }
        guard let size = ReferenceCanvas.size(
            rule: references.canvas, intent: canvasIntent,
            references: [ReferenceCanvas.uprightPixels(ofBase64: last)],
            resolution: recipe.resolution) else { return }
        width = size.width
        height = size.height
    }
}
