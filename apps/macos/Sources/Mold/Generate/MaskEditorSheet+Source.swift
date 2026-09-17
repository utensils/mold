import AppKit
import ImageIO
import MoldClient

/// Decoding the source picture, and the scale between its own pixels and
/// the sheet's display canvas.
///
/// `MaskStroke` points are always stored in SOURCE pixels (see
/// `MaskEditorSheet`'s own doc comment); everything here is what converts
/// between that and the ~640x480 canvas the sheet actually draws.
extension MaskEditorSheet {
    /// Not `private`: `canvasSize` below reads it, and the main file's
    /// `canvas`/`body` size themselves against it indirectly through
    /// `canvasSize`.
    static let maxCanvasSize = CGSize(width: 640, height: 480)

    /// Not `private`: the main file's `.task` calls it.
    func loadSource() {
        guard let base64 = draft.sourceImage, let data = Data(base64Encoded: base64) else { return }
        source = NSImage(data: data)
        if let cgSource = CGImageSourceCreateWithData(data as CFData, nil),
           let properties = CGImageSourceCopyPropertiesAtIndex(cgSource, 0, nil) as? [CFString: Any],
           let width = properties[kCGImagePropertyPixelWidth] as? Int,
           let height = properties[kCGImagePropertyPixelHeight] as? Int {
            sourcePixelSize = CGSize(width: width, height: height)
        }
    }

    /// The display canvas, fit inside `maxCanvasSize` at the source's own
    /// aspect. Every gesture location is divided by `displayScale` going in
    /// and multiplied by it going out, so strokes always land in SOURCE
    /// pixels regardless of how big the sheet draws them.
    ///
    /// Not `private`: the main file's `body` and `draw(_:in:)` both read it.
    var canvasSize: CGSize {
        let aspect = sourcePixelSize.width / max(sourcePixelSize.height, 1)
        var size = Self.maxCanvasSize
        if aspect > size.width / size.height {
            size.height = size.width / aspect
        } else {
            size.width = size.height * aspect
        }
        return size
    }

    /// Not `private`, same reason.
    var displayScale: CGFloat {
        guard sourcePixelSize.width > 0 else { return 1 }
        return canvasSize.width / sourcePixelSize.width
    }
}
