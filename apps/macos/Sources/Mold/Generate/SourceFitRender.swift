import CoreGraphics
import Foundation
import ImageIO
import MoldClient
import UniformTypeIdentifiers

/// Putting a source picture onto the canvas, in pixels.
///
/// The server records `source_fit` but never reads it -- the fitting happens
/// HERE, before the bytes ship (`types.rs:3268-3273`) -- so this is the half
/// of the contract that actually has to be right.
///
/// `nonisolated` throughout, like `PictureImport`: the app defaults to
/// MainActor isolation and a 50 MP still redrawn on every canvas change would
/// stall the window.
nonisolated enum SourceFitRender {
    /// The fitted picture, or `nil` when there is nothing to do -- the source
    /// already fills the canvas, or the bytes will not decode. `nil` means
    /// "ship the original", never "fail silently": the ORIGINAL is always a
    /// picture the engine reads, and the engine resizes what it is given.
    static func fit(
        _ data: Data, name: String, target: (width: Int, height: Int), policy: SourceFit
    ) async -> ImportedPicture? {
        await Task.detached(priority: .userInitiated) {
            guard let source = CGImageSourceCreateWithData(data as CFData, nil),
                  let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
            else { return nil }
            let transform = SourceFitTransform.resolve(
                source: (image.width, image.height), target: target, policy: policy)
            guard !(image.width == target.width && image.height == target.height) else { return nil }
            guard let png = draw(image, transform) else { return nil }
            return ImportedPicture(
                encoded: png.base64EncodedString(),
                name: (name as NSString).deletingPathExtension + ".png",
                data: png)
        }.value
    }

    /// The transform a given source lands on, for the mask to follow. Cheap:
    /// reads the header only, never the pixels.
    static func transform(
        of data: Data, target: (width: Int, height: Int), policy: SourceFit
    ) -> SourceFitTransform? {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
              let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil)
                  as? [CFString: Any],
              let width = properties[kCGImagePropertyPixelWidth] as? Int,
              let height = properties[kCGImagePropertyPixelHeight] as? Int
        else { return nil }
        return SourceFitTransform.resolve(source: (width, height), target: target, policy: policy)
    }

    private static func draw(_ image: CGImage, _ transform: SourceFitTransform) -> Data? {
        guard transform.outputWidth > 0, transform.outputHeight > 0,
              let context = CGContext(
                  data: nil, width: transform.outputWidth, height: transform.outputHeight,
                  bitsPerComponent: 8, bytesPerRow: 0,
                  space: CGColorSpaceCreateDeviceRGB(),
                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { return nil }
        // Padded pixels are TRANSPARENT, not black: a `pad-repaint` marks them
        // in the mask and the model paints them, and a `pad-fit` says plainly
        // that nothing was there.
        context.interpolationQuality = .high
        // Core Graphics has its origin at the bottom; the transform's `offsetY`
        // is measured from the TOP, as every canvas API in the fleet measures
        // it, so the two are reconciled here and nowhere else.
        let flippedY = transform.outputHeight - transform.offsetY - transform.drawHeight
        context.draw(image, in: CGRect(x: transform.offsetX, y: flippedY,
                                       width: transform.drawWidth, height: transform.drawHeight))
        guard let fitted = context.makeImage() else { return nil }
        return encodePNG(fitted)
    }

    static func encodePNG(_ image: CGImage) -> Data? {
        let data = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(
            data, UTType.png.identifier as CFString, 1, nil) else { return nil }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else { return nil }
        return data as Data
    }
}
