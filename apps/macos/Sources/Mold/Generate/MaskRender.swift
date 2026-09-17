import CoreGraphics
import Foundation
import ImageIO
import MoldClient
import UniformTypeIdentifiers

/// The mask as the server decodes it.
///
/// An OPAQUE 8-bit grayscale PNG: black preserves, white repaints
/// (`crates/mold-inference/src/img_utils.rs:128-130`). Alpha is NOT a channel
/// here -- `to_luma8()` (`img_utils.rs:147`) reads RGB and ignores it, so a
/// transparent white pixel would repaint. The canvas is composited onto black
/// before it is encoded, which is the same reason desktop's `invertPixels`
/// promotes a transparent pixel to opaque (`desktop/src/lib/maskEditor.ts`).
///
/// Sized to the source image's own pixels. The engine resizes to the LATENT
/// grid with Lanczos3 (`img_utils.rs:141-145`), so the size need not be exact
/// -- but the ASPECT must match or the mask stretches across the picture.
enum MaskRender {
    static func png(_ strokes: MaskStrokes, size: CGSize) -> Data? {
        let width = max(1, Int(size.width.rounded()))
        let height = max(1, Int(size.height.rounded()))
        let colorSpace = CGColorSpaceCreateDeviceGray()
        guard let context = CGContext(
            data: nil, width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: 0,
            space: colorSpace, bitmapInfo: CGImageAlphaInfo.none.rawValue
        ) else { return nil }

        // Preserve is black (0), repaint is white (1). Painted exactly as
        // drawn -- `isInverted` is applied afterward as a whole-canvas flip,
        // never a per-stroke swap, mirroring desktop's `invertPixels`, which
        // runs over the fully rendered canvas at export time rather than
        // changing what each stroke means while painting.
        context.setFillColor(gray: 0, alpha: 1)
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))

        for stroke in strokes.strokes {
            context.setFillColor(gray: stroke.erases ? 0 : 1, alpha: 1)
            let diameter = CGFloat(stroke.radius) * 2
            for point in stroke.points {
                let rect = CGRect(x: point.x - CGFloat(stroke.radius),
                                  y: point.y - CGFloat(stroke.radius),
                                  width: diameter, height: diameter)
                context.fillEllipse(in: rect)
            }
        }

        if strokes.isInverted {
            invert(context, width: width, height: height)
        }

        guard let image = context.makeImage() else { return nil }
        let data = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(
            data, UTType.png.identifier as CFString, 1, nil
        ) else { return nil }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else { return nil }
        return data as Data
    }

    /// Flips every sample in place: `255 - value`. Walked by the context's
    /// own `bytesPerRow`, which CoreGraphics may pad past `width`.
    private static func invert(_ context: CGContext, width: Int, height: Int) {
        guard let buffer = context.data else { return }
        let bytesPerRow = context.bytesPerRow
        let bytes = buffer.assumingMemoryBound(to: UInt8.self)
        for row in 0 ..< height {
            let rowStart = row * bytesPerRow
            for column in 0 ..< width {
                let index = rowStart + column
                bytes[index] = 255 - bytes[index]
            }
        }
    }
}
