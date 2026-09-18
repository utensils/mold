import CoreGraphics
import Foundation
import ImageIO
import MoldClient

// Keeping a painted mask through a re-fit.
//
// The mask is an OPAQUE 8-bit grayscale PNG, black preserving and white
// repainting (`MaskRender`, `img_utils.rs:128-130`), painted over the FITTED
// source and therefore the size of the canvas.
//
// A re-fit COMPOSES it rather than replacing it, which is what
// `sourceFitCanvas.ts:78-96` (`buildMask`) and
// `desktop/src/lib/sourceFitPreprocess.ts:104-107` do: draw what was painted,
// then fill the bands the new fit added. It used to be REPLACED by the bands
// alone, so a crop-fill -- which adds none -- silently deleted an inpaint mask
// somebody had painted, on a canvas nudge and on every re-appearance of the
// Source well.
extension SourceFitRender {
    /// The mask for a freshly fitted source: what was painted, plus the bands
    /// a `pad-repaint` added. `nil` only when there is neither -- a recipe
    /// with nothing to repaint must not grow an all-black mask.
    static func mask(existing: Data?, transform: SourceFitTransform) async -> Data? {
        let padding = transform.maskPadding
        guard existing != nil || !padding.isEmpty else { return nil }
        return await Task.detached(priority: .userInitiated) {
            let width = max(1, transform.outputWidth)
            let height = max(1, transform.outputHeight)
            guard let context = CGContext(
                data: nil, width: width, height: height, bitsPerComponent: 8,
                bytesPerRow: 0, space: CGColorSpaceCreateDeviceGray(),
                bitmapInfo: CGImageAlphaInfo.none.rawValue) else { return nil }
            // Black preserves. Anything neither painted nor padded stays
            // preserved, which is the safe reading: repainting pixels nobody
            // asked about destroys them.
            context.setFillColor(gray: 0, alpha: 1)
            context.fill(CGRect(x: 0, y: 0, width: width, height: height))

            if let existing, let source = CGImageSourceCreateWithData(existing as CFData, nil),
               let painted = CGImageSourceCreateImageAtIndex(source, 0, nil) {
                // Drawn to the WHOLE canvas: this mask was painted over the
                // fitted source, so it is already in canvas space. A canvas
                // that has since changed size rescales it, which keeps the
                // painted region where the person put it relative to the
                // picture -- the alternative is throwing their work away.
                context.interpolationQuality = .none
                context.draw(painted, in: CGRect(x: 0, y: 0, width: width, height: height))
            }
            context.setFillColor(gray: 1, alpha: 1)
            for rect in padding {
                // The rectangles are measured from the TOP, as every canvas
                // API in the fleet measures them; Core Graphics' origin is at
                // the bottom, and the two are reconciled here and nowhere else.
                context.fill(CGRect(x: rect.x,
                                    y: transform.outputHeight - rect.y - rect.height,
                                    width: rect.width, height: rect.height))
            }
            guard let image = context.makeImage() else { return nil }
            return encodePNG(image)
        }.value
    }
}
