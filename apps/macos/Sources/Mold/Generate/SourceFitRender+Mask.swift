import CoreGraphics
import Foundation
import MoldClient

// Keeping a painted mask pixel-aligned with the source it is over.
//
// The mask is an OPAQUE 8-bit grayscale PNG, black preserving and white
// repainting, sized to the source picture's own pixels (`MaskRender`,
// `img_utils.rs:128-130`). A mask is painted over the FITTED source -- which
// is what the well shows and what ships -- so while the fit holds, the two are
// aligned by construction and nothing here runs.
//
// When the fit MOVES -- a different canvas, a different policy, a different
// picture -- the painted mask describes pixels that are no longer there. It is
// replaced rather than rescaled: a rescaled mask is a plausible-looking lie
// about which pixels somebody chose, and the engine's own Lanczos resize would
// then smear it across the render. What survives is only what the fit itself
// implies: the bands a `pad-repaint` added, which are white by definition
// (`maskPaddingRectangles`, `sourceFit.ts:229-259`).
extension SourceFitRender {
    /// The mask a freshly fitted source starts with: the `pad-repaint` bands,
    /// or `nil` where the fit adds none. `nil` means the draft carries no
    /// mask at all, which is what a recipe with no padded pixels wants.
    static func paddingMask(_ transform: SourceFitTransform) async -> Data? {
        let padding = transform.maskPadding
        guard !padding.isEmpty else { return nil }
        return await Task.detached(priority: .userInitiated) {
            let width = max(1, transform.outputWidth)
            let height = max(1, transform.outputHeight)
            guard let context = CGContext(
                data: nil, width: width, height: height, bitsPerComponent: 8,
                bytesPerRow: 0, space: CGColorSpaceCreateDeviceGray(),
                bitmapInfo: CGImageAlphaInfo.none.rawValue) else { return nil }
            // Black preserves. Everything the fit did not add stays preserved,
            // which is the safe reading: repainting pixels nobody asked about
            // destroys them.
            context.setFillColor(gray: 0, alpha: 1)
            context.fill(CGRect(x: 0, y: 0, width: width, height: height))
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
