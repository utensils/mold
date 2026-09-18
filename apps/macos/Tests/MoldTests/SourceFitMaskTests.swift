import CoreGraphics
import Foundation
import ImageIO
import MoldClient
import Testing
import UniformTypeIdentifiers

@testable import Mold

/// A painted mask across a re-fit.
///
/// **Fails today**: `refit()` ended with `draft.media.maskImage =
/// padding?.base64EncodedString()`, and for the default `crop-fill` there are
/// no pad bands -- so nudging the canvas preset after painting an inpaint mask
/// deleted it with no warning, as did every re-appearance of the Source well.
/// Studio and desktop COMPOSE it (`sourceFitCanvas.ts:78-96`,
/// `sourceFitPreprocess.ts:104-107`).
@MainActor
struct SourceFitMaskTests {
    /// A mask with a white square in its middle, at `size`.
    private func paintedMask(_ size: Int) -> Data {
        let context = CGContext(
            data: nil, width: size, height: size, bitsPerComponent: 8, bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGImageAlphaInfo.none.rawValue)!
        context.setFillColor(gray: 0, alpha: 1)
        context.fill(CGRect(x: 0, y: 0, width: size, height: size))
        context.setFillColor(gray: 1, alpha: 1)
        context.fill(CGRect(x: size / 4, y: size / 4, width: size / 2, height: size / 2))
        return SourceFitRender.encodePNG(context.makeImage()!)!
    }

    /// Reads one pixel's grey level, 0...255.
    private func grey(_ data: Data, x: Int, y: Int) -> Int? {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else { return nil }
        let context = CGContext(
            data: nil, width: image.width, height: image.height, bitsPerComponent: 8,
            bytesPerRow: image.width, space: CGColorSpaceCreateDeviceGray(),
            bitmapInfo: CGImageAlphaInfo.none.rawValue)!
        context.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
        guard let pixels = context.data else { return nil }
        return Int(pixels.assumingMemoryBound(to: UInt8.self)[y * image.width + x])
    }

    /// A crop-fill re-fit adds no bands -- and must therefore leave the
    /// painted mask exactly where it was rather than clearing it.
    @Test func acropFillRefitKeepsThePaintedMask() async {
        let painted = paintedMask(64)
        let transform = SourceFitTransform.resolve(
            source: (128, 64), target: (64, 64), policy: .default)
        #expect(transform.maskPadding.isEmpty)

        let composed = await SourceFitRender.mask(existing: painted, transform: transform)
        let result = try! #require(composed)
        // The painted middle survives; the corner is still preserving.
        #expect(grey(result, x: 32, y: 32) ?? 0 > 200)
        #expect(grey(result, x: 2, y: 2) ?? 255 < 50)
    }

    /// A `pad-repaint` adds its bands ON TOP of what was painted, never
    /// instead of it.
    @Test func padRepaintAddsItsBandsToThePaintedMask() async {
        let painted = paintedMask(64)
        let transform = SourceFitTransform.resolve(
            source: (128, 64), target: (64, 64), policy: .padRepaint)
        #expect(transform.maskPadding.isEmpty == false)

        let composed = await SourceFitRender.mask(existing: painted, transform: transform)
        let result = try! #require(composed)
        // The top band is repainted...
        #expect(grey(result, x: 32, y: 2) ?? 0 > 200)
        // ...and the painted middle is still there.
        #expect(grey(result, x: 32, y: 32) ?? 0 > 200)
    }

    /// No mask and no bands is no mask -- a recipe with nothing to repaint
    /// must not grow an all-black one.
    @Test func nothingPaintedAndNothingPaddedIsStillNoMask() async {
        let transform = SourceFitTransform.resolve(
            source: (128, 64), target: (64, 64), policy: .default)
        #expect(await SourceFitRender.mask(existing: nil, transform: transform) == nil)
    }
}
