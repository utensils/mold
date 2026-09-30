import CoreGraphics
import Foundation
import ImageIO
import Testing
import UniformTypeIdentifiers

@testable import MoldClient

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

    @Test func aSourceSpaceMaskIsCroppedWithThePicture() async throws {
        let context = try #require(CGContext(data: nil, width: 128, height: 64, bitsPerComponent: 8,
            bytesPerRow: 0, space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGImageAlphaInfo.none.rawValue))
        context.setFillColor(gray: 0, alpha: 1)
        context.fill(CGRect(x: 0, y: 0, width: 128, height: 64))
        context.setFillColor(gray: 1, alpha: 1)
        context.fill(CGRect(x: 8, y: 0, width: 16, height: 64))
        context.fill(CGRect(x: 64, y: 0, width: 8, height: 64))
        let image = try #require(context.makeImage())
        let painted = try #require(SourceFitRender.encodePNG(image))
        let transform = SourceFitTransform.resolve(source: (128, 64), target: (64, 64), policy: .default)
        let result = try #require(await SourceFitRender.mask(existing: painted, transform: transform, sourceSpace: true))
        #expect(grey(result, x: 8, y: 32) == 0, "the left painted strip was cropped away")
        #expect(grey(result, x: 34, y: 32) == 255, "the middle strip follows the cropped source")
        #expect(grey(result, x: 60, y: 32) == 0)
    }

    @Test func portraitJPEGOrientationSurvivesFitting() async throws {
        let context = try #require(CGContext(data: nil, width: 128, height: 64, bitsPerComponent: 8,
            bytesPerRow: 0, space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
        let image = try #require(context.makeImage())
        let jpeg = NSMutableData()
        let destination = try #require(CGImageDestinationCreateWithData(jpeg, UTType.jpeg.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(destination, image, [kCGImagePropertyOrientation: 6] as CFDictionary)
        #expect(CGImageDestinationFinalize(destination))
        let data = jpeg as Data
        #expect(PictureImport.pixelSize(of: data)?.width == 64)
        #expect(PictureImport.pixelSize(of: data)?.height == 128)
        let transform = try #require(SourceFitRender.transform(of: data, target: (64, 64), policy: .default))
        #expect(transform.drawWidth == 64)
        #expect(transform.drawHeight == 128)
        let fitted = try #require(await SourceFitRender.fit(data, name: "portrait.jpg", target: (32, 64), policy: .default))
        #expect(PictureImport.pixelSize(of: fitted.data)?.width == 32)
        #expect(PictureImport.pixelSize(of: fitted.data)?.height == 64)
    }

    /// No mask and no bands is no mask -- a recipe with nothing to repaint
    /// must not grow an all-black one.
    @Test func nothingPaintedAndNothingPaddedIsStillNoMask() async {
        let transform = SourceFitTransform.resolve(
            source: (128, 64), target: (64, 64), policy: .default)
        #expect(await SourceFitRender.mask(existing: nil, transform: transform) == nil)
    }
}
