import CoreGraphics
import SwiftUI
import Testing
@testable import Mold

@MainActor
struct InspectorFieldLayoutTests {
    @Test(arguments: [200, 280, 440])
    func controlsFillAvailableWidthAndStackWhenNeeded(width: Int) throws {
        let renderer = ImageRenderer(content: InspectorFieldLayout {
            Color(red: 1, green: 0, blue: 0).frame(width: 80, height: 12)
            Color(red: 0, green: 1, blue: 0).frame(height: 22)
        }.frame(width: CGFloat(width)).background(.black))
        renderer.scale = 1
        let image = try #require(renderer.cgImage)
        let stacks = width < 80 + Int(InspectorFieldLayout.horizontalGap + InspectorFieldLayout.minimumControlWidth)
        #expect(image.width == width)
        #expect(image.height == (stacks ? 39 : 22))
        var pixels = [UInt8](repeating: 0, count: image.width * image.height * 4)
        let context = try #require(CGContext(data: &pixels, width: image.width, height: image.height,
            bitsPerComponent: 8, bytesPerRow: image.width * 4, space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
        context.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
        var greenColumns = Set<Int>()
        for y in 0..<image.height {
            for x in 0..<image.width {
                let offset = (y * image.width + x) * 4
                if pixels[offset + 1] > 150 && pixels[offset] < 50 { greenColumns.insert(x) }
            }
        }
        #expect(greenColumns.min() == (stacks ? 0 : 92))
        #expect(greenColumns.max() == width - 1)
    }
}
