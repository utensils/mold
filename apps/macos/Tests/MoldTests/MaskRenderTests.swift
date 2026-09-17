import AppKit
import Foundation
import MoldClient
import Testing

@testable import Mold

/// `MaskRender` asserted on real decoded bytes, not on the drawing calls
/// that produced them -- "white = repaint" is a contract with the server,
/// so the test reads the same channel `to_luma8()` would.
@MainActor
struct MaskRenderTests {
    private func corner(of data: Data) -> UInt8? {
        guard let rep = NSBitmapImageRep(data: data) else { return nil }
        guard let color = rep.colorAt(x: 0, y: 0) else { return nil }
        return UInt8((color.whiteComponent * 255).rounded())
    }

    @Test func anEmptyNonInvertedStackRendersOpaqueBlack() {
        let data = MaskRender.png(MaskStrokes(), size: CGSize(width: 8, height: 8))
        #expect(data != nil)
        #expect(corner(of: data!) == 0)
    }

    @Test func anEmptyInvertedStackRendersOpaqueWhite() {
        var strokes = MaskStrokes()
        strokes.invert()
        let data = MaskRender.png(strokes, size: CGSize(width: 8, height: 8))
        #expect(data != nil)
        #expect(corner(of: data!) == 255)
    }

    @Test func aStrokeAtTheCentrePaintsThatPixelWhite() {
        var strokes = MaskStrokes()
        strokes.add(MaskStroke(points: [CGPoint(x: 4, y: 4)], radius: 8, erases: false))
        let data = MaskRender.png(strokes, size: CGSize(width: 8, height: 8))
        #expect(data != nil)
        guard let png = data, let rep = NSBitmapImageRep(data: png),
              let color = rep.colorAt(x: 4, y: 4)
        else {
            Issue.record("expected a decodable pixel")
            return
        }
        #expect(color.whiteComponent > 0.9)
    }
}
