import CoreGraphics
import MoldClient
import SwiftUI
import Testing

@testable import Mold

@MainActor
struct PromptPanelLayoutTests {
    /// Render the actual capsule, including its flexible height frame. Checking
    /// only PromptTuck's alignment misses a panel centered inside that frame.
    @Test(arguments: [CGFloat(300), CGFloat(900)])
    func capsuleRendersAtBottom(height: CGFloat) throws {
        let hosts = HostStore(hosts: []) { _ in FakeBackend(host: MoldHost(name: "test", baseURL: URL(string: "http://test")!)) }
        let controller = GenerateController(hosts: hosts, defaults: ConfigStore(hosts: hosts))
        let panel = PromptPanel(
            recipe: nil, draft: .constant(RenderDraft()), model: nil, host: nil,
            destination: .constant(.generate), submit: {}, cancel: {}, stopAll: {},
            maxBatch: 1, chainLimits: nil, maxHeight: height)
            .environment(controller)
            .environment(ReuseStore(hosts: hosts))
            .environment(ExpandStore())
            .environment(PromptHistoryStore(hosts: hosts))
            .frame(width: 760, height: height)
        let renderer = ImageRenderer(content: panel)
        renderer.scale = 1
        let image = try #require(renderer.cgImage)
        let width = image.width
        var pixels = [UInt8](repeating: 0, count: width * image.height * 4)
        let context = try #require(CGContext(
            data: &pixels, width: width, height: image.height, bitsPerComponent: 8,
            bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
        // Drawing a CGImage directly gives top-down bitmap rows.
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: image.height))
        var occupiedRows: [Int] = []
        for row in 0..<image.height {
            for column in 20..<(width - 20) {
                let alpha = pixels[(row * width + column) * 4 + 3]
                if alpha > 200 {
                    occupiedRows.append(row)
                    break
                }
            }
        }
        let bottom = try #require(occupiedRows.last)
        #expect(bottom >= image.height - 3)
        let top = try #require(occupiedRows.first)
        #expect(top > image.height / 2)
    }
}
