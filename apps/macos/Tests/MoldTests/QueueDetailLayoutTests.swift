import Foundation
import MoldClient
import SwiftUI
import Testing
@testable import Mold

@MainActor
struct QueueDetailLayoutTests {
    @Test func longPromptsWrapIntoTheAvailableColumnWithoutFixedHeightClipping() throws {
        let text = String(repeating: "A detailed landscape with mountains, trees and a winding river. ", count: 8)
        let metadata = try MoldJSON.decoder.decode(OutputMetadata.self, from:
            JSONSerialization.data(withJSONObject: ["prompt": text]))
        let group = try #require(PrintDetails.groups(for: metadata).first { $0.title == "Prompt" })
        func measure(_ width: CGFloat) throws -> CGSize {
            let renderer = ImageRenderer(content: QueueDetailFacts(group: group).frame(width: width))
            renderer.scale = 1
            let image = try #require(renderer.cgImage)
            return CGSize(width: image.width, height: image.height)
        }
        let narrow = try measure(400)
        let wide = try measure(640)
        #expect(narrow.width == 400)
        #expect(wide.width == 640)
        #expect(narrow.height > wide.height)
        #expect(wide.height > 60)
    }
}
