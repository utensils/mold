import MoldClient
import SwiftUI
import Testing

@testable import Mold

/// The shared seed control's layout in the inspector. Its logic is
/// MoldClient's `SeedChoice` (`SeedChoiceTests`).
struct SeedControlTests {
    @MainActor @Test func fullSeedFitsNarrowAndWideControlsWithoutClipping() throws {
        var draft = RenderDraft()
        draft.seed = UInt64.max
        draft.locksSeed = true
        var heights: [Int: Int] = [:]
        for width in [240, 320, 440] {
            let renderer = ImageRenderer(content: NativeSeedControl(draft: .constant(draft)))
            renderer.scale = 1
            renderer.proposedSize = .init(width: CGFloat(width), height: nil)
            let image = try #require(renderer.cgImage)
            #expect(image.width <= width)
            heights[width] = image.height
        }
        #expect(try #require(heights[240]) > #require(heights[440]))
    }
}
