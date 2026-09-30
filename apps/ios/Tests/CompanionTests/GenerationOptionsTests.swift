import CoreGraphics
import Foundation
import MoldClient
import Testing

@testable import MoldCompanion

@MainActor
struct GenerationOptionsTests {
    @Test(arguments: [(16, 9), (1, 1), (9, 16), (20, 11), (11, 20)])
    func aspectIconsPreserveTheirActualRatio(dimensions: (Int, Int)) {
        let size = AspectRatioIcon.size(width: dimensions.0, height: dimensions.1, bound: 24)
        #expect(abs(size.width / size.height - Double(dimensions.0) / Double(dimensions.1)) < 0.0001)
        #expect(max(size.width, size.height) == 24)
    }

    @Test func sourceFittingUsesTheDesktopDefaultAndCapabilityGates() {
        #expect(SourceFit.default == .cropFill(alignX: .center, alignY: .center))
        #expect(SourceFitOptions.policy(for: .padRepaint, supportsMask: false) == .default)
        #expect(SourceFitOptions.policy(for: .padFit, supportsMask: false) == .padFit)
    }

    @Test func seedsAreRandomUntilExplicitlyLocked() {
        var draft = RenderDraft()
        #expect(!draft.locksSeed)
        #expect(RenderRequest.one(draft, model: "fixture").seed == nil)
        draft.seed = 42
        #expect(RenderRequest.one(draft, model: "fixture").seed == nil)
        draft.locksSeed = true
        #expect(RenderRequest.one(draft, model: "fixture").seed == 42)
    }
}
