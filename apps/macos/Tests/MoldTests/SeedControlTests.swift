import MoldClient
import SwiftUI
import Testing

@testable import Mold

struct SeedControlTests {
    @MainActor @Test func fullSeedFitsNarrowAndWideControlsWithoutClipping() throws {
        var draft = RenderDraft()
        draft.seed = UInt64.max
        draft.locksSeed = true
        var heights: [Int: Int] = [:]
        for width in [240, 320, 440] {
            let renderer = ImageRenderer(content: SeedControl(draft: .constant(draft)))
            renderer.scale = 1
            renderer.proposedSize = .init(width: CGFloat(width), height: nil)
            let image = try #require(renderer.cgImage)
            #expect(image.width <= width)
            heights[width] = image.height
        }
        #expect(try #require(heights[240]) > #require(heights[440]))
    }

    @Test func choosingFixedWithoutASeedChoosesOneOnce() {
        var draft = RenderDraft()
        SeedControl.select(.fixed, in: &draft, newSeed: 42)
        #expect(draft.locksSeed)
        #expect(draft.seed == 42)
        #expect(RenderRequest.one(draft, model: "fixture").seed == 42)
        SeedControl.select(.fixed, in: &draft, newSeed: 99)
        #expect(draft.seed == 42)
    }

    @Test func randomModeOmitsTheRememberedSeedAndFixedRestoresIt() {
        var draft = RenderDraft()
        draft.seed = UInt64.max
        draft.locksSeed = true
        SeedControl.select(.random, in: &draft, newSeed: 42)
        #expect(!draft.locksSeed)
        #expect(draft.seed == UInt64.max)
        #expect(SeedControl.Mode.resolve(draft) == .random)
        #expect(RenderRequest.one(draft, model: "fixture").seed == nil)
        SeedControl.select(.fixed, in: &draft, newSeed: 42)
        #expect(draft.seed == UInt64.max)
        #expect(RenderRequest.one(draft, model: "fixture").seed == UInt64.max)
    }

    @Test func newSeedChangesTheFixedStartingSeedWithoutChangingBatchSemantics() {
        var draft = RenderDraft()
        draft.seed = 12
        SeedControl.pickNewSeed(in: &draft, newSeed: 99)
        #expect(SeedControl.Mode.resolve(draft) == .fixed)
        #expect(draft.seed == 99)
        #expect(RenderRequest.batch(draft, model: "fixture", copies: 3, randomBase: 7).map(\.seed) == [99, 100, 101])
        SeedControl.select(.random, in: &draft, newSeed: 42)
        #expect(RenderRequest.batch(draft, model: "fixture", copies: 3, randomBase: 7).map(\.seed) == [7, 8, 9])
    }

    @Test func anOldLockedDraftWithoutASeedDoesNotClaimToBeFixed() {
        var draft = RenderDraft()
        draft.locksSeed = true
        #expect(SeedControl.Mode.resolve(draft) == .random)
        #expect(RenderRequest.one(draft, model: "fixture").seed == nil)
    }
}
