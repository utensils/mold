import Foundation
import MoldClient
import Testing

@testable import Mold

/// S6b: `MediaWell`, keyframes, an extend continuation, and the two
/// conditioning wells -- tested on the pure functions the views switch on,
/// the same way `ClipTests` tests S6a's.
@MainActor
struct ClipMediaTests {
    // MARK: - Overlap: absent by default, snapped to the recipe's own grid

    @Test func anOverlapNobodySetIsAbsentFromTheWire() throws {
        let auto = try Self.recipe("recipe-ltx2.json", "auto")
        var draft = RenderDraft().adopting(auto, isNewModel: true)
        draft.media.settingExtend(video: "AAAA", name: "clip.mp4")
        #expect(draft.media.extendOverlapFrames == nil)
        #expect(draft.request(model: "ltx-2.5-22b-dev:bf16").extendVideo == "AAAA")
        #expect(draft.request(model: "ltx-2.5-22b-dev:bf16").extendOverlapFrames == nil)
    }

    /// 8k+1 on LTX-2's fixture, 4k+1 on wan's -- read generically off each
    /// recipe's own `temporal.frames.step`, no family named.
    @Test func anOverlapSnapsToTheRecipesOwnGrid() throws {
        for fixture in ["recipe-ltx2.json", "recipe-wan.json"] {
            let recipe = try Self.recipe(fixture, fixture == "recipe-ltx2.json" ? "auto" : "default")
            let temporal = try #require(recipe.temporal)
            let draft = RenderDraft().adopting(recipe, isNewModel: true)

            let snapped = draft.snappedOverlap(20, temporal: temporal)
            #expect((snapped - 1) % temporal.frames.step == 0)
            #expect(snapped < (draft.frames ?? temporal.frames.default))
        }
    }

    @Test func anOverlapAtOrAboveTheFrameCountIsRefused() throws {
        let auto = try Self.recipe("recipe-ltx2.json", "auto")
        let temporal = try #require(auto.temporal)
        var draft = RenderDraft().adopting(auto, isNewModel: true)
        draft.frames = 10

        let snapped = draft.snappedOverlap(1000, temporal: temporal)
        #expect(snapped < 10)
        #expect((snapped - 1) % temporal.frames.step == 0)
    }

    // MARK: - Keyframes and an extend park each other

    @Test func keyframesAndAnExtendParkEachOther() {
        var draft = RenderDraft()
        draft.media.addingKeyframe(KeyframeCondition(frame: 0, image: "AAAA"))
        #expect(draft.media.keyframes.count == 1)

        draft.media.settingExtend(video: "BBBB", name: "clip.mp4")
        #expect(draft.media.extendVideo == "BBBB")
        #expect(draft.media.keyframes.isEmpty)
        #expect(draft.media.parked.keyframes.count == 1)

        draft.media.addingKeyframe(KeyframeCondition(frame: 8, image: "CCCC"))
        #expect(draft.media.extendVideo == nil)
        #expect(draft.media.parked.extendVideo == "BBBB")
        // The keyframe parked when extend won is restored alongside the new one.
        #expect(draft.media.keyframes.map(\.frame) == [0, 8])
    }

    @Test func anExtendParksTheSourceImage() {
        var draft = RenderDraft()
        draft.media.sourceImage = "AAAA"
        draft.media.sourceImageName = "a.png"

        draft.media.settingExtend(video: "BBBB", name: "clip.mp4")
        #expect(draft.media.sourceImage == nil)
        #expect(draft.media.sourceImageName == nil)
        #expect(draft.media.parked.sourceImage == "AAAA")

        // The request-time belt refuses to send both even if the source
        // well was used again afterward with no knowledge of the extend.
        draft.media.sourceImage = "STALE"
        #expect(draft.request(model: "m").sourceImage == nil)
        #expect(draft.request(model: "m").extendVideo == "BBBB")
    }

    // MARK: - A keyframe past the clip's length is refused

    @Test func aKeyframePastTheClipsLengthIsRefused() throws {
        let auto = try Self.recipe("recipe-ltx2.json", "auto")
        let temporal = try #require(auto.temporal)
        let snapped = KeyframeTable.snappedFrame(10_000, temporal: temporal, frames: 121)
        #expect(snapped < 121)
        let atZeroFrames = KeyframeTable.snappedFrame(5, temporal: temporal, frames: 0)
        #expect(atZeroFrames == 0)
    }

    // MARK: - The Clip group shows for a temporal recipe with no audio

    @Test func theClipGroupShowsForATemporalRecipeWithoutAudio() {
        let extendOnly = FakeFixtures.recipeCapabilities(supportsAudio: false, supportsExtend: true)
        #expect(ClipGroup.isShown(capabilities: extendOnly))

        let keyframesOnly = FakeFixtures.recipeCapabilities(supportsAudio: false, keyframes: true)
        #expect(ClipGroup.isShown(capabilities: keyframesOnly))

        let nothing = FakeFixtures.recipeCapabilities(supportsAudio: false)
        #expect(ClipGroup.isShown(capabilities: nothing) == false)
    }

    // MARK: - Fixture loading, same path `ClipTests` uses

    private static func recipe(_ name: String, _ id: String) throws -> GenerationRecipe {
        let fixtures = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent() // Tests/MoldTests
            .deletingLastPathComponent() // Tests
            .deletingLastPathComponent() // apps/macos
            .appending(path: "Packages/MoldClient/Tests/MoldClientTests/Fixtures")
        let data = try Data(contentsOf: fixtures.appending(path: name))
        let set = try MoldJSON.decoder.decode(GenerationProfileSet.self, from: data)
        return try #require(set.recipe(named: id))
    }
}
