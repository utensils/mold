import Foundation
import Testing
@testable import MoldClient

struct VideoDurationLengthTests {
    private func grid() throws -> VideoLengthGrid {
        VideoLengthGrid.resolve(temporal: try VideoLengthGridTests.h3Temporal(), fps: nil,
                                limits: try VideoLengthGridTests.h3Limits())
    }

    /// Duration trims to the exact length, so its slider is seconds, not frames.
    @Test func theSliderShowsAndWritesExactSeconds() throws {
        let grid = try grid()
        var draft = VideoDurationDraft(seconds: "14", scenesEnabled: false)
        #expect(draft.lengthMilliseconds == 14_000)
        #expect(VideoLengthGrid.label(milliseconds: draft.lengthMilliseconds ?? 0) == "14.0 s")
        #expect(!draft.fit(to: grid, prompt: "x") && draft.seconds == "14", "an off-grid restored value is not rewritten")
        draft.setLength(tenths: 72, on: grid)
        #expect(draft.seconds == "7.2")
        draft.setLength(tenths: 999, on: grid)
        #expect(draft.seconds == "14.3")
        draft.setLength(tenths: 0, on: grid)
        #expect(draft.seconds == "1")
    }

    @Test func longerVideoStartsFromThePromptAndTheCurrentLength() throws {
        let grid = try grid()
        var draft = VideoDurationDraft(seconds: "14.375", scenesEnabled: false)
        draft.setLonger(true, prompt: "a heron lands", on: grid)
        #expect(draft.isLonger)
        #expect(draft.scenes.map(\.prompt) == ["a heron lands"])
        #expect(draft.scenes.map(\.seconds) == ["14.3"], "capped at the always-one-clip scene")
        draft.addScene(on: grid)
        #expect(draft.scenes.count == 2 && draft.seconds == "19.3")
        draft.setScene(draft.scenes[1].id, tenths: 999, on: grid)
        #expect(draft.scenes[1].seconds == "14.3" && draft.seconds == "28.6")
        draft.scenes[1].prompt = "it takes off"
        let request = try draft.request(generation: .init(prompt: "x", model: "h3", width: 1, height: 1, steps: 1, guidance: 0))
        #expect(request.scenes?.map(\.durationMs) == [14_300, 14_300] && request.durationMs == 28_600)
        draft.setLonger(false, prompt: "", on: grid)
        #expect(!draft.isLonger && draft.seconds == "14.3", "a total past one generation clamps to the slider's end")
        var short = VideoDurationDraft(seconds: "9.25", scenesEnabled: false)
        short.setLonger(true, prompt: "p", on: grid)
        short.setLonger(false, prompt: "", on: grid)
        #expect(short.seconds == "9.3", "a length the single slider can show is kept to its tenth")
        draft.setLonger(true, prompt: "ignored", on: grid)
        #expect(draft.scenes.count == 2, "turning it back on keeps the authored scenes")
    }

    @Test func aSavedLengthTheSlidersCannotShowBecomesEqualScenes() throws {
        let grid = try grid()
        var single = VideoDurationDraft(seconds: "40", scenesEnabled: false)
        let splitSingle = single.fit(to: grid, prompt: "storm")
        #expect(splitSingle)
        #expect(single.scenes.map(\.seconds) == ["13.333", "13.333", "13.334"])
        #expect(single.scenes.allSatisfy { $0.prompt == "storm" } && single.seconds == "40")
        var scenes = VideoDurationDraft(seconds: "35", scenes: [.init(prompt: "a", seconds: "5"), .init(prompt: "b", seconds: "30")])
        let splitScene = scenes.fit(to: grid, prompt: "unused")
        #expect(splitScene)
        #expect(scenes.scenes.map(\.prompt) == ["a", "b", "b", "b"] && scenes.seconds == "35")
        var fits = VideoDurationDraft(seconds: "14.375", scenesEnabled: false)
        let untouched = fits.fit(to: grid, prompt: "x")
        #expect(!untouched)
    }
}

extension VideoDurationLengthTests {
    private var generation: GenerateRequest { .init(prompt: "x", model: "h3", width: 1, height: 1, steps: 1, guidance: 0) }

    @Test func anEmptyScenePromptNamesTheSceneAndALongOneSaysTooLong() throws {
        var draft = VideoDurationDraft(seconds: "10", scenes: [.init(prompt: "a", seconds: "5"), .init(prompt: "  ", seconds: "5")])
        #expect(throws: SequenceDraftError.invalid("Describe scene 2.")) { try draft.request(generation: generation) }
        draft.scenes[1].prompt = String(repeating: "a", count: 77_001)
        #expect(throws: SequenceDraftError.invalid("Scene 2's prompt is too long. Shorten it.")) {
            try draft.request(generation: generation)
        }
        // Phrased so the shared byte-count rewriter never touches it.
        #expect(UserFacingError.describe(SequenceDraftError.invalid("Scene 2's prompt is too long. Shorten it."))
            == "Scene 2's prompt is too long. Shorten it.")
    }

    /// Both Duration sliders end on a 0.1 s step, so the readout is exact.
    @Test func durationSlidersEndOnAFlooredTenth() throws {
        let grid = try grid()
        #expect(grid.singleMaxTenths == 143 && grid.sceneMaxTenths == 143)
        var draft = VideoDurationDraft(seconds: "5", scenesEnabled: false)
        draft.setLength(tenths: grid.singleMaxTenths, on: grid)
        #expect(draft.seconds == "14.3")
    }
}
