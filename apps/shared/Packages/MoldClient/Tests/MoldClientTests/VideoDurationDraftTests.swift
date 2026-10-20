import Testing
@testable import MoldClient

struct VideoDurationDraftTests {
    @Test func decimalSecondsHaveMillisecondPrecisionWithoutFloatingPointOrArbitraryCap() throws {
        #expect(try VideoDurationDraft.milliseconds("30") == 30_000)
        #expect(try VideoDurationDraft.milliseconds("0.001") == 1)
        #expect(try VideoDurationDraft.milliseconds(" 30.125 ") == 30_125)
        #expect(try VideoDurationDraft.milliseconds("18446744073709551.615") == UInt64.max)
        for invalid in ["", "0", "0.000", "-1", "+1", "1e3", "1.0001", "1,000", "18446744073709551.616"] {
            #expect(throws: (any Error).self) { try VideoDurationDraft.milliseconds(invalid) }
        }
    }
    @Test func scenesRemainOptionalAndSumExactlyWithoutOverflow() throws {
        let generation = GenerateRequest(prompt: "one prompt", model: "model", width: 768, height: 768, steps: 2, guidance: 1)
        var draft = VideoDurationDraft(seconds: "30")
        #expect(try draft.request(generation: generation).scenes == nil)
        draft.scenes = [.init(prompt: "first", seconds: "10"), .init(prompt: "second", seconds: "20")]
        let request = try draft.request(generation: generation)
        #expect(request.durationMs == 30_000)
        #expect(request.scenes?.map(\.durationMs) == [10_000, 20_000])
        draft.scenes[1].seconds = "19.999"
        #expect(throws: (any Error).self) { try draft.request(generation: generation) }
        draft.scenes[1].seconds = "20"; draft.scenes[1].prompt = " "
        #expect(throws: (any Error).self) { try draft.request(generation: generation) }
    }
    @Test func libraryExtensionBindsExactSourceAndUsesAddedDuration() throws {
        let generation = GenerateRequest(prompt: "camera pans right", model: "model", width: 768, height: 768, steps: 2, guidance: 1)
        let source = VideoJobSource(filename: "ordinary-print.mp4", expectedArchiveIdentitySha256: String(repeating: "a", count: 64))
        let request = try VideoDurationDraft(seconds: "30", source: source).request(generation: generation)
        #expect(request.source == source)
        #expect(request.durationMs == 30_000)
        #expect(request.generation.prompt == "camera pans right")
    }
    @Test func turningOffScenesRetainsTheirEditsWithoutAffectingSinglePromptAdmission() throws {
        let generation = GenerateRequest(prompt: "ordinary", model: "model", width: 768, height: 768, steps: 2, guidance: 1)
        var draft = VideoDurationDraft(seconds: "30", scenes: [.init(prompt: "unfinished scene", seconds: "invalid")])
        draft.scenesEnabled = false
        #expect(try draft.request(generation: generation).scenes == nil)
        #expect(draft.scenes[0].prompt == "unfinished scene")
        draft.scenesEnabled = true
        #expect(throws: (any Error).self) { try draft.request(generation: generation) }
    }
    @Test func displayedDecimalSecondsRoundTripExactly() throws {
        for value: UInt64 in [1, 10, 100, 1_000, 30_125, UInt64.max] {
            #expect(try VideoDurationDraft.milliseconds(VideoDurationDraft.seconds(milliseconds: value)) == value)
        }
        #expect(VideoDurationDraft.seconds(milliseconds: 30_100) == "30.1")
    }

    @Test func sceneTotalsFollowEditsButInvalidOrOverflowingInputCannotReplaceTheRequestedTime() throws {
        var draft = VideoDurationDraft(seconds: "30", scenes: [.init(prompt: "one", seconds: "10"), .init(prompt: "two", seconds: "20")])
        draft.scenes.append(.init(prompt: "three", seconds: "5"))
        let added = draft.updateDurationFromScenes()
        #expect(added)
        #expect(draft.seconds == "35")
        draft.scenes[0].seconds = "not yet a number"
        let refused = draft.updateDurationFromScenes()
        #expect(!refused)
        #expect(draft.seconds == "35")
        draft.scenes = [.init(prompt: "one", seconds: "18446744073709551.615"), .init(prompt: "two", seconds: "0.001")]
        let overflow = draft.updateDurationFromScenes()
        #expect(!overflow)
        #expect(draft.seconds == "35")
    }

}
