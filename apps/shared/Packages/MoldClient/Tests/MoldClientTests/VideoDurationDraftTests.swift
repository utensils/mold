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
}
