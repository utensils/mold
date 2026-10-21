import Testing

@testable import MoldClient

struct VideoDurationUpscaleProjectionTests {
    @Test func historicalAuthoredProtocolRefusesUpscaleRecipeRatherThanDroppingIt() throws {
        for spatial in [true, false] {
            var request = GenerateRequest(
                prompt: "A garden", model: "video", width: 768, height: 768, steps: 8, guidance: 1, seed: 42)
            if spatial { request.spatialUpscale = "2x" } else { request.temporalUpscale = "2x" }
            #expect(throws: (any Error).self) { try AuthoredChainRequest.requireRepresentable(request) }
        }
    }
}
