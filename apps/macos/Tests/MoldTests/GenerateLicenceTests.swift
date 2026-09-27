import Foundation
import MoldClient
import Testing

@testable import Mold

/// Generate asks for a gated model's terms BEFORE it queues a render that
/// would fetch it -- Qwen Image 2.1 and its turbo tiers, Qwen Research.
@MainActor
struct GenerateLicenceTests {
    private let gated = """
    {"outcome":"infeasible","pending_downloads":[
      {"kind":"model","name":"qwen-image-2.1-turbo:q8","repo":"Qwen/Qwen-Image-2.1","bytes":1,
       "install_model":"qwen-image-2.1-turbo:q8",
       "licenses":[{"id":"qwen-research","name":"Qwen Research License","url":"https://x/raw",
                    "canonical":"https://x","sha256":"abc","summary":"Research use."}]}]}
    """

    private func preview(_ json: String) throws -> PlacementPreview {
        try MoldJSON.decoder.decode(PlacementPreview.self, from: Data(json.utf8))
    }

    @Test func aRenderThatWouldFetchAGatedModelAsksFirst() throws {
        let licence = GeneratePane.licenceToAsk(
            placement: try preview(gated), answeredFor: "qwen-image-2.1-turbo:q8",
            submitting: "qwen-image-2.1-turbo:q8", accepted: [])
        #expect(licence?.id == "qwen-research")
    }

    @Test func acceptedTermsAreNotAskedForTwice() throws {
        #expect(GeneratePane.licenceToAsk(
            placement: try preview(gated), answeredFor: "qwen-image-2.1-turbo:q8",
            submitting: "qwen-image-2.1-turbo:q8", accepted: ["qwen-research"]) == nil)
    }

    @Test func anAnswerAboutAnotherModelAsksNothing() throws {
        // The probe is debounced: right after a model switch it still holds
        // the previous model's answer.
        #expect(GeneratePane.licenceToAsk(
            placement: try preview(gated), answeredFor: "qwen-image-2.1-turbo:q8",
            submitting: "flux-dev:q4", accepted: []) == nil)
        #expect(GeneratePane.licenceToAsk(
            placement: try preview(#"{"outcome":"planned"}"#), answeredFor: "flux-dev:q4",
            submitting: "flux-dev:q4", accepted: []) == nil)
    }
}
