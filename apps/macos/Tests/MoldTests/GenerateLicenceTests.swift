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

    private let workstation = UUID()
    private let laptop = UUID()

    private func gate(
        _ json: String?, answeredFor: String? = "qwen-image-2.1-turbo:q8", answeredOn: UUID? = nil,
        submitting: String? = "qwen-image-2.1-turbo:q8", on host: UUID? = nil,
        accepted: Set<String> = []
    ) throws -> GeneratePane.LicenceGate {
        GeneratePane.licenceGate(
            placement: try json.map(preview), answeredFor: answeredFor,
            answeredOn: json == nil ? nil : (answeredOn ?? workstation),
            submitting: submitting, on: host ?? workstation, accepted: accepted)
    }

    @Test func aRenderThatWouldFetchAGatedModelAsksFirst() throws {
        guard case .ask(let licence) = try gate(gated) else {
            Issue.record("expected the licence to be asked for")
            return
        }
        #expect(licence.id == "qwen-research")
    }

    @Test func acceptedTermsAreNotAskedForTwice() throws {
        #expect(try gate(gated, accepted: ["qwen-research"]) == .clear)
    }

    @Test func aCurrentAnswerWithNothingToFetchIsClear() throws {
        #expect(try gate(#"{"outcome":"planned"}"#, answeredFor: "flux-dev:q4",
                         submitting: "flux-dev:q4") == .clear)
    }

    /// The probe is debounced: right after a model switch it still holds the
    /// previous model's answer, and before its first answer it holds none.
    /// Neither may read as "no licence needed" -- Generate must ask afresh.
    @Test func noCurrentAnswerIsUnknownNeverClear() throws {
        #expect(try gate(gated, submitting: "flux-dev:q4") == .unknown)
        #expect(try gate(nil) == .unknown)
    }

    /// Licence requirements belong to a MACHINE: an answer for the same model
    /// from another host says nothing about this one.
    @Test func anAnswerFromAnotherHostIsUnknown() throws {
        #expect(try gate(#"{"outcome":"planned"}"#, answeredOn: laptop, on: workstation) == .unknown)
    }
}
