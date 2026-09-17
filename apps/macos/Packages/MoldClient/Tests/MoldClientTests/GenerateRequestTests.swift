import Foundation
import Testing

@testable import MoldClient

private func encoded(_ request: GenerateRequest) throws -> [String: Any] {
    let data = try MoldJSON.encoder.encode(request)
    return try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
}

@Test func encodesTheFieldsTheServerRequiresInSnakeCase() throws {
    let json = try encoded(GenerateRequest(
        prompt: "a tin robot", model: "flux-dev:q4",
        width: 1024, height: 1024, steps: 20, guidance: 3.5
    ))

    #expect(json["prompt"] as? String == "a tin robot")
    #expect(json["model"] as? String == "flux-dev:q4")
    #expect(json["batch_size"] as? Int == 1)
    #expect(json["steps"] as? Int == 20)
}

@Test func omitsUnsetOptionalsRatherThanSendingNull() throws {
    let json = try encoded(GenerateRequest(
        prompt: "p", model: "m", width: 512, height: 512, steps: 4, guidance: 0
    ))
    // An absent seed means "you pick one". A null could be read as an
    // explicit clear, which is a different instruction.
    #expect(json["seed"] == nil)
    #expect(json["negative_prompt"] == nil)
    #expect(json["save_to_gallery"] == nil)
}

@Test func sendsOptionalsThatWereSet() throws {
    let json = try encoded(GenerateRequest(
        prompt: "p", model: "m", width: 512, height: 512, steps: 4, guidance: 0,
        negativePrompt: "blurry", seed: 42, saveToGallery: false
    ))
    #expect(json["seed"] as? UInt64 == 42)
    #expect(json["negative_prompt"] as? String == "blurry")
    #expect(json["save_to_gallery"] as? Bool == false)
}

@Test func aBatchCarriesAStableClientIdForRecovery() {
    let admission = BatchAdmission(requests: [])
    #expect(!admission.clientBatchId.isEmpty)
    #expect(UUID(uuidString: admission.clientBatchId) != nil)
}

@Test func filingAndBatchFieldsEncodeInSnakeCaseWhenSet() throws {
    var request = GenerateRequest(
        prompt: "p", model: "m", width: 512, height: 512, steps: 4, guidance: 0
    )
    request.outputFormat = "webp"
    request.upscaleModel = "real-esrgan-x4plus:fp16"
    request.title = "Smurf village at dusk"
    request.tags = ["blue", "village"]
    request.collection = .named("Smurf Village")
    request.originalPrompt = "a village"
    request.batchId = "b1"
    request.batchIndex = 1
    request.batchCount = 4

    let json = try encoded(request)
    #expect(json["output_format"] as? String == "webp")
    #expect(json["upscale_model"] as? String == "real-esrgan-x4plus:fp16")
    #expect(json["title"] as? String == "Smurf village at dusk")
    #expect(json["tags"] as? [String] == ["blue", "village"])
    #expect((json["collection"] as? [String: Any])?["name"] as? String == "Smurf Village")
    #expect(json["original_prompt"] as? String == "a village")
    #expect(json["batch_id"] as? String == "b1")
    #expect(json["batch_index"] as? Int == 1)
    #expect(json["batch_count"] as? Int == 4)
}

@Test func filingAndBatchFieldsAreOmittedWhenUnset() throws {
    let json = try encoded(GenerateRequest(
        prompt: "p", model: "m", width: 512, height: 512, steps: 4, guidance: 0
    ))
    for key in [
        "output_format", "upscale_model", "title", "tags", "collection",
        "original_prompt", "prompt_transform", "batch_id", "batch_index", "batch_count",
    ] {
        #expect(json[key] == nil)
    }
}
