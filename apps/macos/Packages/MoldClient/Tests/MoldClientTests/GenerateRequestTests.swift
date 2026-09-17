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

// MARK: - S2: mask, adapters, identity

@Test func aMaskWithoutASourceNeverReachesTheWire() throws {
    var draft = RenderDraft()
    draft.media.maskImage = "MASK"
    let noSource = try encoded(draft.request(model: "m"))
    #expect(noSource["mask_image"] == nil)

    draft.media.sourceImage = "SRC"
    let withSource = try encoded(draft.request(model: "m"))
    #expect(withSource["mask_image"] as? String == "MASK")
}

@Test func onePhotographEncodesAsIdImageAndFourAsIdImages() throws {
    var draft = RenderDraft()
    draft.media.identity = IdentityConditioning(photos: [IdentityPhoto(encoded: "AAAA", name: "face.png")])
    let single = try encoded(draft.request(model: "m", maxIdentityPhotos: 4))
    #expect(single["id_image"] as? String == "AAAA")
    #expect(single["id_image_name"] as? String == "face.png")
    #expect(single["id_images"] == nil)

    draft.media.identity = IdentityConditioning(photos: (0 ..< 4).map {
        IdentityPhoto(encoded: "P\($0)", name: "p\($0).png")
    })
    let several = try encoded(draft.request(model: "m", maxIdentityPhotos: 4))
    #expect((several["id_images"] as? [String])?.count == 4)
    #expect((several["id_image_names"] as? [String])?.count == 4)
    #expect(several["id_image"] == nil)
}

@Test func neitherFormEverAppearsBesideTheOther() throws {
    var draft = RenderDraft()
    draft.media.identity = IdentityConditioning(photos: [IdentityPhoto(encoded: "A", name: "a.png")])
    var json = try encoded(draft.request(model: "m", maxIdentityPhotos: 4))
    #expect(!(json.keys.contains("id_image") && json.keys.contains("id_images")))

    draft.media.identity = IdentityConditioning(photos: [
        IdentityPhoto(encoded: "A", name: "a.png"), IdentityPhoto(encoded: "B", name: "b.png"),
    ])
    json = try encoded(draft.request(model: "m", maxIdentityPhotos: 4))
    #expect(!(json.keys.contains("id_image") && json.keys.contains("id_images")))
    #expect(json.keys.contains("id_images"))
}

@Test func aHostThatTakesOnePhotoSendsTheSingularFormFromAListOfThree() throws {
    var draft = RenderDraft()
    draft.media.identity = IdentityConditioning(photos: [
        IdentityPhoto(encoded: "A", name: "a.png"),
        IdentityPhoto(encoded: "B", name: "b.png"),
        IdentityPhoto(encoded: "C", name: "c.png"),
    ])
    // The host understands only `id_image` -- the rest is dropped rather
    // than silently truncated into the plural form.
    let json = try encoded(draft.request(model: "m", maxIdentityPhotos: 1))
    #expect(json["id_image"] as? String == "A")
    #expect(json["id_images"] == nil)
}

@Test func idStartStepIsClampedBelowTheStepCount() throws {
    var draft = RenderDraft()
    draft.steps = 4
    draft.media.identity = IdentityConditioning(
        photos: [IdentityPhoto(encoded: "A", name: "a.png")], weight: 1, startStep: 20
    )
    let json = try encoded(draft.request(model: "m", maxIdentityPhotos: 4))
    #expect(json["id_start_step"] as? Int == 3)
}

@Test func anAdapterStackNeverWritesTheLegacyLoraField() throws {
    var draft = RenderDraft()
    draft.media.loras = [LoraChoice(path: "/x.safetensors", scale: 0.8, name: "X")]
    let json = try encoded(draft.request(model: "m"))
    #expect(json["lora"] == nil)
    let loras = try #require(json["loras"] as? [[String: Any]])
    #expect(loras.count == 1)
    #expect(loras[0]["path"] as? String == "/x.safetensors")
    #expect(loras[0]["scale"] as? Double == 0.8)
    #expect(loras[0]["name"] == nil)
}

@Test func identityWeightAndStartStepRideOnlyWithAPhotograph() throws {
    let draft = RenderDraft()
    let json = try encoded(draft.request(model: "m", maxIdentityPhotos: 4))
    #expect(json["id_weight"] == nil)
    #expect(json["id_start_step"] == nil)
}
