import Foundation
import Testing

@testable import MoldClient

private func loadModels() throws -> [Model] {
    try MoldJSON.decoder.decode([Model].self, from: RepoFixtures.fixture("models.json"))
}

@Test func decodesModelsCapturedFromALiveHost() throws {
    let models = try loadModels()
    #expect(models.count == 2)

    let dev = try #require(models.first { $0.name == "flux-dev:q4" })
    #expect(dev.family == "flux")
    #expect(dev.generationProfile != nil)
}

@Test func splitsTheManifestDescriptionIntoAHeadlineAndATradeOff() throws {
    let schnell = try #require(loadModels().first { $0.name == "flux-schnell:bf16" })

    // "FLUX.1 Schnell BF16 — fast 4-step, full precision (23.8GB transformer)"
    #expect(schnell.headline == "FLUX.1 Schnell BF16")
    #expect(schnell.tradeOff == "fast 4-step, full precision (23.8GB transformer)")
}

@Test func aDescriptionWithNoEmDashStillYieldsAHeadline() {
    let model = Model(
        name: "odd:tag", family: "flux", description: "No separator here",
        sizeGb: nil, isLoaded: nil, downloaded: nil, hfRepo: nil,
        displayName: nil, remainingDownloadBytes: nil, generationProfile: nil, supportsAudio: nil, diskUsageBytes: nil, kind: nil, modality: nil,
              nsfw: nil, runtimeAvailable: nil, runtimeUnavailableReason: nil
    )
    #expect(model.headline == "No separator here")
    #expect(model.tradeOff == nil)
}

@Test func partialInstallsReadAsRepairNotAsMissing() throws {
    let schnell = try #require(loadModels().first { $0.name == "flux-schnell:bf16" })
    // Not downloaded on workstation, with bytes outstanding.
    #expect(schnell.isReady == false)
    #expect(schnell.repairBytes != nil)
}

@Test func onlyPictureMakingFamiliesAreGenerators() {
    func model(family: String) -> Model {
        Model(name: "x", family: family, description: "d", sizeGb: nil, isLoaded: nil,
              downloaded: nil, hfRepo: nil, displayName: nil,
              remainingDownloadBytes: nil, generationProfile: nil, supportsAudio: nil, diskUsageBytes: nil, kind: nil, modality: nil,
              nsfw: nil, runtimeAvailable: nil, runtimeUnavailableReason: nil)
    }
    #expect(model(family: "flux").isGenerator)
    #expect(model(family: "wan").isGenerator)
    // A prompt-expansion LLM, an upscaler and a ControlNet are none of them
    // things a person picks to make a picture.
    #expect(!model(family: "qwen3-expand").isGenerator)
    #expect(!model(family: "upscaler").isGenerator)
    #expect(!model(family: "controlnet").isGenerator)
    #expect(!model(family: "hunyuan3d-paint").isGenerator)
}

@Test func splitsANameIntoItsBaseAndTag() {
    func model(_ name: String, _ description: String = "X — y") -> Model {
        Model(name: name, family: "flux", description: description, sizeGb: nil,
              isLoaded: nil, downloaded: nil, hfRepo: nil, displayName: nil,
              remainingDownloadBytes: nil, generationProfile: nil, supportsAudio: nil, diskUsageBytes: nil, kind: nil, modality: nil,
              nsfw: nil, runtimeAvailable: nil, runtimeUnavailableReason: nil)
    }
    #expect(model("flux-dev:q4").baseName == "flux-dev")
    #expect(model("flux-dev:q4").tag == "q4")
    #expect(model("wuerstchen").tag == nil)
    #expect(model("wuerstchen").baseName == "wuerstchen")
}

@Test func aGroupHeadingDropsTheQuantizationButKeepsRealWords() throws {
    let models = try loadModels()
    let dev = try #require(models.first { $0.name == "flux-dev:q4" })
    let schnell = try #require(models.first { $0.name == "flux-schnell:bf16" })

    // "FLUX.1 Dev Q4" -> "FLUX.1 Dev";  "FLUX.1 Schnell BF16" -> "FLUX.1 Schnell"
    #expect(dev.baseTitle == "FLUX.1 Dev")
    #expect(schnell.baseTitle == "FLUX.1 Schnell")
    // "Dev" is a real word, not a quantization, so it must survive.
    #expect(dev.baseTitle.hasSuffix("Dev"))
}

@Test func aCatalogIdIsANamespaceNotAVariantTag() {
    func model(_ name: String) -> Model {
        Model(name: name, family: "sdxl", description: "D — t", sizeGb: nil, isLoaded: nil,
              downloaded: nil, hfRepo: nil, displayName: nil,
              remainingDownloadBytes: nil, generationProfile: nil, supportsAudio: nil, diskUsageBytes: nil, kind: nil, modality: nil,
              nsfw: nil, runtimeAvailable: nil, runtimeUnavailableReason: nil)
    }
    // Two unrelated Civitai checkpoints must not share a base name, or they
    // group together as variants of each other.
    #expect(model("cv:252914").baseName == "cv:252914")
    #expect(model("cv:1759168").baseName == "cv:1759168")
    #expect(model("cv:252914").baseName != model("cv:1759168").baseName)
    #expect(model("cv:252914").tag == nil)
    #expect(model("hf:owner/repo").baseName == "hf:owner/repo")
    // A real variant tag still splits.
    #expect(model("flux-dev:q4").baseName == "flux-dev")
}

@Test func moreThanOneTrailingVariantWordComesOff() {
    func titled(_ description: String) -> String {
        Model(name: "n:q4", family: "flux2", description: description, sizeGb: nil,
              isLoaded: nil, downloaded: nil, hfRepo: nil, displayName: nil,
              remainingDownloadBytes: nil, generationProfile: nil, supportsAudio: nil, diskUsageBytes: nil, kind: nil, modality: nil,
              nsfw: nil, runtimeAvailable: nil, runtimeUnavailableReason: nil).baseTitle
    }
    #expect(titled("FLUX.2 [dev] Q4 GGUF — smallest dev tier") == "FLUX.2 [dev]")
    #expect(titled("Flux.2 Klein-4B Base Q4 GGUF — undistilled") == "Flux.2 Klein-4B Base")
    #expect(titled("FLUX.1 Dev Q4 — good quality") == "FLUX.1 Dev")
    // A model whose whole name looks like a variant keeps at least one word.
    #expect(titled("Q4 — x") == "Q4")
}
