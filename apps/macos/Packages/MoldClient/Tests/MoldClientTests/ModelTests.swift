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
        displayName: nil, remainingDownloadBytes: nil, generationProfile: nil
    )
    #expect(model.headline == "No separator here")
    #expect(model.tradeOff == nil)
}

@Test func partialInstallsReadAsRepairNotAsMissing() throws {
    let schnell = try #require(loadModels().first { $0.name == "flux-schnell:bf16" })
    // Not downloaded on plato, with bytes outstanding.
    #expect(schnell.isReady == false)
    #expect(schnell.repairBytes != nil)
}

@Test func onlyPictureMakingFamiliesAreGenerators() {
    func model(family: String) -> Model {
        Model(name: "x", family: family, description: "d", sizeGb: nil, isLoaded: nil,
              downloaded: nil, hfRepo: nil, displayName: nil,
              remainingDownloadBytes: nil, generationProfile: nil)
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
