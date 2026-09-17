import Foundation
import Testing

@testable import MoldClient

// Fixtures captured read-only from plato (100.105.134.43:7680, mold 0.29.0)
// on 2026-09-16, `GET /api/models` -> each model's `generation_profile`.
// `recipe-ltx2.json` is trimmed to the `auto`, `t2a` and `ic-lora` recipes,
// the three that differ; `recipe-wan.json` is `wan22-t2v-a14b:q8`, a
// text-to-video-only tier, chosen specifically because its `source_image` is
// `unsupported` -- the negative case `flux-schnell`'s absent field is not.

private func profile(_ name: String) throws -> GenerationProfileSet {
    try MoldJSON.decoder.decode(GenerationProfileSet.self, from: RepoFixtures.fixture(name))
}

private func recipe(_ name: String, _ id: String = "default") throws -> GenerationRecipe {
    let set = try profile(name)
    return try #require(set.recipe(named: id))
}

// MARK: - source_image absence, fact 1

@Test func anAbsentSourceImageBlockMeansTheRecipeReadsOne() throws {
    let flux = try recipe("recipe-flux-schnell.json")
    #expect(flux.capabilities.sourceImage == nil)
    #expect(flux.capabilities.readsSourceImage == true)
    #expect(flux.capabilities.requiresSourceImage == false)
}

@Test func anUnsupportedSourceImageIsANo() throws {
    let wanT2v = try recipe("recipe-wan.json")
    #expect(wanT2v.capabilities.sourceImage == .unsupported)
    #expect(wanT2v.capabilities.readsSourceImage == false)
}

// MARK: - ControlNet, fact 3

@Test func sd15AdvertisesAnAdjustableControlnetWithOneSlot() throws {
    let sd15 = try recipe("recipe-sd15.json")
    let controlnet = try #require(sd15.capabilities.controlNet)
    #expect(controlnet.mode == .adjustable)
    #expect(controlnet.maxCount == 1)
}

@Test func aHiddenControlnetCarriesTheServersOwnSentence() throws {
    let zimage = try recipe("recipe-zimage.json")
    #expect(zimage.capabilities.controlNet == nil)
    #expect(
        zimage.capabilities.controlNetReason
            == "ControlNet generation is available for SD1.5 models.")
}

// MARK: - Mask / keyframes, fact 4 and the still-image survey

@Test func everyStillRecipeTakesAMaskAndNoKeyframes() throws {
    for fixture in ["recipe-flux-schnell.json", "recipe-sd15.json", "recipe-zimage.json"] {
        let still = try recipe(fixture)
        #expect(still.capabilities.acceptsMask == true, "\(fixture) should accept a mask")
        #expect(
            still.capabilities.acceptsKeyframes == false,
            "\(fixture) should not accept keyframes")
    }
}

@Test func ltx2TakesKeyframesAudioAndASourceVideo() throws {
    let auto = try recipe("recipe-ltx2.json", "auto")
    #expect(auto.capabilities.acceptsKeyframes == true)
    #expect(auto.capabilities.acceptsSourceAudio == true)
    #expect(auto.capabilities.acceptsSourceVideo == true)
}

@Test func theT2aRecipeHidesAudioAndDeliversWav() throws {
    let t2a = try recipe("recipe-ltx2.json", "t2a")
    #expect(t2a.capabilities.acceptsSourceAudio == false)
    #expect(t2a.capabilities.acceptsSourceVideo == false)
    let output = try #require(t2a.capabilities.output)
    #expect(output.defaultFormat == "wav")
    #expect(output.formats == ["wav"])
    #expect(output.isFixed == true)
}

// MARK: - Schedulers, arriving absent not empty

@Test func schedulersArriveAbsentNotEmpty() throws {
    let zimage = try recipe("recipe-zimage.json")
    #expect(zimage.capabilities.schedulers == nil)

    let sd15 = try recipe("recipe-sd15.json")
    #expect(sd15.capabilities.schedulers == ["ddim", "euler-ancestral", "uni-pc"])
}

// MARK: - request_selector, "the recipe blocks, as they really arrive"

@Test func theAutoRecipeSelectsNoPipeline() throws {
    let set = try profile("recipe-ltx2.json")
    let auto = try #require(set.recipe(named: "auto"))
    #expect(auto.requestSelector?.pipeline == nil)

    for id in ["t2a", "ic-lora"] {
        let recipe = try #require(set.recipe(named: id))
        #expect(recipe.requestSelector?.pipeline == id)
    }
}

// MARK: - Identity, fact 6, synthetic

@Test func aHostWithoutMultiPhotoOffersOne() throws {
    let json = Data("""
        {"identity":{"multi_photo":false,"max_photos":4}}
        """.utf8)
    let caps = try MoldJSON.decoder.decode(Capabilities.self, from: json)
    #expect(caps.maxIdentityPhotos == 1)
}

@Test func anAbsentIdentityBlockOffersNone() throws {
    let caps = try MoldJSON.decoder.decode(Capabilities.self, from: Data("{}".utf8))
    #expect(caps.maxIdentityPhotos == 0)
}

// MARK: - LoRA fallback, fact "an older host still stacks"

@Test func anOlderRecipeWithSupportsLoraAndNoBlockStillStacks() throws {
    let json = Data("""
        {"supports_lora":true}
        """.utf8)
    let caps = try MoldJSON.decoder.decode(RecipeCapabilities.self, from: json)
    let stack = try #require(caps.loraStack)
    #expect(stack.mode == .adjustable)
    #expect(stack.maxCount == Lora.defaultMaxStack)
}
