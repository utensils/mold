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

@Test func aCuratedIdTheServerDoesNotNameIsNamedByTheGeneratedTable() throws {
    // The fixture predates the presentation fields, so the generated table
    // (the same Rust authority) names it -- no splitting of the description.
    let schnell = try #require(loadModels().first { $0.name == "flux-schnell:bf16" })

    #expect(schnell.presentation == ModelPresentationTable.models["flux-schnell:bf16"])
    #expect(schnell.headline == "FLUX.1 Schnell BF16")
    #expect(schnell.title == "FLUX.1 Schnell")
    #expect(schnell.variantLabel == "BF16")
    #expect(schnell.summary == "Fast 4-step, full precision")
    #expect(schnell.familyLabel == "FLUX.1")
}

@Test func theServersPresentationFieldsWinOverTheGeneratedTable() throws {
    let model = try MoldJSON.decoder.decode(Model.self, from: Data(#"""
        {"name":"flux-dev:q4","family":"flux","description":"Legacy — old",
         "display_name":"FLUX.1 Dev Q4","title":"Server Dev","variant":"Q4 small",
         "summary":"Said by the server","family_label":"Server FLUX"}
        """#.utf8))

    #expect(model.presentation == ModelPresentation(
        title: "Server Dev", variant: "Q4 small", summary: "Said by the server", familyLabel: "Server FLUX"))
    #expect(model.headline == "Server Dev Q4 small")
    #expect(model.title == "Server Dev")
    #expect(model.variantLabel == "Q4 small")
    #expect(model.summary == "Said by the server")
    #expect(model.familyLabel == "Server FLUX")
}

@Test func presentationFieldsRoundTripUnderTheirWireNames() throws {
    let model = try MoldJSON.decoder.decode(Model.self, from: Data(#"""
        {"name":"flux-dev:q4","family":"flux","description":"d","title":"T","family_label":"F"}
        """#.utf8))
    let json = try #require(String(data: MoldJSON.encoder.encode(model), encoding: .utf8))
    #expect(json.contains(#""title":"T""#))
    #expect(json.contains(#""family_label":"F""#))
    #expect(try MoldJSON.decoder.decode(Model.self, from: MoldJSON.encoder.encode(model)) == model)
}

@Test func catalogInstallsKeepTheirProviderTitles() throws {
    let civitai = try MoldJSON.decoder.decode(Model.self, from: Data(#"""
        {"name":"cv:252914","family":"sdxl","description":"Juggernaut XL","display_name":"Juggernaut XL"}
        """#.utf8))
    #expect(civitai.presentation == nil)
    #expect(civitai.headline == "Juggernaut XL")
    #expect(civitai.title == "Juggernaut XL")
    #expect(civitai.variantLabel == nil)
    // The description IS the provider title; it is not repeated as a reason.
    #expect(civitai.summary == nil)
    #expect(civitai.familyLabel == "SDXL")

    let bare = try MoldJSON.decoder.decode(Model.self, from: Data(#"""
        {"name":"hf:owner/my-cool_model","family":"flux","description":""}
        """#.utf8))
    #expect(bare.headline == "my cool model")
    #expect(try MoldJSON.decoder.decode(Model.self, from: Data(#"{"name":"cv:42","family":"sdxl","description":""}"#.utf8))
        .headline == "Civitai model #42")
}

@Test func anUnknownModelFallsBackToItsOwnWords() {
    let model = Model(
        name: "odd:tag", family: "my-family", description: "No separator here",
        sizeGb: nil, isLoaded: nil, downloaded: nil, hfRepo: nil,
        displayName: nil, remainingDownloadBytes: nil, generationProfile: nil, supportsAudio: nil, diskUsageBytes: nil, kind: nil, modality: nil,
              nsfw: nil, runtimeAvailable: nil, runtimeUnavailableReason: nil
    )
    #expect(model.presentation == nil)
    #expect(model.headline == "No separator here")
    #expect(model.summary == nil)
    // A family the authority has never heard of shows its own slug.
    #expect(model.familyLabel == "my-family")
}

@Test func aBareFamilySlugReadsThroughTheGeneratedLabels() {
    #expect(Model.familyLabel(forFamily: "ltx2") == "LTX-2")
    #expect(Model.familyLabel(forFamily: "minimax-h3") == "MiniMax H3")
    #expect(Model.familyLabel(forFamily: "brand-new") == "brand-new")
}

@Test func aBareIdIsNamedByTheGeneratedTable() {
    #expect(Model.headline(forName: "flux-dev:q4") == "FLUX.1 Dev Q4")
    #expect(Model.headline(forName: "cv:42") == "cv:42")
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

@Test func aQueueRowNamesACuratedIdTheMachineDidNotName() throws {
    let row = try MoldJSON.decoder.decode(QueueEntry.self, from: Data(#"{"id":"j","state":"queued","model":"flux-dev:q8"}"#.utf8))
    #expect(row.modelHeadline == "FLUX.1 Dev Q8")
    let catalog = try MoldJSON.decoder.decode(QueueEntry.self, from: Data(#"{"id":"k","state":"queued","model":"cv:42"}"#.utf8))
    #expect(catalog.modelHeadline == "cv:42")
}

@Test func curatedDiscoverySearchesRepositoryAndAllWords() throws {
    let model = try MoldJSON.decoder.decode(Model.self, from: Data(#"{"name":"flux-dev:q4","family":"flux","description":"FLUX.1 Dev Q4 — compact","hf_repo":"black-forest-labs/FLUX.1-dev","downloaded":false}"#.utf8))
    let query = CatalogQuery(text: "forest dev", source: "hf")
    #expect(query.queryString.contains("source=hf"))
    #expect(!query.queryString.contains("huggingface"))
    #expect(model.matchesDiscovery(query))
    #expect(!model.matchesDiscovery(CatalogQuery(text: "wan")))
    #expect(!model.matchesDiscovery(CatalogQuery(source: "civitai")))
    #expect(!model.matchesDiscovery(CatalogQuery(family: "wan")))
}
