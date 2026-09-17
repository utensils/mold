import Foundation
import Testing

@testable import MoldClient

// `Fixtures/config-plato.json` is a live `GET /api/config` from plato: 63
// entries, 16 of them `models.*` rows for two configured models
// (`flux-dev:q8`, `flux2-klein:q8`), every value `null`, `source:"db"` --
// which is what "configured but nothing set" looks like on a real host.

private func live() throws -> ConfigListing {
    try MoldJSON.decoder.decode(
        ConfigListing.self, from: RepoFixtures.fixture("config-plato.json"))
}

@Test func aModelNobodyConfiguredHasNoDefaults() throws {
    let defaults = ModelDefaults(from: try live(), model: "totally-unconfigured-model")
    #expect(defaults.isEmpty)
}

/// The state plato is actually in: a config row exists for the model but
/// every field is null. A present key with a null value must not read as
/// "0 steps" -- it means the same thing as no row at all.
@Test func aConfiguredModelWithEveryValueNullAlsoHasNoDefaults() throws {
    let defaults = ModelDefaults(from: try live(), model: "flux-dev:q8")
    #expect(defaults.isEmpty)
    #expect(defaults.steps == nil)
    #expect(defaults.guidance == nil)
    #expect(defaults.negativePrompt == nil)
}

/// Pinned against the literal `MODEL_FIELDS` order and spelling
/// (`crates/mold-core/src/config_keys.rs:359-368`).
@Test func theEightKeysAreSpelledTheWayTheRegistrySpellsThem() {
    let keys = ModelDefaults.keys(for: "flux-dev:q8")
    #expect(keys == [
        "models.flux-dev:q8.default_steps",
        "models.flux-dev:q8.default_guidance",
        "models.flux-dev:q8.default_width",
        "models.flux-dev:q8.default_height",
        "models.flux-dev:q8.scheduler",
        "models.flux-dev:q8.negative_prompt",
        "models.flux-dev:q8.lora",
        "models.flux-dev:q8.lora_scale",
    ])
}

/// A value the wire types as string | number | bool | null.
@Test func aConfigScalarRoundTripsEveryWireShape() throws {
    func roundTrip(_ json: String) throws -> ConfigScalar {
        try MoldJSON.decoder.decode(ConfigScalar.self, from: Data(json.utf8))
    }
    #expect(try roundTrip("42").int == 42)
    #expect(try roundTrip("3.5").double == 3.5)
    #expect(try roundTrip("\"euler-ancestral\"").text == "euler-ancestral")
    #expect(try roundTrip("true") == .bool(true))
    #expect(try roundTrip("null") == .null)
    #expect(try roundTrip("null").int == nil)
    #expect(try roundTrip("null").text == nil)
}

@Test func writesCoverOnlyTheFourNumbersAndTheNegativePrompt() {
    var draft = RenderDraft()
    draft.steps = 30
    draft.guidance = 4.5
    draft.width = 768
    draft.height = 512
    draft.negativePrompt = "blurry"
    let writes = ModelDefaults().writes(for: draft, model: "flux-dev:q8")
    let keys = writes.map(\.0)
    #expect(keys == [
        "models.flux-dev:q8.default_steps",
        "models.flux-dev:q8.default_guidance",
        "models.flux-dev:q8.default_width",
        "models.flux-dev:q8.default_height",
        "models.flux-dev:q8.negative_prompt",
    ])
}
