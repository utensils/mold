import Foundation
import Testing

@testable import MoldClient

// The corollary of `PromptTransform.swift`'s own rule -- an `OpenWireEnum`
// may never be ENCODED as `.unknown` -- applied to the PROVENANCE, which is
// the one place a value decoded from a newer host gets written back out.

/// **Fails today**: `wireSafe` does not exist, so a provenance built from a
/// task this build cannot spell encodes `"task":"unknown"`. The Rust
/// `PromptTransformProvenance.task` is a required strict enum with no such
/// variant (`types.rs:711-727`), so serde refuses the whole
/// `POST /api/generation-batches` body -- and the poisoned block lives on in
/// the draft, so every later press of Generate fails naming
/// `prompt_transform` rather than the wand that wrote it.
@Test func aProvenanceNamingATaskThisBuildCannotSpellIsNotSent() throws {
    let json = Data("""
    {"source_prompt":"a cat","source_kind":"direct","task":"text-to-hologram",\
    "variants":[{"prompt":"a cat, lit","dimensions":["lighting"]}]}
    """.utf8)
    let response = try MoldJSON.decoder.decode(RemixResponse.self, from: json)
    #expect(response.task == .unknown)

    let provenance = PromptTransformProvenance(
        operation: .remix, sourcePrompt: response.sourcePrompt, sourceKind: .direct,
        task: response.task, dimensions: response.variants[0].dimensions)
    #expect(provenance.wireSafe == nil)
}

/// `operation` is required and strict too (`types.rs:713`), with no default
/// to fall back to -- so it drops the block for the same reason `task` does.
@Test func aProvenanceNamingAnOperationThisBuildCannotSpellIsNotSent() {
    let provenance = PromptTransformProvenance(
        operation: .unknown, sourcePrompt: "a cat", task: .textToImage)
    #expect(provenance.wireSafe == nil)
}

/// `source_kind` is `#[serde(default)]` Direct (`types.rs:698, 724`), so an
/// unspellable one has somewhere honest to land and the record survives.
@Test func aProvenanceWithAnUnspellableSourceKindFallsBackToTheServerDefault() throws {
    let provenance = PromptTransformProvenance(
        operation: .expand, sourcePrompt: "a cat", sourceKind: .unknown, task: .textToImage)
    let safe = try #require(provenance.wireSafe)
    #expect(safe.sourceKind == .direct)
}

/// A dimension this build cannot spell is dropped on its own: the rest of the
/// provenance is a true record of how the prompt was written, and losing it
/// over one word would be the same overreaction in reverse.
@Test func aProvenanceDropsOnlyTheDimensionItCannotSpell() throws {
    let provenance = PromptTransformProvenance(
        operation: .remix, rootPrompt: "a cat", sourcePrompt: "a cat",
        sourceKind: .current, task: .textToImage,
        dimensions: [.lighting, .unknown, .mood])
    let safe = try #require(provenance.wireSafe)
    #expect(safe.dimensions == [.lighting, .mood])
    #expect(safe.task == .textToImage)
    #expect(safe.rootPrompt == "a cat")
    #expect(safe.operation == .remix)
    #expect(safe.sourceKind == .current)
}

/// The ordinary case changes nothing -- this is a filter, not a rewrite.
@Test func aProvenanceWithNothingUnknownIsItself() {
    let provenance = PromptTransformProvenance(
        operation: .expand, rootPrompt: "a cat", sourcePrompt: "a cat asleep",
        sourceKind: .current, task: .textToVideo, dimensions: [.camera])
    #expect(provenance.wireSafe == provenance)
}

/// The whole point: what a `GenerateRequest` carrying a poisoned provenance
/// puts on the wire is no `prompt_transform` at all, never `"unknown"`.
@Test func aRequestCarryingAPoisonedProvenanceSendsNoPromptTransform() throws {
    var request = GenerateRequest(prompt: "a cat", model: "flux-dev:q8", width: 1024,
                                  height: 1024, steps: 20, guidance: 3.5)
    request.promptTransform = PromptTransformProvenance(
        operation: .remix, sourcePrompt: "a cat", task: .unknown)
    let object = try #require(
        try JSONSerialization.jsonObject(with: MoldJSON.encoder.encode(request))
            as? [String: Any])
    #expect(object["prompt_transform"] == nil)
}

/// ...and a good one still rides, with its unspellable dimension gone.
@Test func aRequestCarryingAGoodProvenanceStillSendsIt() throws {
    var request = GenerateRequest(prompt: "a cat", model: "flux-dev:q8", width: 1024,
                                  height: 1024, steps: 20, guidance: 3.5)
    request.promptTransform = PromptTransformProvenance(
        operation: .remix, sourcePrompt: "a cat", task: .textToImage,
        dimensions: [.mood, .unknown])
    let object = try #require(
        try JSONSerialization.jsonObject(with: MoldJSON.encoder.encode(request))
            as? [String: Any])
    let transform = try #require(object["prompt_transform"] as? [String: Any])
    #expect(transform["task"] as? String == "text-to-image")
    #expect(transform["dimensions"] as? [String] == ["mood"])
}
