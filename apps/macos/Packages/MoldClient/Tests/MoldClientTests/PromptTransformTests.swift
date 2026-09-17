import Foundation
import Testing

@testable import MoldClient

// `Fixtures/expand-ignored-hunyuan3d.json` is HAND-BUILT from the corpus, not
// captured: `ignored_prompt_advice` (`crates/mold-core/src/expand_prompts.rs`)
// returns the hunyuan3d family guide's own headline plus its `Generation
// context` H2 section verbatim (`crates/mold-core/src/prompting/families/
// hunyuan3d.md:20-31`) -- that section contains no `{{word_limit}}`
// placeholder, so nothing in the corpus's `excerpt()` pipeline shortens or
// rewrites it, and quoting the file exactly reproduces what a live host
// answers.

@Test func aFamilyThatReadsNoPromptIsAnsweredNotExpanded() throws {
    let response = try MoldJSON.decoder.decode(
        ExpandResponse.self, from: RepoFixtures.fixture("expand-ignored-hunyuan3d.json"))
    #expect(response.original == "a brass gear")
    #expect(response.expanded.count == 1)
    #expect(response.expanded[0].hasPrefix("hunyuan3d reads no prompt"))
}

/// The M2 lesson (an open enum degrading silently on a spelling mismatch),
/// applied before it can be made again: every case is pinned against the
/// exact kebab-case spelling `#[serde(rename_all = "kebab-case")]` produces,
/// and a value this build never heard of degrades to `.unknown` rather than
/// failing the decode.
@Test func everyKebabCaseTaskMatchesItsWireSpelling() throws {
    let spellings: [ExpandTask: String] = [
        .textToImage: "text-to-image",
        .textToVideo: "text-to-video",
        .imageToVideo: "image-to-video",
        .videoToVideo: "video-to-video",
        .retake: "retake",
        .keyframeInterpolation: "keyframe-interpolation",
        .audioDrivenVideo: "audio-driven-video",
        .referenceToAudioVideo: "reference-to-audio-video",
        .textToAudio: "text-to-audio",
    ]
    for (task, wire) in spellings {
        #expect(task.rawValue == wire)
        let decoded = try MoldJSON.decoder.decode(
            ExpandTask.self, from: Data("\"\(wire)\"".utf8))
        #expect(decoded == task)
    }
    let unheardOf = try MoldJSON.decoder.decode(
        ExpandTask.self, from: Data("\"some-future-task\"".utf8))
    #expect(unheardOf == .unknown)
}

@Test func aRemixVariantKeepsTheDimensionsItVaried() throws {
    let json = Data("""
    {"source_prompt":"a cat asleep on a warm windowsill","root_prompt":"a cat",\
    "source_kind":"original","task":"text-to-image","variants":[\
    {"prompt":"a cat asleep on a windowsill at golden hour","dimensions":["lighting","mood"]},\
    {"prompt":"a cat curled on a rain-streaked windowsill","dimensions":["setting"]}]}
    """.utf8)
    let response = try MoldJSON.decoder.decode(RemixResponse.self, from: json)
    #expect(response.sourcePrompt == "a cat asleep on a warm windowsill")
    #expect(response.rootPrompt == "a cat")
    #expect(response.sourceKind == .original)
    #expect(response.task == .textToImage)
    #expect(response.variants.count == 2)
    #expect(response.variants[0].dimensions == [.lighting, .mood])
    #expect(response.variants[1].dimensions == [.setting])

    let encoded = try MoldJSON.encoder.encode(response.variants[0])
    let roundTripped = try MoldJSON.decoder.decode(RemixVariant.self, from: encoded)
    #expect(roundTripped == response.variants[0])
}

/// `task` omitted means "let the server infer it from the family"; sending
/// the literal string `"unknown"` is a 422 on a route whose whole job is to
/// be optional, so an `OpenWireEnum` must never be ENCODED as `.unknown`.
@Test func anExpandRequestWithNoTaskOmitsTheKeyRatherThanSendingUnknown() throws {
    let request = ExpandRequest(prompt: "a cat", modelFamily: "flux", variations: 3, task: nil)
    let encoded = try MoldJSON.encoder.encode(request)
    let object = try #require(try JSONSerialization.jsonObject(with: encoded) as? [String: Any])
    #expect(Set(object.keys) == ["model_family", "variations", "prompt"])
    #expect(object["prompt"] as? String == "a cat")
    #expect(object["model_family"] as? String == "flux")
    #expect(object["variations"] as? Int == 3)
}
