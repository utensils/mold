import Foundation
import Testing

@testable import MoldClient

// The layout projection and the request pruning that follow from
// `capabilities.reference_images.source_relation` (findings 02#1, 01#4).

private func recipe(_ name: String, _ id: String = "default") throws -> GenerationRecipe {
    let set = try MoldJSON.decoder.decode(
        GenerationProfileSet.self, from: RepoFixtures.fixture(name))
    return try #require(set.recipe(named: id))
}

private func references(
    relation: String, primaryIsTarget: Bool = false, mode: String = "adjustable"
) throws -> ReferenceImagesCapability {
    try MoldJSON.decoder.decode(ReferenceImagesCapability.self, from: Data("""
    {"mode":"\(mode)","required":false,"max_count":4,
     "primary_is_target":\(primaryIsTarget),"source_relation":"\(relation)"}
    """.utf8))
}

// MARK: - The projection

@Test func everyAdvertisedRelationProjectsOntoItsOwnLayout() throws {
    #expect(SourceImageMode(references: nil) == .single)
    #expect(try SourceImageMode(references: references(relation: "replaces")) == .references)
    #expect(try SourceImageMode(references: references(relation: "exclusive"))
        == .singleOrReferences)
    #expect(try SourceImageMode(references: references(relation: "combines"))
        == .singleAndReferences)
    // The target-first strip outranks the relation it carries.
    #expect(try SourceImageMode(
        references: references(relation: "replaces", primaryIsTarget: true)) == .qwenEdit)
}

/// **Fails today**: `combines` fell into no arm at all -- the view's `else if`
/// drew the strip and hid the well, so img2img, strength and inpainting were
/// unreachable on every sd15 and sdxl recipe.
@Test func sd15CombinesKeepsBothWellsAndParksNeither() throws {
    let sd15 = try recipe("recipe-sd15.json")
    let block = try #require(sd15.capabilities.referenceImages(family: "sd15", model: "x"))
    #expect(block.sourceRelation == .combines)

    let mode = sd15.capabilities.sourceImageMode(family: "sd15", model: "x")
    #expect(mode == .singleAndReferences)
    #expect(mode.showsSourceWell)
    #expect(mode.showsReferenceStrip)
    #expect(!mode.replacesSourceImage)
}

@Test func anUnknownRelationDrawsBothRatherThanHidingTheStrip() throws {
    // A host NEWER than this build. Hiding the strip would make a whole
    // protocol unreachable; both wells let admission answer instead.
    #expect(try SourceImageMode(references: references(relation: "wormhole"))
        == .singleAndReferences)
}

// MARK: - The exclusive parking rule

@Test func theExclusiveWellsParkTheOneNotWrittenLast() {
    #expect(ExclusiveWells.resolve(hasSource: false, referenceCount: 0, lastWrite: nil).active == nil)
    #expect(ExclusiveWells.resolve(hasSource: true, referenceCount: 0, lastWrite: nil).parked
        == .references)
    #expect(ExclusiveWells.resolve(hasSource: false, referenceCount: 2, lastWrite: nil).parked
        == .source)
    // Both hold media: the last write decides, and an unmarked restore reads
    // as the source well.
    #expect(ExclusiveWells.resolve(hasSource: true, referenceCount: 1, lastWrite: nil).active
        == .source)
    #expect(ExclusiveWells.resolve(hasSource: true, referenceCount: 1, lastWrite: .references).active
        == .references)
}

// MARK: - What the request carries

@Test func anExclusiveRequestCarriesOneWellAndNeverBoth() throws {
    var draft = RenderDraft()
    draft.media.sourceMode = .singleOrReferences
    draft.media.sourceImage = "SRC"
    draft.media.editImages = ["REF"]
    draft.media.lastExclusiveWrite = .references
    #expect(draft.media.requestConditioning == .references)

    var request = RenderRequest.one(draft, model: "flux2-klein")
    #expect(request.sourceImage == nil)
    #expect(request.editImages == ["REF"])
    // Strength and the mask travel with a source image that ships.
    draft.media.maskImage = "MASK"
    request = RenderRequest.one(draft, model: "flux2-klein")
    #expect(request.strength == nil)
    #expect(request.maskImage == nil)

    draft.media.lastExclusiveWrite = .source
    request = RenderRequest.one(draft, model: "flux2-klein")
    #expect(request.sourceImage == "SRC")
    #expect(request.editImages == nil)
    #expect(request.maskImage == "MASK")
}

/// **Fails today**: the builder shipped `edit_images` whenever the strip held
/// anything, so an additive IP-Adapter render lost its img2img source.
@Test func anAdditiveRequestCarriesBothTheSourceAndTheReferences() {
    var draft = RenderDraft()
    draft.media.sourceMode = .singleAndReferences
    draft.media.sourceImage = "SRC"
    draft.media.editImages = ["REF"]
    draft.media.referenceWeight = 0.8
    #expect(draft.media.requestConditioning == .both)

    let request = RenderRequest.one(draft, model: "sd15")
    #expect(request.sourceImage == "SRC")
    #expect(request.editImages == ["REF"])
    #expect(request.referenceWeight == 0.8)
    #expect(request.strength != nil)
}

@Test func aReplacesRecipeParksTheSourceOutright() {
    var draft = RenderDraft()
    draft.media.sourceImage = "SRC"
    draft.media.editImages = ["REF"]
    draft.media.sourceMode = .references
    #expect(draft.media.requestConditioning == .references)
    #expect(RenderRequest.one(draft, model: "flux2-dev").sourceImage == nil)
    // Nothing parks for an exclusive-style layout question here: `replaces`
    // has no source path at all, so `exclusiveWells` answers nil.
    #expect(draft.media.exclusiveWells == nil)
}

// MARK: - The legacy fallback (01#4)

/// **Fails today**: an absent block and a `hidden` block read the same, so a
/// host predating the contract parked every reference and the one model whose
/// recipe REQUIRES one could never be submitted.
@Test func anAbsentBlockFallsBackToTheLegacyFamilyRule() throws {
    let legacy = try #require(
        ReferenceImagesCapability.legacy(family: "qwen-image-edit", model: "qwen-image-edit:q8"))
    #expect(legacy.required)
    #expect(legacy.primaryIsTarget)
    #expect(legacy.sourceRelation == .replaces)
    // An older host never advertised an adapter strength, so no slider.
    #expect(legacy.weight == nil)

    let flux2 = try #require(ReferenceImagesCapability.legacy(family: "flux2", model: "flux2-dev:q4"))
    #expect(flux2.maxCount == 4)
    #expect(!flux2.primaryIsTarget)

    // Klein's protocol shipped WITH the contract, so an older host has no
    // engine for it and the wells must not be offered.
    #expect(ReferenceImagesCapability.legacy(family: "flux2", model: "flux2-klein:q8") == nil)
    #expect(ReferenceImagesCapability.legacy(family: "flux", model: "flux-dev:q4") == nil)
}

@Test func aHiddenBlockIsTheServerSayingNoAndNeverFallsBack() throws {
    let hidden = try references(relation: "replaces", mode: "hidden")
    let capabilities = try MoldJSON.decoder.decode(RecipeCapabilities.self, from: Data("""
    {"reference_images":{"mode":"hidden","required":false,"primary_is_target":false,
     "source_relation":"replaces","reason":"not on this checkpoint"}}
    """.utf8))
    #expect(hidden.mode.isVisible == false)
    // Qwen's legacy rule WOULD answer for this family -- a hidden block must
    // still refuse, because only ABSENCE means an older server.
    #expect(capabilities.referenceImages(family: "qwen-image-edit", model: "q") == nil)
    #expect(capabilities.sourceImageMode(family: "qwen-image-edit", model: "q") == .single)
}
