import Foundation
import Testing

@testable import MoldClient

// `recipe-qwen21.json` is `qwen-image-2.1:bf16`'s profile, copied verbatim
// from `docs/generated/generation-profiles-v1.json` (the generated contract
// the server emits), 2026-09-27.

private func qwen21() throws -> GenerationRecipe {
    let set = try MoldJSON.decoder.decode(
        GenerationProfileSet.self, from: RepoFixtures.fixture("recipe-qwen21.json"))
    return try #require(set.defaultRecipe)
}

// MARK: - reference_images: canvas and formats

@Test func qwen21AdvertisesTenOrderedReferencesThatSetTheCanvas() throws {
    let references = try #require(try qwen21().capabilities.referenceImages)
    #expect(references.maxCount == 10)
    #expect(references.sourceRelation == .replaces)
    #expect(references.canvas == .lastReference)
    #expect(references.formats == ["png", "jpeg", "webp"])
}

@Test func anOlderReferenceBlockHasNoCanvasRuleAndNoFormats() throws {
    let json = Data("""
    {"mode":"adjustable","required":false,"max_count":4,"primary_is_target":false,
     "source_relation":"exclusive"}
    """.utf8)
    let block = try MoldJSON.decoder.decode(ReferenceImagesCapability.self, from: json)
    #expect(block.canvas == nil)
    #expect(block.formats == nil)
}

@Test func aCanvasRuleNewerThanThisBuildIsNotLastReference() throws {
    let json = Data("""
    {"mode":"adjustable","required":false,"primary_is_target":false,
     "source_relation":"replaces","canvas":"first-reference"}
    """.utf8)
    let block = try MoldJSON.decoder.decode(ReferenceImagesCapability.self, from: json)
    #expect(block.canvas == .unknown)
}

// MARK: - transparency

@Test func qwen21OffersAnAdjustableTransparentBackground() throws {
    let block = try #require(try qwen21().capabilities.transparency)
    #expect(block.mode == .adjustable)
    #expect(block.default == false)
    #expect(block.formats == ["png", "webp"])
    #expect(block.nativeAlpha == true)
    let control = try #require(try qwen21().capabilities.transparencyControl)
    #expect(control.formats == ["png", "webp"])
}

@Test func aHiddenOrAbsentTransparencyBlockOffersNoToggle() throws {
    // The z-image fixture predates the block entirely: an OLDER server, and
    // there is nothing to fall back to.
    let zimage = try MoldJSON.decoder.decode(
        GenerationProfileSet.self, from: RepoFixtures.fixture("recipe-zimage.json"))
    #expect(try #require(zimage.defaultRecipe).capabilities.transparencyControl == nil)

    let hidden = Data("""
    {"mode":"hidden","default":false,"formats":[],"native_alpha":false,
     "reason":"This model does not render transparent backgrounds (transparent_background)."}
    """.utf8)
    let block = try MoldJSON.decoder.decode(TransparencyCapability.self, from: hidden)
    #expect(block.control == nil)
    #expect(block.reason?.contains("transparent") == true)
}

@Test func anAdjustableBlockWithNoAlphaFormatOffersNoToggle() throws {
    let json = Data("""
    {"mode":"adjustable","default":false,"formats":[],"native_alpha":false}
    """.utf8)
    let block = try MoldJSON.decoder.decode(TransparencyCapability.self, from: json)
    #expect(block.control == nil)
}

// MARK: - Output: WebP is a still here

@Test func qwen21OffersWebPAsAStillFormat() throws {
    let output = try #require(try qwen21().capabilities.output)
    #expect(output.formats.contains("webp"))
    #expect(try qwen21().temporal == nil)
}

@Test func aFinishedWebPStillIsDrawnAsAPicture() {
    // Stills are not video: a WebP result opens as a picture on the canvas,
    // and a still WebP print is a picture in the Library.
    #expect(BatchResult(filename: "mold-qwen-1.webp").playbackKind == .picture)
}
