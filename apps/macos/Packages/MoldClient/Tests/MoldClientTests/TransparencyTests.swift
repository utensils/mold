import CoreGraphics
import Foundation
import ImageIO
import Testing

@testable import MoldClient

private func qwen21() throws -> GenerationRecipe {
    let set = try MoldJSON.decoder.decode(
        GenerationProfileSet.self, from: RepoFixtures.fixture("recipe-qwen21.json"))
    return try #require(set.defaultRecipe)
}

private func zimage() throws -> GenerationRecipe {
    let set = try MoldJSON.decoder.decode(
        GenerationProfileSet.self, from: RepoFixtures.fixture("recipe-zimage.json"))
    return try #require(set.defaultRecipe)
}

private func wire(_ draft: RenderDraft) throws -> [String: Any] {
    let body = try MoldJSON.encoder.encode(RenderRequest.one(draft, model: "qwen-image-2.1:bf16"))
    return try #require(try JSONSerialization.jsonObject(with: body) as? [String: Any])
}

// MARK: - The request

@Test func anOrdinaryRenderCarriesNoTransparencyField() throws {
    let draft = RenderDraft().adopting(try qwen21(), isNewModel: true)
    #expect(try wire(draft)["transparent_background"] == nil)
}

@Test func theToggleSendsTrueOnlyWhereTheRecipeOffersIt() throws {
    let qwen = try qwen21()
    var draft = RenderDraft().adopting(qwen, isNewModel: true)
    draft = draft.settingTransparentBackground(true, output: qwen.capabilities.output)
    #expect(try wire(draft)["transparent_background"] as? Bool == true)

    // Parked, not dropped: a model with no toggle leaves the wire clean, and
    // coming back to one that has it hands the choice back.
    let away = draft.adopting(try zimage(), isNewModel: true)
    #expect(away.transparentBackground == true)
    #expect(try wire(away)["transparent_background"] == nil)
    let back = away.adopting(qwen, isNewModel: true)
    #expect(try wire(back)["transparent_background"] as? Bool == true)
}

@Test func turningItOnMovesJPEGToTheFirstAlphaFormat() throws {
    let qwen = try qwen21()
    var draft = RenderDraft().adopting(qwen, isNewModel: true)
    draft.outputFormat = "jpeg"
    draft = draft.settingTransparentBackground(true, output: qwen.capabilities.output)
    #expect(draft.outputFormat == "png")
    // WebP carries alpha, so it stays.
    draft.outputFormat = "webp"
    draft = draft.settingTransparentBackground(true, output: qwen.capabilities.output)
    #expect(draft.outputFormat == "webp")
    // Off leaves the format alone.
    draft.outputFormat = "jpeg"
    draft = draft.settingTransparentBackground(false, output: qwen.capabilities.output)
    #expect(draft.outputFormat == "jpeg")
    #expect(try wire(draft)["transparent_background"] == nil)
}

@Test func whileOnJPEGIsNotAFormatThePickerOffers() throws {
    let qwen = try qwen21()
    var draft = RenderDraft().adopting(qwen, isNewModel: true)
    #expect(draft.transparencyBlocksFormat("jpeg") == false)
    draft = draft.settingTransparentBackground(true, output: qwen.capabilities.output)
    #expect(draft.transparencyBlocksFormat("jpeg") == true)
    #expect(draft.transparencyBlocksFormat("png") == false)
    #expect(draft.transparencyBlocksFormat("webp") == false)
    // And a stray pick of it (a restored draft) is moved on adopt.
    draft.outputFormat = "jpeg"
    #expect(draft.adopting(qwen, isNewModel: false).outputFormat == "png")
}

@Test func whileOnPickingJPEGIsRefused() throws {
    // A menu-style Picker on macOS ignores `.disabled` on its items, so the
    // JPEG row stays clickable: the pick itself must be refused, or the
    // request carries a pair admission rejects.
    let qwen = try qwen21()
    let output = qwen.capabilities.output
    var draft = RenderDraft().adopting(qwen, isNewModel: true)
    draft = draft.settingTransparentBackground(true, output: output)
    // The pick is a no-op: the recipe's own default (PNG) still applies.
    #expect(draft.selectingOutputFormat("jpeg", output: output) == draft)
    #expect(draft.selectingOutputFormat("webp", output: output).outputFormat == "webp")
    // Off, JPEG is an ordinary pick again.
    draft = draft.settingTransparentBackground(false, output: output)
    #expect(draft.selectingOutputFormat("jpeg", output: output).outputFormat == "jpeg")
}

// MARK: - Reuse and persistence

private func metadata(_ json: String) throws -> OutputMetadata {
    try MoldJSON.decoder.decode(OutputMetadata.self, from: Data(json.utf8))
}

@Test func reuseRestoresTheToggleFromTheRequestNotTheFile() throws {
    let asked = try metadata(#"{"prompt":"a cup","model":"qwen-image-2.1:bf16","transparent_background":true}"#)
    #expect(RenderDraft(reusing: asked).transparentBackground == true)
    // `has_alpha` describes the FILE -- a transparent reference edited with
    // the toggle off -- and must not turn the toggle on.
    let file = try metadata(#"{"prompt":"a cup","model":"qwen-image-2.1:bf16","has_alpha":true}"#)
    #expect(RenderDraft(reusing: file).transparentBackground == false)
}

@Test func theDraftDescriptorRoundTripsTheToggle() throws {
    var draft = RenderDraft()
    draft.transparentBackground = true
    let descriptor = DraftDescriptor(draft, model: nil, family: nil, recipeID: nil)
    let data = try MoldJSON.localEncoder.encode(descriptor)
    var restored = RenderDraft()
    try MoldJSON.localDecoder.decode(DraftDescriptor.self, from: data).apply(to: &restored)
    #expect(restored.transparentBackground == true)
}

@Test func aDescriptorWrittenBeforeTheToggleStillRestores() throws {
    let descriptor = DraftDescriptor(RenderDraft(), model: nil, family: nil, recipeID: nil)
    var object = try #require(
        try JSONSerialization.jsonObject(with: MoldJSON.localEncoder.encode(descriptor))
            as? [String: Any])
    object.removeValue(forKey: "transparentBackground")
    let old = try JSONSerialization.data(withJSONObject: object)
    var restored = RenderDraft()
    restored.transparentBackground = true
    try MoldJSON.localDecoder.decode(DraftDescriptor.self, from: old).apply(to: &restored)
    #expect(restored.transparentBackground == false)
}

// MARK: - The checkerboard

@Test func theAlphaBedFollowsTheFileOrTheRequest() throws {
    #expect(try metadata(#"{"has_alpha":true}"#).showsAlphaBed)
    #expect(try metadata(#"{"transparent_background":true}"#).showsAlphaBed)
    #expect(try metadata(#"{"has_alpha":false,"transparent_background":false}"#).showsAlphaBed == false)
    #expect(try metadata(#"{"prompt":"an opaque print"}"#).showsAlphaBed == false)
}

// MARK: - A finished result, read from its own bytes

private func png(opaque: Bool, metadata: String? = nil) -> Data {
    let context = CGContext(
        data: nil, width: 4, height: 4, bitsPerComponent: 8, bytesPerRow: 0,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: (opaque ? CGImageAlphaInfo.noneSkipLast : .premultipliedLast).rawValue)!
    let data = NSMutableData()
    let destination = CGImageDestinationCreateWithData(data, "public.png" as CFString, 1, nil)!
    CGImageDestinationAddImage(destination, context.makeImage()!, nil)
    CGImageDestinationFinalize(destination)
    guard let metadata else { return data as Data }
    // A `tEXt` chunk right after IHDR, the way mold embeds `mold:parameters`.
    var file = data as Data
    let text = Data("mold:parameters".utf8) + Data([0]) + Data(metadata.utf8)
    var chunk = Data()
    withUnsafeBytes(of: UInt32(text.count).bigEndian) { chunk.append(contentsOf: $0) }
    chunk.append(Data("tEXt".utf8) + text)
    chunk.append(contentsOf: [0, 0, 0, 0])
    file.insert(contentsOf: chunk, at: 33)
    return file
}

@Test func aResultCarryingAlphaIsDrawnOnTheBed() {
    #expect(ResultAlpha.showsBed(png(opaque: false), named: "a.png"))
    #expect(ResultAlpha.showsBed(png(opaque: true), named: "a.png") == false)
    // The request's own flag, embedded in the file, answers too.
    #expect(ResultAlpha.showsBed(
        png(opaque: true, metadata: #"{"prompt":"x","transparent_background":true}"#),
        named: "a.png"))
    #expect(ResultAlpha.showsBed(Data("not a picture".utf8), named: "a.png") == false)
}
