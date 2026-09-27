import CoreGraphics
import Foundation
import ImageIO
import Testing

@testable import MoldClient

// The SAME goldens `mold_core::validation` and `studio/lib/referenceCanvas.test.ts`
// pin: diffusers `calculate_dimensions(1024 * 1024, w / h)`
// (`pipeline_qwenimage21.py:149-156`) under CPython 3.13. A client and the
// engine may never land on different sides of a tie.

@Test func roundingSendsHalvesToTheEvenNeighbourLikePython() {
    #expect(ReferenceCanvas.roundHalfToEven(0.5) == 0)
    #expect(ReferenceCanvas.roundHalfToEven(1.5) == 2)
    #expect(ReferenceCanvas.roundHalfToEven(2.5) == 2)
    #expect(ReferenceCanvas.roundHalfToEven(32.5) == 32)
    #expect(ReferenceCanvas.roundHalfToEven(33.5) == 34)
    #expect(ReferenceCanvas.roundHalfToEven(2.4999) == 2)
    #expect(ReferenceCanvas.roundHalfToEven(2.5001) == 3)
}

@Test(arguments: [
    (4225, 4096, 1024, 1024), (4096, 4225, 1024, 1024), (1600, 900, 1376, 768),
    (1024, 1024, 1024, 1024), (3000, 2000, 1248, 832), (1, 1, 1024, 1024),
    (640, 480, 1184, 896), (1080, 1920, 768, 1376),
])
func fitMatchesUpstreamCalculateDimensions(w: Int, h: Int, ew: Int, eh: Int) {
    let fitted = ReferenceCanvas.fitToTargetAreaTiesEven(
        width: w, height: h, targetArea: 1024 * 1024, alignment: 32)
    #expect(fitted == SourcePixels(width: ew, height: eh), "\(w)x\(h)")
}

@Test func aDegenerateAspectNeverProducesAZeroAxis() {
    let fitted = ReferenceCanvas.fitToTargetAreaTiesEven(
        width: 100_000, height: 1, targetArea: 1024 * 1024, alignment: 32)
    #expect(fitted.height == 32)
}

/// Qwen Image 2.1's advertised `resolution` bounds.
private let qwen21 = ResolutionProfile(
    domain: .dynamic, alignment: 32, minWidth: 64, minHeight: 64,
    maxPixels: 2400 * 1792, maxAxisPixels: 2752, offBucket: nil, aspectGroups: nil)

@Test(arguments: [
    (8000, 1000, 2752, 320), (1000, 8000, 320, 2752), (7300, 1000, 2752, 384),
    (100, 1, 2752, 64), (100_000, 1, 2752, 64), (1, 100_000, 64, 2752),
    (1920, 1080, 1376, 768),
])
func lastReferenceCanvasClampsIntoTheRecipe(w: Int, h: Int, ew: Int, eh: Int) {
    #expect(ReferenceCanvas.lastReference(width: w, height: h, limits: qwen21)
        == SourcePixels(width: ew, height: eh), "\(w)x\(h)")
}

@Test func theClampLeavesAFittingCanvasAloneAndHonoursThePixelCeiling() {
    #expect(ReferenceCanvas.clamp(width: 2400, height: 1792, limits: qwen21)
        == SourcePixels(width: 2400, height: 1792))
    #expect(ReferenceCanvas.clamp(width: 2752, height: 2752, limits: qwen21)
        == SourcePixels(width: 2048, height: 2048))
}

// MARK: - When the rule applies (`referenceCanvasSize`)

@Test func theLastReferenceSizesTheDefaultCanvas() {
    let size = ReferenceCanvas.size(
        rule: .lastReference, intent: .modelDefault,
        references: [SourcePixels(width: 1024, height: 1024), SourcePixels(width: 1600, height: 900)],
        resolution: qwen21)
    #expect(size == SourcePixels(width: 1376, height: 768))
}

@Test func theAreaIsUpstreamsNotTheReferencesOwn() {
    let size = ReferenceCanvas.size(
        rule: .lastReference, intent: .modelDefault,
        references: [SourcePixels(width: 4000, height: 3000)], resolution: qwen21)
    #expect(size == SourcePixels(width: 1184, height: 896))
}

@Test func theRuleLeavesTheCanvasAloneWhenItShould() {
    let wide = [SourcePixels(width: 1600, height: 900)]
    // An empty strip keeps whatever the canvas is (a restored draft hydrating).
    #expect(ReferenceCanvas.size(rule: .lastReference, intent: .modelDefault,
                                 references: [], resolution: qwen21) == nil)
    // A canvas somebody chose never moves.
    #expect(ReferenceCanvas.size(rule: .lastReference, intent: .manual,
                                 references: wide, resolution: qwen21) == nil)
    // No rule, or a rule newer than this build.
    #expect(ReferenceCanvas.size(rule: nil, intent: .modelDefault,
                                 references: wide, resolution: qwen21) == nil)
    #expect(ReferenceCanvas.size(rule: .unknown, intent: .modelDefault,
                                 references: wide, resolution: qwen21) == nil)
    // An unreadable last reference waits rather than guessing.
    #expect(ReferenceCanvas.size(rule: .lastReference, intent: .modelDefault,
                                 references: [wide[0], nil], resolution: qwen21) == nil)
}

// MARK: - Upright size from the bytes

@Test func aReferencesSizeIsReadFromItsBytes() throws {
    #expect(ReferenceCanvas.uprightPixels(ofBase64: encodedPicture(width: 117, height: 253))
        == SourcePixels(width: 117, height: 253))
    #expect(ReferenceCanvas.uprightPixels(ofBase64: "") == nil)
    #expect(ReferenceCanvas.uprightPixels(ofBase64: "not base64!") == nil)
}

@Test func aRotatedPhonePhotoIsReadUpright() throws {
    // Landscape pixels plus `Orientation = 6`: how a phone stores a portrait.
    let rotated = encodedPicture(width: 160, height: 90, type: "public.jpeg", orientation: 6)
    #expect(ReferenceCanvas.uprightPixels(ofBase64: rotated) == SourcePixels(width: 90, height: 160))
}

// MARK: - The draft follows its references

private func recipe() throws -> GenerationRecipe {
    let set = try MoldJSON.decoder.decode(
        GenerationProfileSet.self, from: RepoFixtures.fixture("recipe-qwen21.json"))
    return try #require(set.defaultRecipe)
}

/// A real, decodable picture -- ImageIO will not size a bare header.
private func encodedPicture(
    width: Int, height: Int, type: String = "public.png", orientation: Int? = nil
) -> String {
    let context = CGContext(
        data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
    let image = context.makeImage()!
    let data = NSMutableData()
    let destination = CGImageDestinationCreateWithData(data, type as CFString, 1, nil)!
    let properties = orientation.map { [kCGImagePropertyOrientation: $0] as CFDictionary }
    CGImageDestinationAddImage(destination, image, properties)
    CGImageDestinationFinalize(destination)
    return (data as Data).base64EncodedString()
}

private func pngHeader(width: Int, height: Int) -> String {
    encodedPicture(width: width, height: height)
}

@Test func addingReorderingAndRemovingReferencesReshapesADefaultCanvas() throws {
    let qwen = try recipe()
    var draft = RenderDraft().adopting(qwen, isNewModel: true)
    #expect(draft.canvasIntent == .modelDefault)
    draft.media.editImages = [pngHeader(width: 1024, height: 1024),
                              pngHeader(width: 1600, height: 900)]
    draft.followLastReference(recipe: qwen)
    #expect((draft.width, draft.height) == (1376, 768))
    // Reorder: the square one is now last.
    draft.media.editImages.swapAt(0, 1)
    draft.followLastReference(recipe: qwen)
    #expect((draft.width, draft.height) == (1024, 1024))
    // Remove the last: the wide one sets the canvas again.
    draft.media.editImages.removeLast()
    draft.followLastReference(recipe: qwen)
    #expect((draft.width, draft.height) == (1376, 768))
    // The intent is untouched: the canvas is still the model's to derive.
    #expect(draft.canvasIntent == .modelDefault)
}

@Test func aChosenCanvasIgnoresItsReferences() throws {
    let qwen = try recipe()
    var draft = RenderDraft().adopting(qwen, isNewModel: true)
    draft.canvasIntent = .manual
    draft.width = 2048
    draft.height = 2048
    draft.media.editImages = [pngHeader(width: 1600, height: 900)]
    draft.followLastReference(recipe: qwen)
    #expect((draft.width, draft.height) == (2048, 2048))
}

@Test func adoptingQwen21WithReferencesStagedTakesTheLastOnesShape() throws {
    var draft = RenderDraft()
    draft.media.editImages = [pngHeader(width: 1080, height: 1920)]
    let adopted = draft.adopting(try recipe(), isNewModel: true)
    #expect((adopted.width, adopted.height) == (768, 1376))
}
