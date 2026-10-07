import CoreGraphics
import Foundation
import ImageIO
import Testing
@testable import MoldClient

@Suite struct StillReferenceMutationTests {
    private func capability(_ extra: String = "") throws -> ReferenceImagesCapability {
        try MoldJSON.decoder.decode(ReferenceImagesCapability.self, from: Data("""
        {"mode":"adjustable","required":true,"max_count":10,"primary_is_target":false,
         "source_relation":"replaces"\(extra)}
        """.utf8))
    }

    @Test func processingBudgetsDoNotRefuseOriginalReferences() throws {
        let cap = try capability(",\"max_pixels_single\":100,\"max_pixels_multi\":50")
        #expect(cap.maxPixelsSingle == 100)
        #expect(cap.stillRefusal(images: [png(10, 10)]) == nil)
        #expect(cap.stillRefusal(images: [png(10, 10), png(5, 5)]) == nil)
        #expect(cap.stillRefusal(images: []) != nil)
        #expect(cap.stillRefusal(images: ["bad bytes"]) != nil)
    }

    @Test func ordinaryOversizedReferencesQueueAcrossProcessingBudgets() throws {
        for budget in [1_048_576, 4_096_576] {
            let cap = try capability(",\"max_pixels_single\":\(budget),\"max_pixels_multi\":1048576")
            #expect(cap.stillRefusal(images: [png(1280, 853)]) == nil)
            #expect(cap.stillRefusal(images: [png(1280, 853), png(1600, 900)]) == nil)
        }
        #expect(try capability().stillRefusal(images: [png(1280, 853)]) == nil)
    }

    @Test func ingestionBoundsRemainIndependentOfProcessingBudgets() throws {
        let cap = try capability()
        #expect(cap.stillRefusal(images: [png(1, 201)]) != nil)
        #expect(cap.stillRefusal(images: [png(16_385, 1)]) != nil)
        #expect(cap.stillRefusal(images: [png(200, 1)]) == nil)
    }

    @Test func sharedMutationsRecomputeDefaultCanvasAndKeepChosenCanvas() throws {
        let profile = try MoldJSON.decoder.decode(GenerationProfileSet.self,
            from: RepoFixtures.fixture("recipe-qwen21.json"))
        let recipe = try #require(profile.defaultRecipe)
        var draft = RenderDraft().adopting(recipe, isNewModel: true)
        draft.media.editImages = [png(100, 100), png(160, 90)]
        DraftPictureAttachment.moveReference(from: 0, to: 1, in: &draft, recipe: recipe)
        #expect(draft.width == draft.height)
        DraftPictureAttachment.removeReference(at: 1, from: &draft, recipe: recipe)
        #expect(draft.width > draft.height)
        let data = Data(base64Encoded: png(90, 160))!
        DraftPictureAttachment.replaceReference(ImportedPicture(encoded: data.base64EncodedString(), name: "portrait", data: data),
            at: 0, in: &draft, recipe: recipe)
        #expect(draft.height > draft.width)
        draft.canvasIntent = .manual
        draft.width = 512; draft.height = 512
        DraftPictureAttachment.removeReference(at: 0, from: &draft, recipe: recipe)
        #expect(draft.width == 512 && draft.height == 512)
        DraftPictureAttachment.removeReference(at: 42, from: &draft, recipe: recipe)
        #expect(draft.media.editImages.isEmpty)
    }

    @Test func retainedStillAuthorityExemptsOnlyMissingRequiredBytes() throws {
        var document = try JSONSerialization.jsonObject(with: RepoFixtures.fixture("recipe-qwen21.json")) as! [String: Any]
        var recipes = document["recipes"] as! [[String: Any]]
        var caps = recipes[0]["capabilities"] as! [String: Any]
        caps["reference_images"] = ["mode": "adjustable", "required": true, "max_count": 1,
            "primary_is_target": true, "source_relation": "replaces", "max_pixels_single": 100]
        recipes[0]["capabilities"] = caps; document["recipes"] = recipes
        let profile = try MoldJSON.decoder.decode(GenerationProfileSet.self, from: JSONSerialization.data(withJSONObject: document))
        let recipe = try #require(profile.defaultRecipe)
        var draft = RenderDraft().adopting(recipe, isNewModel: true)
        draft.prompt = "Edit this"
        #expect(draft.refusal(for: recipe) != nil)
        #expect(draft.refusal(for: recipe, retainedFields: [.editImages]) == nil)
        #expect(draft.refusal(for: recipe, retainedFields: [.sourceImage]) != nil)
        draft.media.editImages = ["bad bytes"]
        #expect(draft.refusal(for: recipe, retainedFields: [.editImages]) != nil)
        draft.media.editImages = [png(11, 10)]
        #expect(draft.refusal(for: recipe, retainedFields: [.editImages]) == nil)
        draft.media.editImages = [png(5, 5), png(5, 5)]
        #expect(draft.refusal(for: recipe, retainedFields: [.editImages]) != nil)
    }

    private func png(_ width: Int, _ height: Int) -> String {
        let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8,
            bytesPerRow: 0, space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        let data = NSMutableData()
        let destination = CGImageDestinationCreateWithData(data, "public.png" as CFString, 1, nil)!
        CGImageDestinationAddImage(destination, context.makeImage()!, nil)
        CGImageDestinationFinalize(destination)
        return (data as Data).base64EncodedString()
    }
}
