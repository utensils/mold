import Foundation
import MoldClient
import Testing

@testable import Mold

/// Which picture wells the pane draws, on the pure layout gate -- no view
/// needed, the same idiom `GenerateInspectorTests` uses.
///
/// The predicate `PromptPanel.showsSourceWell(for:)` was always right; the
/// `else if` it fed was not (finding 02#1), which is the argument for testing
/// the BRANCH and not only the predicate.
@MainActor
struct ImageConditioningWellsTests {
    /// A recipe carrying one `reference_images` block, decoded the way the
    /// wire produces it -- `GenerationRecipe` has no public memberwise init.
    private func recipe(references: String) -> GenerationRecipe {
        let json = #"""
        {"id": "r", "label": "R",
         "defaults": {"width": 1024, "height": 1024, "steps": 20, "guidance": 3.5,
                      "frames": null, "fps": null, "negative_prompt": null},
         "resolution": {"domain": "dynamic", "alignment": 16, "min_width": 256, "min_height": 256,
                        "max_pixels": null, "max_axis_pixels": null, "off_bucket": null,
                        "aspect_groups": null},
         "steps": {"default": 20, "min": 1, "max": 50, "step": 1, "recommended": null,
                   "mode": "adjustable", "note": null},
         "guidance": {"default": 3.5, "min": 0, "max": 10, "step": 0.1, "mode": "adjustable",
                      "note": null},
         "temporal": null,
         "capabilities": {"reference_images": \#(references), "source_image": null}}
        """#
        return try! MoldJSON.decoder.decode(GenerationRecipe.self, from: Data(json.utf8))
    }

    private func block(_ relation: String, primaryIsTarget: Bool = false) -> String {
        #"""
        {"mode": "adjustable", "required": false, "max_count": 4,
         "primary_is_target": \#(primaryIsTarget), "source_relation": "\#(relation)"}
        """#
    }

    /// **Fails today**: `PromptPanel` renders the strip OR the well, so the
    /// two families most people use for img2img lost their source picture,
    /// their strength and their mask the moment IP-Adapter was advertised.
    @Test func anAdditiveRecipeDrawsBothWellsAndParksNeither() {
        let layout = ImageConditioningWells.layout(
            recipe: recipe(references: block("combines")), model: nil, media: DraftMedia())
        #expect(layout.showsSourceWell)
        #expect(layout.references != nil)
        #expect(layout.parked == nil)
        #expect(layout.note == nil)
    }

    @Test func anExclusiveRecipeDrawsBothAndParksTheOneNotInUse() {
        var media = DraftMedia()
        media.editImages = ["REF"]
        let layout = ImageConditioningWells.layout(
            recipe: recipe(references: block("exclusive")), model: nil, media: media)
        #expect(layout.showsSourceWell)
        #expect(layout.references != nil)
        #expect(layout.parked == .source)
        #expect(layout.note == ExclusiveWells.note)
    }

    @Test func aReplacesRecipeDrawsTheStripAlone() {
        for relation in [block("replaces"), block("replaces", primaryIsTarget: true)] {
            let layout = ImageConditioningWells.layout(
                recipe: recipe(references: relation), model: nil, media: DraftMedia())
            #expect(!layout.showsSourceWell)
            #expect(layout.references != nil)
            #expect(layout.parked == nil)
        }
    }

    @Test func aRecipeWithNoReferenceProtocolDrawsTheSourceWellAlone() {
        let layout = ImageConditioningWells.layout(
            recipe: recipe(references: "null"), model: nil, media: DraftMedia())
        #expect(layout.showsSourceWell)
        #expect(layout.references == nil)
    }

    /// **Fails today**: an absent block read as `hidden`, so reference
    /// editing died on a host that predates the contract (finding 01#4).
    @Test func anOlderHostStillDrawsQwensTargetStrip() {
        let older = recipe(references: "null")
        let qwen = FakeFixtures.model("qwen-image-edit:q8", family: "qwen-image-edit")
        let layout = ImageConditioningWells.layout(recipe: older, model: qwen, media: DraftMedia())
        let references = try? #require(layout.references)
        #expect(references?.primaryIsTarget == true)
        #expect(references?.required == true)
        // `replaces`, so the strip stands alone.
        #expect(!layout.showsSourceWell)

        // And a family the legacy rule says nothing about is unchanged.
        let flux = FakeFixtures.model("flux-dev:q4", family: "flux")
        let plain = ImageConditioningWells.layout(recipe: older, model: flux, media: DraftMedia())
        #expect(plain.references == nil)
        #expect(plain.showsSourceWell)
    }

    /// A `hidden` block is the server SAYING NO; only ABSENCE falls back.
    @Test func aHiddenBlockNeverReachesTheLegacyRule() {
        let hidden = recipe(references: #"""
        {"mode": "hidden", "required": false, "primary_is_target": false,
         "source_relation": "replaces", "reason": "not on this checkpoint"}
        """#)
        let qwen = FakeFixtures.model("qwen-image-edit:q8", family: "qwen-image-edit")
        let layout = ImageConditioningWells.layout(recipe: hidden, model: qwen, media: DraftMedia())
        #expect(layout.references == nil)
        #expect(layout.showsSourceWell)
    }

    // MARK: - Qwen Image 2.1: numbered, and the last one sets the canvas

    private var lastReference: String {
        #"""
        {"mode": "adjustable", "required": false, "max_count": 10,
         "primary_is_target": false, "source_relation": "replaces",
         "canvas": "last-reference", "formats": ["png", "jpeg", "webp"]}
        """#
    }

    @Test func aLastReferenceRecipeSaysTheLastPictureSetsTheCanvas() {
        var media = DraftMedia()
        let empty = ImageConditioningWells.layout(
            recipe: recipe(references: lastReference), model: nil, media: media)
        #expect(empty.setsCanvas)
        // Nothing staged, nothing to explain.
        #expect(empty.note == nil)
        media.editImages = ["A", "B"]
        let staged = ImageConditioningWells.layout(
            recipe: recipe(references: lastReference), model: nil, media: media)
        #expect(staged.note == ImageConditioningWells.canvasNote)
    }

    @Test func noOtherRecipeClaimsTheCanvas() {
        for relation in ["replaces", "exclusive", "combines"] {
            let layout = ImageConditioningWells.layout(
                recipe: recipe(references: block(relation)), model: nil, media: DraftMedia())
            #expect(layout.setsCanvas == false, "\(relation)")
        }
    }

    @Test func referencesAreNumberedTheWayThePromptAddressesThem() {
        #expect(ReferenceStrip.ordinal(index: 0, base: 0) == 1)
        #expect(ReferenceStrip.ordinal(index: 2, base: 0) == 3)
        // An additive recipe ships the source first, so its first reference
        // is image 2 while a source is held.
        var media = DraftMedia()
        media.sourceImage = "SRC"
        let additive = ImageConditioningWells.layout(
            recipe: recipe(references: block("combines")), model: nil, media: media)
        #expect(additive.ordinalBase == 1)
        #expect(ReferenceStrip.ordinal(index: 0, base: additive.ordinalBase) == 2)
        let exclusive = ImageConditioningWells.layout(
            recipe: recipe(references: block("exclusive")), model: nil, media: media)
        #expect(exclusive.ordinalBase == 0)
    }

    @Test func aReferenceWellTakesOnlyTheContainersTheRecipeReads() throws {
        let qwen = try #require(recipe(references: lastReference).capabilities.referenceImages)
        #expect(ReferenceStrip.accepting(qwen) == [
            "public.png", "public.jpeg", "org.webmproject.webp"])
        // A recipe that predates the list reads PNG and JPEG; anything else is
        // converted here (to PNG, which keeps alpha) rather than refused.
        let older = try #require(recipe(references: block("exclusive")).capabilities.referenceImages)
        #expect(ReferenceStrip.accepting(older) == ["public.png", "public.jpeg"])
    }
}
