import Foundation
import MoldClient
import Testing

@testable import Mold

/// IP-Adapter's pull, resolved from `capabilities.reference_images.weight`
/// alone. **Fails today**: there is no reference-weight control anywhere in
/// the app, so a `combines` recipe advertising a weight offers nothing.
@MainActor
struct ReferenceWeightTests {
    /// A recipe with the reference block spelled out. `weight` is the axis
    /// under test, so it is the parameter; everything else is the neutral
    /// shape `RecipeCapabilities` decodes from an otherwise-empty object.
    private func recipe(relation: String, weight: String?) -> GenerationRecipe {
        let references = """
        {"mode": "adjustable", "required": false, "max_count": 1,
         "primary_is_target": false, "source_relation": "\(relation)",
         "reason": null, "weight": \(weight ?? "null")}
        """
        let json = """
        {"id": "r", "label": "R",
         "defaults": {"width": 512, "height": 512, "steps": 20, "guidance": 7.0,
                      "frames": null, "fps": null, "negative_prompt": null},
         "resolution": {"domain": "dynamic", "alignment": 8, "min_width": 256, "min_height": 256,
                        "max_pixels": null, "max_axis_pixels": null, "off_bucket": null,
                        "aspect_groups": null},
         "steps": {"default": 20, "min": 1, "max": 100, "step": 1, "recommended": null,
                   "mode": "adjustable", "note": null},
         "guidance": {"default": 7.0, "min": 0, "max": 20, "step": 0.1, "mode": "adjustable", "note": null},
         "temporal": null,
         "capabilities": {"reference_images": \(references)},
         "request_selector": null}
        """
        return try! MoldJSON.decoder.decode(GenerationRecipe.self, from: Data(json.utf8))
    }

    private let advertised = """
    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05, "mode": "adjustable", "note": null}
    """

    private func media(references: Int) -> DraftMedia {
        var media = DraftMedia()
        media.sourceMode = .singleAndReferences
        media.editImages = (0 ..< references).map { "REF\($0)" }
        return media
    }

    @Test func aCombinesRecipeWithAReferenceAttachedOffersTheAdvertisedRange() throws {
        let control = try #require(ReferenceWeightControl.resolve(
            recipe: recipe(relation: "combines", weight: advertised),
            model: FakeFixtures.model("sd15:fp16", family: "sd15"),
            media: media(references: 1)))
        // The range, step and default are the CAPABILITY's, never a constant
        // of this app's -- that is the whole reason the block carries them.
        #expect(control.min == 0.0)
        #expect(control.max == 2.0)
        #expect(control.step == 0.05)
        #expect(control.default == 1.0)
    }

    @Test func anEmptyStripOffersNoWeightAtAll() {
        #expect(ReferenceWeightControl.resolve(
            recipe: recipe(relation: "combines", weight: advertised),
            model: FakeFixtures.model("sd15:fp16", family: "sd15"),
            media: media(references: 0)) == nil)
    }

    /// The `weight` field is an `Option` on the block. A host that advertises
    /// references and no weight is an OLDER host, not a refusal: its
    /// references still ship and only this slider is absent.
    @Test func aHostThatAdvertisesNoWeightShowsNoSlider() {
        #expect(ReferenceWeightControl.resolve(
            recipe: recipe(relation: "combines", weight: nil),
            model: FakeFixtures.model("sd15:fp16", family: "sd15"),
            media: media(references: 1)) == nil)
    }

    /// An EXCLUSIVE recipe holding a source picture parks the strip, so the
    /// request carries no references -- and a weight over conditioning that
    /// does not ship is furniture. `requestConditioning`, never
    /// `editImages.isEmpty`.
    @Test func aParkedStripOnAnExclusiveRecipeOffersNoWeight() {
        var media = DraftMedia()
        media.sourceMode = .singleOrReferences
        media.editImages = ["REF"]
        media.sourceImage = "SRC"
        media.lastExclusiveWrite = .source
        #expect(ReferenceWeightControl.resolve(
            recipe: recipe(relation: "exclusive", weight: advertised),
            model: FakeFixtures.model("flux2-klein:q8", family: "flux2"),
            media: media) == nil)
    }

    /// A slider sitting at the recipe's own default has nothing to reset, so
    /// no menu is attached at all -- the unpainted-mask-row rule.
    @Test func theWeightRowOffersResetOnlyOnceItHasBeenMoved() {
        #expect(GenerateMenus.referenceWeight(isAtDefault: true).isEmpty)
        #expect(GenerateMenus.referenceWeight(isAtDefault: false).map(\.title) == ["Reset Weight"])
    }
}
