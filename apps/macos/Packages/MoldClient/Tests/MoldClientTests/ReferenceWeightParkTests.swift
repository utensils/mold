import Foundation
import Testing

@testable import MoldClient

/// The IP-Adapter weight across a model switch.
///
/// **Fails today**: `DraftMedia.reconcile` set `referenceWeight = nil`
/// outright, so switching to a model with no reference protocol and back
/// silently threw away a deliberately-chosen strength -- exactly the loss
/// parking exists to prevent for every other conditioning input.
struct ReferenceWeightParkTests {
    private func recipe(references: String) -> GenerationRecipe {
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

    private let weighted = """
    {"mode": "adjustable", "required": false, "max_count": 1, "primary_is_target": false,
     "source_relation": "combines", "reason": null,
     "weight": {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                "mode": "adjustable", "note": null}}
    """
    /// References, and no weight control -- every `replaces` recipe, and any
    /// host that predates the field.
    private let unweighted = """
    {"mode": "adjustable", "required": false, "max_count": 4, "primary_is_target": false,
     "source_relation": "replaces", "reason": null, "weight": null}
    """

    @Test func aRecipeWithNoWeightControlParksTheWeightAndOneWithItHandsItBack() {
        var draft = RenderDraft()
        draft.media.editImages = ["REF"]
        draft.media.referenceWeight = 1.6

        let parked = draft.adopting(recipe(references: unweighted), isNewModel: false)
        #expect(parked.media.referenceWeight == nil)
        #expect(parked.media.parked.referenceWeight == 1.6)

        let restored = parked.adopting(recipe(references: weighted), isNewModel: false)
        #expect(restored.media.referenceWeight == 1.6)
        #expect(restored.media.parked.referenceWeight == nil)
    }

    /// A live value always beats a parked one -- parking is a rescue, not a
    /// history.
    @Test func aWeightAlreadySetIsNotOverwrittenByAParkedOne() {
        var draft = RenderDraft()
        draft.media.editImages = ["REF"]
        draft.media.parked.referenceWeight = 0.2
        draft.media.referenceWeight = 1.9

        let adopted = draft.adopting(recipe(references: weighted), isNewModel: false)
        #expect(adopted.media.referenceWeight == 1.9)
        #expect(adopted.media.parked.referenceWeight == 0.2)
    }

    /// A weight only ever reaches the wire alongside references that actually
    /// ship -- the request builder already gates both on the same answer.
    @Test func aParkedWeightNeverReachesTheWire() {
        var draft = RenderDraft()
        draft.media.editImages = ["REF"]
        draft.media.referenceWeight = 1.6
        let parked = draft.adopting(recipe(references: unweighted), isNewModel: false)
        #expect(parked.request(model: "m").referenceWeight == nil)
    }
}
