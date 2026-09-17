import Foundation
import Testing

@testable import MoldClient

/// #1166: the canvas follows an attached source per the RECORDED intent, and
/// never per a comparison with the size the canvas happens to hold.
/// **Fails today**: the draft recorded no intent and no source moved it.
struct CanvasIntentTests {
    private func recipe(_ groups: String) -> GenerationRecipe {
        let json = """
        {"id": "r", "label": "R",
         "defaults": {"width": 1024, "height": 1024, "steps": 20, "guidance": 3.5,
                      "frames": null, "fps": null, "negative_prompt": null},
         "resolution": {"domain": "buckets", "alignment": 64, "min_width": 256,
                        "min_height": 256, "max_pixels": null, "max_axis_pixels": null,
                        "off_bucket": "reject", "aspect_groups": \(groups)},
         "steps": {"default": 20, "min": 1, "max": 100, "step": 1, "recommended": null,
                   "mode": "adjustable", "note": null},
         "guidance": {"default": 3.5, "min": 0, "max": 10, "step": 0.1, "mode": "adjustable",
                      "note": null},
         "temporal": null, "capabilities": {}, "request_selector": null}
        """
        return try! MoldJSON.decoder.decode(GenerationRecipe.self, from: Data(json.utf8))
    }

    private let ladder = """
    [{"id": "square", "label": "1:1",
      "presets": [{"id": "s1", "width": 1024, "height": 1024, "tier": "recommended"},
                  {"id": "s2", "width": 512, "height": 512, "tier": null}]},
     {"id": "wide", "label": "16:9",
      "presets": [{"id": "w1", "width": 1344, "height": 768, "tier": "recommended"}]}]
    """

    /// A first picture lands on the model's own ladder rather than on the
    /// picture's raw pixels, and RECORDS that the canvas is now following it.
    @Test func aFirstSourceMovesTheCanvasOntoTheNearestTier() {
        var draft = RenderDraft()
        draft.attachSourceShape((1920, 1080), recipe: recipe(ladder), replaced: true)
        #expect(draft.canvasIntent == .source)
        #expect(draft.width == 1344)
        #expect(draft.height == 768)
    }

    /// No presets at that aspect means the NEAREST one, measured
    /// logarithmically so portrait and landscape are treated symmetrically --
    /// a portrait source on a ladder with no portrait tier lands square, not
    /// on the widest thing available.
    @Test func aShapeTheLadderDoesNotHaveLandsOnTheNearestItDoes() {
        var draft = RenderDraft()
        draft.attachSourceShape((1080, 1920), recipe: recipe(ladder), replaced: true)
        #expect(draft.width == 1024)
        #expect(draft.height == 1024)
    }

    /// The whole point of #1166. With the SAME picture still attached -- a
    /// model switch, a recipe switch, a re-fit -- a canvas somebody chose is
    /// never moved, and the reason is the recorded intent and not the size,
    /// which here is deliberately the 1024x1024 the model also defaults to.
    @Test func aChosenCanvasIsNeverMovedEvenWhenItLooksLikeTheDefault() {
        var draft = RenderDraft()
        draft.canvasIntent = .manual
        draft.width = 1024
        draft.height = 1024
        draft.attachSourceShape((1920, 1080), recipe: recipe(ladder), replaced: false)
        #expect(draft.width == 1024)
        #expect(draft.height == 1024)
        #expect(draft.canvasIntent == .manual)
    }

    /// Putting a DIFFERENT picture in re-arms the automatic choice even over
    /// a canvas somebody chose, and re-records the intent to say so --
    /// studio's rule verbatim (`CreatePage.vue:1174-1180`). A programmatic
    /// Reuse is the exception, and pins it manual instead.
    @Test func adifferentPictureReArmsTheCanvasUnlessItWasRestored() {
        var manual = RenderDraft()
        manual.canvasIntent = .manual
        manual.width = 512
        manual.height = 512
        manual.attachSourceShape((1920, 1080), recipe: recipe(ladder), replaced: true)
        #expect(manual.width == 1344)
        #expect(manual.canvasIntent == .source)

        var restored = RenderDraft()
        restored.width = 512
        restored.height = 512
        restored.attachSourceShape((1920, 1080), recipe: recipe(ladder),
                                   replaced: true, preserveReplacement: true)
        #expect(restored.width == 512)
        #expect(restored.canvasIntent == .manual)
    }

    /// `source-exact` is the source's own aligned, capped size -- measured
    /// against the CONTRACT, so a bucket recipe does not snap it to a tier.
    @Test func sourceExactKeepsThePicturesOwnAlignedSize() {
        var draft = RenderDraft()
        draft.canvasIntent = .sourceExact
        draft.attachSourceShape((1000, 600), recipe: recipe(ladder), replaced: true)
        // `source-exact` survives a replacement -- it is a stronger statement
        // than "follow the source" and is not downgraded to it.
        #expect(draft.canvasIntent == .sourceExact)
        #expect(draft.width == 1024)
        #expect(draft.height == 576)
    }

    /// A recipe with no presets has no ladder to land on, so the source's own
    /// size is the only honest answer.
    @Test func aRecipeWithNoPresetsFallsBackToTheSourcesOwnSize() {
        var draft = RenderDraft()
        draft.attachSourceShape((1000, 600), recipe: recipe("null"), replaced: true)
        #expect(draft.width == 1024)
        #expect(draft.height == 576)
    }
}
