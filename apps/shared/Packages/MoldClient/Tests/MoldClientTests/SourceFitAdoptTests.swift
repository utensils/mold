import Foundation
import Testing

@testable import MoldClient

/// What a MODEL SWITCH does to the fit policy and to a canvas that was
/// following a source.
struct SourceFitAdoptTests {
    private func recipe(mask: Bool, groups: String = "null") -> GenerationRecipe {
        let maskBlock = mask
            ? #"{"mode": "adjustable", "required": false, "reason": null}"#
            : #"{"mode": "hidden", "required": false, "reason": null}"#
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
         "temporal": null, "capabilities": {"mask": \(maskBlock)},
         "request_selector": null}
        """
        return try! MoldJSON.decoder.decode(GenerationRecipe.self, from: Data(json.utf8))
    }

    private let ladder = """
    [{"id": "square", "label": "1:1",
      "presets": [{"id": "s1", "width": 1024, "height": 1024, "tier": "recommended"}]},
     {"id": "wide", "label": "16:9",
      "presets": [{"id": "w1", "width": 1344, "height": 768, "tier": "recommended"}]}]
    """

    /// **Fails today**: `coercedForMaskless` had no callers, so `pad-repaint`
    /// carried onto a recipe with no mask path -- and the fit then wrote a
    /// white-band mask into a draft for a model that cannot repaint, and
    /// shipped `source_fit: {"mode":"pad-repaint"}` as provenance for it.
    @Test func padRepaintIsCoercedOntoARecipeWithNoMaskPath() {
        var draft = RenderDraft()
        draft.media.sourceImage = "SRC"
        draft.media.sourceFit = .padRepaint

        let maskless = draft.adopting(recipe(mask: false), isNewModel: false)
        #expect(maskless.media.sourceFit == .default)
        // And the request carries the coerced policy, not the original.
        #expect(RenderRequest.one(maskless, model: "m").sourceFit == .default)

        // A mask-capable recipe keeps whatever is set -- the coercion is one
        // way, exactly as `coerceSourceFitForMaskless` is.
        var repaint = draft
        repaint.media.sourceFit = .padRepaint
        #expect(repaint.adopting(recipe(mask: true), isNewModel: false)
            .media.sourceFit == .padRepaint)
    }

    /// The other half of #1166. **Fails today**: `adopting` wrote the new
    /// recipe's default size and never re-consulted `canvasIntent`, so a
    /// canvas that had been following an attached photograph went square on
    /// the next model switch and stayed there.
    @Test func acanvasFollowingASourceKeepsFollowingItAcrossAModelSwitch() {
        var draft = RenderDraft()
        draft.media.sourceImage = "SRC"
        draft.media.sourceImagePixels = SourcePixels(width: 1920, height: 1080)
        draft.canvasIntent = .source

        let adopted = draft.adopting(recipe(mask: true, groups: ladder), isNewModel: true)
        #expect(adopted.width == 1344)
        #expect(adopted.height == 768)
        #expect(adopted.canvasIntent == .source)
    }

    /// A canvas somebody CHOSE is still never moved by a model switch.
    @Test func amanualCanvasIsStillNeverMovedByAModelSwitch() {
        var draft = RenderDraft()
        draft.media.sourceImage = "SRC"
        draft.media.sourceImagePixels = SourcePixels(width: 1920, height: 1080)
        draft.canvasIntent = .manual
        draft.width = 1024
        draft.height = 1024

        let adopted = draft.adopting(recipe(mask: true, groups: ladder), isNewModel: false)
        #expect(adopted.width == 1024)
        #expect(adopted.height == 1024)
    }

    /// With no source attached there is nothing to follow, and the recipe's
    /// own default stands.
    @Test func withNoSourceTheRecipesOwnDefaultStands() {
        var draft = RenderDraft()
        draft.canvasIntent = .source
        let adopted = draft.adopting(recipe(mask: true, groups: ladder), isNewModel: true)
        #expect(adopted.width == 1024)
        #expect(adopted.height == 1024)
    }
}
