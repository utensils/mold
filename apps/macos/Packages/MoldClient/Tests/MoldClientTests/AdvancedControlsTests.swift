import Foundation
import Testing

@testable import MoldClient

/// The sampler controls: what a recipe offers, what a switch parks, and what
/// reaches the wire. **Fails today**: none of these controls existed.
struct AdvancedControlsTests {
    private func recipe(
        id: String = "r", schedulers: String = "null", wanRecipe: String = "null",
        pipeline: String = "null"
    ) -> GenerationRecipe {
        let json = """
        {"id": "\(id)", "label": "R",
         "defaults": {"width": 512, "height": 512, "steps": 20, "guidance": 7.0,
                      "frames": null, "fps": null, "negative_prompt": null},
         "resolution": {"domain": "dynamic", "alignment": 8, "min_width": 64, "min_height": 64,
                        "max_pixels": null, "max_axis_pixels": null, "off_bucket": null,
                        "aspect_groups": null},
         "steps": {"default": 20, "min": 1, "max": 100, "step": 1, "recommended": null,
                   "mode": "adjustable", "note": null},
         "guidance": {"default": 7.0, "min": 0, "max": 20, "step": 0.1, "mode": "adjustable",
                      "note": null},
         "temporal": null,
         "capabilities": {"schedulers": \(schedulers), "wan_recipe": \(wanRecipe)},
         "request_selector": {"pipeline": \(pipeline)}}
        """
        return try! MoldJSON.decoder.decode(GenerationRecipe.self, from: Data(json.utf8))
    }

    private func profile(_ recipes: [GenerationRecipe]) -> GenerationProfileSet {
        GenerationProfileSet(schemaVersion: 1, profileId: "p", profileHash: "h",
                             defaultRecipeId: recipes[0].id, recipes: recipes)
    }

    private let wanBlock = """
    {"mode": "adjustable", "supports_distill_strength": true,
     "supports_first_last_frame": false, "first_last_frame_min_frames": null, "reason": null}
    """

    /// The server omits `schedulers` when the list is empty, so an absent key
    /// on a recipe IN HAND is a definitive "this tier pins its sampler" --
    /// not an older host, and not a reason to fall back to a family guess.
    @Test func aRecipeThatAdvertisesNoSolversOffersNoPicker() {
        let offered = AdvancedControlsOffered.resolve(
            recipe: recipe(), in: nil, family: "wan")
        #expect(offered.schedulers.isEmpty)
        #expect(offered.sampleShift == false)
        #expect(offered.guidance == false)
        #expect(offered.offersAnything == false)
    }

    @Test func aWanRecipeOffersItsSolversAndItsRecipeControls() {
        let offered = AdvancedControlsOffered.resolve(
            recipe: recipe(schedulers: #"["uni-pc", "euler", "dpm-pp"]"#, wanRecipe: wanBlock),
            in: nil, family: "wan")
        #expect(offered.schedulers == ["uni-pc", "euler", "dpm-pp"])
        #expect(offered.sampleShift)
        #expect(offered.distillStrength)
        // Neither of the two that are not wan's.
        #expect(offered.cfgPlus == false)
        #expect(offered.guidance == false)
    }

    /// LTX-2 is the family whose recipes are CHOSEN by pipeline, which is what
    /// `require_ltx2_family` gates `guidance_overrides` on -- asked of the
    /// profile's shape rather than of a family name.
    @Test func guidanceOverridesFollowTheProfilesPipelinesNotAFamilyName() {
        let auto = recipe(id: "auto")
        let t2a = recipe(id: "t2a", pipeline: "\"t2a\"")
        let ltx2 = profile([auto, t2a])
        #expect(AdvancedControlsOffered.resolve(recipe: auto, in: ltx2, family: "ltx2").guidance)
        // An audio-only pipeline has no video modality to guide against.
        let audioOnly = AdvancedControlsOffered.resolve(recipe: t2a, in: ltx2, family: "ltx2")
        #expect(audioOnly.guidance)
        #expect(audioOnly.modalityScale == false)
        #expect(AdvancedControlsOffered.resolve(recipe: auto, in: ltx2, family: "ltx2")
            .modalityScale)
    }

    @Test func cfgPlusIsOfferedForTheTwoFamiliesThatHaveIt() {
        for family in ["sd3", "sd3.5", " SD3.5 "] {
            #expect(AdvancedControlsOffered.resolve(recipe: recipe(), in: nil, family: family)
                .cfgPlus, "\(family)")
        }
        #expect(AdvancedControlsOffered.resolve(recipe: recipe(), in: nil, family: "sdxl")
            .cfgPlus == false)
    }
}
