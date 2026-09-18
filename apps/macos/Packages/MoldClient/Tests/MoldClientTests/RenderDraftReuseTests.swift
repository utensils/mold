import Foundation
import Testing

@testable import MoldClient

/// Use These Settings: the whole recipe back in the draft.
///
/// Port of `applyMetadataToForm` (`desktop/src/lib/generateForm.ts:1455-1609`),
/// read against real prints from hal9000 (`Fixtures/provenance-hal9000.json`).
///
/// **Fails today**: `RenderDraft(reusing:)` restores seven scalars, so the
/// title, the filing, the adapters, the sampler, the crop policy, the
/// identity knobs, the LTX-2 controls and the clip switches are all lost.
@MainActor struct RenderDraftReuseTests {

    private func draft(_ filename: String) throws -> RenderDraft {
        RenderDraft(reusing: try Provenance.metadata(filename))
    }

    @Test func restoresTheNumbersItAlwaysDid() throws {
        let clip = try draft("mold-ltx-2.5-22b-distilled-q8-1789532686738.mp4")
        let metadata = try Provenance.metadata("mold-ltx-2.5-22b-distilled-q8-1789532686738.mp4")
        #expect(clip.prompt == metadata.prompt)
        #expect(clip.width == metadata.generationWidth)
        #expect(clip.height == metadata.generationHeight)
        #expect(clip.steps == metadata.steps)
        #expect(clip.seed == metadata.seed)
        // Restored but NOT locked: reuse means "like that one, but different".
        #expect(clip.locksSeed == false)
    }

    @Test func restoresTheFilingAndTheRewriteProvenance() throws {
        let filed = try draft("mold-hunyuan3d-mini-turbo-fp16-1788387693156~nsfw.glb")
        #expect(!filed.title.isEmpty)
        #expect(!filed.tags.isEmpty)
        #expect(filed.collectionName?.isEmpty == false)
        let rewritten = try draft("mold-jibmix-flux-fp8-1788207348881~nsfw.png")
        #expect(rewritten.originalPrompt?.isEmpty == false)
        #expect(rewritten.promptTransform?.operation == .remix)
    }

    @Test func restoresTheAdapterStackWithAReadableName() throws {
        let stack = try draft("mold-ltx-2-19b-distilled-fp8-1786851614872.mp4").media.loras
        #expect(stack.count == 1)
        let adapter = try #require(stack.first)
        #expect(adapter.path.hasSuffix(".safetensors"))
        #expect(adapter.scale == 0.5)
        // The wire carries no name; the draft's rows need one.
        #expect(adapter.name == "ltx-2-19b-lora-camera-control-dolly-in")
    }

    @Test func restoresALegacySingularAdapterAsAStackOfOne() {
        let legacy = RenderDraft(reusing: Synthetic.metadata("""
        "lora":"/models/looks/painterly.safetensors","lora_scale":0.8
        """))
        #expect(legacy.media.loras.map(\.path) == ["/models/looks/painterly.safetensors"])
        #expect(legacy.media.loras.first?.scale == 0.8)
    }

    @Test func restoresTheSamplerControlsIntoTheirLiveSlots() {
        let sampled = RenderDraft(reusing: Synthetic.metadata("""
        "scheduler":"dpm-pp","cfg_plus":true,"sample_shift":5.5,
        "distill_strength_high":0.9,"distill_strength_low":0.4,
        "guidance_overrides":{"stg_scale":1.5,"stg_blocks":[3,7],"skip_step":2}
        """))
        #expect(sampled.advanced.scheduler == "dpm-pp")
        #expect(sampled.advanced.cfgPlus)
        #expect(sampled.advanced.sampleShift == 5.5)
        #expect(sampled.advanced.distillStrengthHigh == 0.9)
        #expect(sampled.advanced.distillStrengthLow == 0.4)
        #expect(sampled.advanced.stgScale == 1.5)
        // The block list is free text on the draft, parsed at request time.
        #expect(sampled.advanced.stgBlocks == "3, 7")
        #expect(sampled.advanced.skipStep == 2)
    }

    @Test func restoresTheControlNetPairAndTheUpscaler() {
        let refined = RenderDraft(reusing: Synthetic.metadata("""
        "control_model":"controlnet-canny-sd15:fp16","control_scale":0.7,
        "upscale_model":"real-esrgan-x4plus:fp16"
        """))
        #expect(refined.media.control?.model == "controlnet-canny-sd15:fp16")
        #expect(refined.media.control?.scale == 0.7)
        // The picture itself is bytes, and metadata never carries bytes.
        #expect(refined.media.control?.image == nil)
        #expect(refined.upscaleModel == "real-esrgan-x4plus:fp16")
    }

    @Test func restoresTheIdentityKnobsAsAReattachWaitingForItsFace() throws {
        let face = try draft("mold-jibmix-flux-fp8-1788383514251~nsfw.png")
        let identity = try #require(face.media.identity)
        #expect(identity.weight > 0)
        // No photograph: metadata records the digest, never the face. The
        // request builder ships nothing at all from an empty set, so the
        // knobs can never reach the wire without the person they describe.
        #expect(identity.photos.isEmpty)
    }

    @Test func restoresTheClipControlsAndTheCropPolicy() throws {
        let clip = try draft("mold-ltx-2.5-22b-distilled-q8-1789532686738.mp4")
        #expect(clip.frames != nil)
        #expect(clip.fps != nil)
        #expect(clip.enableAudio)
        #expect(clip.strength == 0.75)
        #expect(clip.media.sourceFit == .cropFill(alignX: nil, alignY: nil))
        // A print that recorded the alignment too keeps it.
        #expect(try draft("mold-ltx-2-19b-distilled-fp8-1786851614872.mp4").media.sourceFit
            == .cropFill(alignX: .center, alignY: .center))
        // This print RAN `distilled` without naming it, so reuse pins nothing.
        #expect(clip.pipeline == nil)
    }

    @Test func aRecordedFitIsRestoredEvenWhereItIsNotTheDefaultOne() {
        let padded = RenderDraft(reusing: Synthetic.metadata("""
        "source_fit":{"mode":"pad-fit"}
        """))
        #expect(padded.media.sourceFit == .padFit)
        // A print made before the policy was recorded keeps the app's own
        // default rather than an invented one.
        #expect(RenderDraft(reusing: Synthetic.metadata("\"steps\":4")).media.sourceFit
            == .default)
    }

    @Test func doesNotPromoteAResolvedPipelineIntoAnOverride() {
        // `pipeline` records what RAN; `pipeline_requested` says whether the
        // author named it. Restoring the former on a print that named nothing
        // pins a choice nobody made.
        let resolved = RenderDraft(reusing: Synthetic.metadata("""
        "pipeline":"two-stage","pipeline_requested":false
        """))
        #expect(resolved.pipeline == nil)
        let chosen = RenderDraft(reusing: Synthetic.metadata("""
        "pipeline":"two-stage","pipeline_requested":true
        """))
        #expect(chosen.pipeline == "two-stage")
    }

    @Test func aSequencePrintRestoresItsFirstStageAndNothingElseOfTheWall() throws {
        let sequence = try draft(
            "mold-chain-8c352a5a9ac5d5c23549e66d96f07c97f26331a7798de3fce244cdc4da754073-take-1.mp4")
        #expect(!sequence.prompt.contains("\n"))
        #expect(sequence.prompt.hasPrefix("a slow cinematic dolly"))
    }

    @Test func everyByteBearingWellIsClearedRatherThanPairedWithAnotherPrint() throws {
        var held = RenderDraft()
        held.media.sourceImage = "AAAA"
        held.media.sourceImageName = "someone-elses.png"
        held.media.sourceImageOriginal = "AAAA"
        held.media.maskImage = "BBBB"
        held.media.editImages = ["CCCC"]
        held.media.identity = IdentityConditioning(photos: [
            IdentityPhoto(encoded: "DDDD", name: "a-face.png")])
        held.media.control = ControlConditioning(image: "EEEE", model: "some-controlnet")
        held.media.keyframes = [KeyframeCondition(frame: 0, image: "FFFF")]
        held.media.extendVideo = "GGGG"
        held.media.audioFile = "HHHH"
        held.media.sourceVideo = "IIII"

        // Reuse builds a FRESH draft, so nothing staged can survive it. This
        // is the assertion that would fail if the initializer ever became a
        // mutation of what was already on screen.
        let restored = try draft("mold-qwen-image-q8-1789529980561.png")
        #expect(restored.media.sourceImage == nil)
        #expect(restored.media.sourceImageOriginal == nil)
        #expect(restored.media.maskImage == nil)
        #expect(restored.media.editImages.isEmpty)
        #expect(restored.media.identity == nil)
        #expect(restored.media.control == nil)
        #expect(restored.media.keyframes.isEmpty)
        #expect(restored.media.extendVideo == nil)
        #expect(restored.media.audioFile == nil)
        #expect(restored.media.sourceVideo == nil)
        #expect(held.media.sourceImage == "AAAA")  // untouched, not mutated
    }

    /// The rule the whole restore rests on: reuse writes into the LIVE slots
    /// and the model adoption that follows decides what this recipe can take.
    /// A control restored onto a recipe that does not advertise it is PARKED,
    /// so a print made on one machine's wan tier and reused onto an SD recipe
    /// hands its flow shift back the moment a wan recipe returns -- rather
    /// than being dropped here, where nothing would ever give it back.
    @Test func aControlTheTargetRecipeCannotTakeIsParkedRatherThanLost() {
        let restored = RenderDraft(reusing: Synthetic.metadata("""
        "sample_shift":5.5,"scheduler":"dpm-pp","cfg_plus":true
        """))
        #expect(restored.advanced.sampleShift == 5.5)

        let plain = Fixture.recipe(schedulers: "null", wanRecipe: "null")
        let adopted = restored.adopting(plain, isNewModel: false)
        #expect(adopted.advanced.sampleShift == nil)
        #expect(adopted.advanced.parked.sampleShift == 5.5)
        #expect(adopted.advanced.scheduler == nil)
        #expect(adopted.advanced.parked.scheduler == "dpm-pp")

        let wan = Fixture.recipe(schedulers: #"["dpm-pp","uni-pc"]"#, wanRecipe: """
        {"mode": "adjustable", "supports_distill_strength": true,
         "supports_first_last_frame": false, "first_last_frame_min_frames": null,
         "reason": null}
        """)
        let back = adopted.adopting(wan, isNewModel: false)
        #expect(back.advanced.sampleShift == 5.5)
        #expect(back.advanced.scheduler == "dpm-pp")
    }

    @Test func aRestoredCanvasIsManualSoNothingReResolvesItUnderneath() throws {
        // The recorded size IS the answer. A canvas left "following a source"
        // would be re-derived the moment the retained source is re-attached,
        // and the print would come back a different shape.
        #expect(try draft("mold-ltx-2.5-22b-distilled-q8-1789532686738.mp4")
            .canvasIntent == .manual)
    }
}

/// A recipe with the one capability block a case is about.
enum Fixture {
    static func recipe(schedulers: String, wanRecipe: String) -> GenerationRecipe {
        let json = """
        {"id": "r", "label": "R",
         "defaults": {"width": 512, "height": 512, "steps": 20, "guidance": 7.0,
                      "frames": null, "fps": null, "negative_prompt": null},
         "resolution": {"domain": "dynamic", "alignment": 8, "min_width": 64,
                        "min_height": 64, "max_pixels": null, "max_axis_pixels": null,
                        "off_bucket": null, "aspect_groups": null},
         "steps": {"default": 20, "min": 1, "max": 100, "step": 1, "recommended": null,
                   "mode": "adjustable", "note": null},
         "guidance": {"default": 7.0, "min": 0, "max": 20, "step": 0.1,
                      "mode": "adjustable", "note": null},
         "temporal": null,
         "capabilities": {"schedulers": \(schedulers), "wan_recipe": \(wanRecipe)},
         "request_selector": {"pipeline": null}}
        """
        return try! MoldJSON.decoder.decode(GenerationRecipe.self, from: Data(json.utf8))
    }
}

/// A print with exactly the fields a case is about, and nothing else.
/// Real prints cover the fields hal9000 happens to hold; these cover the ones
/// nothing on it has ever produced. Labelled SYNTHETIC on purpose.
enum Synthetic {
    static func metadata(_ extraFields: String) -> OutputMetadata {
        let row = """
        {"prompt":"p","model":"m","seed":1,"steps":4,"guidance":1.0,
         "width":8,"height":8,"version":"0.29.0",\(extraFields)}
        """
        // A fixture this test wrote itself: a decode failure here is the
        // test's own bug, and there is nothing a caller could do with it.
        return try! MoldJSON.decoder.decode(OutputMetadata.self, from: Data(row.utf8))
    }
}
