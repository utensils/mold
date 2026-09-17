import Foundation
import Testing

@testable import MoldClient

// Conditioning a draft holds for a recipe that cannot currently take it --
// decision 4 in the M4 design: switching to a text-to-video model and back
// used to DESTROY a source image, an identity photo, or an adapter stack
// outright. `RenderDraft+Park.swift` is the rule under test; these fixtures
// build recipes with the one field each test cares about and leave the rest
// at a neutral default.

private let steps = IntegerControl(default: 20, min: 1, max: 100, step: 1,
                                   recommended: nil, mode: .adjustable, note: nil)
private let guidance = FloatControl(default: 3.5, min: 0, max: 10, step: 0.1,
                                    mode: .adjustable, note: nil)

private func capabilities(
    sourceImage: SourceImageCapability? = nil,
    supportsIdentity: Bool? = nil,
    lora: AdapterControl? = nil,
    mask: FeatureControl? = nil,
    referenceImages: ReferenceImagesCapability? = nil,
    controlnet: AdapterControl? = nil
) -> RecipeCapabilities {
    RecipeCapabilities(
        prompt: nil, negativePrompt: nil, output: nil, referenceImages: referenceImages,
        supportsStrength: nil, supportsLora: nil, supportsControlnet: nil,
        supportsIdentity: supportsIdentity, supportsSequence: nil, supportsExtend: nil,
        supportsAudio: nil, sourceImage: sourceImage, lora: lora, controlnet: controlnet,
        mask: mask, keyframes: nil, audio: nil, sourceVideo: nil, schedulers: nil,
        wanRecipe: nil
    )
}

private func recipe(_ capabilities: RecipeCapabilities) -> GenerationRecipe {
    GenerationRecipe(
        id: "r", label: "R",
        defaults: GenerationDefaults(width: 1024, height: 1024, steps: 20, guidance: 3.5,
                                     frames: nil, fps: nil, negativePrompt: nil),
        resolution: ResolutionProfile(domain: .dynamic, alignment: 16, minWidth: 64,
                                      minHeight: 64, maxPixels: nil, maxAxisPixels: nil,
                                      offBucket: nil, aspectGroups: nil),
        steps: steps, guidance: guidance, temporal: nil,
        capabilities: capabilities, requestSelector: nil
    )
}

@Test func aVideoModelParksTheSourceImageAndAStillModelHandsItBack() {
    var draft = RenderDraft()
    draft.sourceImage = "AAAA"
    draft.sourceImageName = "a.png"

    let parked = draft.adopting(recipe(capabilities(sourceImage: .unsupported)), isNewModel: false)
    #expect(parked.sourceImage == nil)
    #expect(parked.sourceImageName == nil)
    #expect(parked.parked.sourceImage == "AAAA")
    #expect(parked.parked.sourceImageName == "a.png")

    // Absence means the recipe READS a source image (fact 1 in the M4
    // design) -- not that there is no source path.
    let restored = parked.adopting(recipe(capabilities(sourceImage: nil)), isNewModel: false)
    #expect(restored.sourceImage == "AAAA")
    #expect(restored.sourceImageName == "a.png")
    #expect(restored.parked.sourceImage == nil)
    #expect(restored.parked.sourceImageName == nil)
}

@Test func aLiveValueBeatsAParkedOne() {
    var draft = RenderDraft()
    draft.parked.sourceImage = "OLD"
    draft.parked.sourceImageName = "old.png"
    draft.sourceImage = "NEW"
    draft.sourceImageName = "new.png"

    // The recipe can read a source image, but one is already sitting there --
    // the parked one is a rescue, not a history, and must not overwrite it.
    let adopted = draft.adopting(recipe(capabilities(sourceImage: nil)), isNewModel: false)
    #expect(adopted.sourceImage == "NEW")
    #expect(adopted.sourceImageName == "new.png")
    #expect(adopted.parked.sourceImage == "OLD")
}

@Test func anUnqualifiedCheckpointParksTheFaceAndRefusesNothing() {
    var draft = RenderDraft()
    let photo = IdentityPhoto(encoded: "AAAA", name: "face.png")
    draft.identity = IdentityConditioning(photos: [photo])

    let adopted = draft.adopting(recipe(capabilities(supportsIdentity: false)), isNewModel: false)
    #expect(adopted.identity == nil)
    #expect(adopted.parked.identity?.photos == [photo])

    // A model that later qualifies gets the face back.
    let restored = adopted.adopting(recipe(capabilities(supportsIdentity: true)), isNewModel: false)
    #expect(restored.identity?.photos == [photo])
    #expect(restored.parked.identity == nil)
}

@Test func aStackLongerThanTheRecipeAllowsParksItsTail() {
    var draft = RenderDraft()
    draft.loras = [
        LoraChoice(path: "/a.safetensors", name: "A"),
        LoraChoice(path: "/b.safetensors", name: "B"),
        LoraChoice(path: "/c.safetensors", name: "C"),
    ]
    let narrow = AdapterControl(mode: .adjustable, maxCount: 2, reason: nil)
    let adopted = draft.adopting(recipe(capabilities(lora: narrow)), isNewModel: false)
    #expect(adopted.loras.map(\.path) == ["/a.safetensors", "/b.safetensors"])
    #expect(adopted.parked.loras.map(\.path) == ["/c.safetensors"])
}

@Test func clearingTheSourceParksTheMaskWithIt() {
    var draft = RenderDraft()
    draft.sourceImage = "AAAA"
    draft.sourceImageName = "a.png"
    draft.maskImage = "MASK"

    let maskCapable = FeatureControl(mode: .adjustable, required: false, reason: nil)
    let videoRecipe = recipe(capabilities(sourceImage: .unsupported, mask: maskCapable))
    let adopted = draft.adopting(videoRecipe, isNewModel: false)
    #expect(adopted.sourceImage == nil)
    // The mask cannot mean anything with no source, even though the recipe's
    // own `mask` block would otherwise allow one.
    #expect(adopted.maskImage == nil)
    #expect(adopted.parked.maskImage == "MASK")
}

@Test func a_recipe_without_controlnet_parks_the_control_and_hands_it_back() {
    var draft = RenderDraft()
    draft.control = ControlConditioning(image: "CTRL", name: "c.png", model: "controlnet-canny-sd15:fp16")

    // SD1.5's own `controlnet` block goes away on a recipe that doesn't
    // advertise one -- `RecipeCapabilities.controlNet` answers nil the same
    // way for a `hidden` block and no block at all (`RecipeCapabilities+Reading.swift`).
    let parked = draft.adopting(recipe(capabilities(controlnet: nil)), isNewModel: false)
    #expect(parked.control == nil)
    #expect(parked.parked.control?.model == "controlnet-canny-sd15:fp16")
    #expect(parked.parked.control?.image == "CTRL")

    // Back on SD1.5, or any recipe that advertises the block again, the
    // control conditioning comes back whole rather than staying lost.
    let adjustable = AdapterControl(mode: .adjustable, maxCount: 1, reason: nil)
    let restored = parked.adopting(recipe(capabilities(controlnet: adjustable)), isNewModel: false)
    #expect(restored.control?.model == "controlnet-canny-sd15:fp16")
    #expect(restored.control?.image == "CTRL")
    #expect(restored.parked.control == nil)
}

@Test func exclusiveReferencesParkTheSourceRatherThanDeletingIt() {
    var draft = RenderDraft()
    draft.sourceImage = "SRC"
    draft.sourceImageName = "s.png"
    draft.editImages = ["A"]

    let exclusive = ReferenceImagesCapability(
        mode: .adjustable, required: false, maxCount: 2, primaryIsTarget: false,
        sourceRelation: .exclusive, reason: nil, weight: nil
    )
    let adopted = draft.adopting(
        recipe(capabilities(sourceImage: .optional, referenceImages: exclusive)), isNewModel: false
    )
    #expect(adopted.editImages == ["A"])
    #expect(adopted.sourceImage == nil)
    #expect(adopted.parked.sourceImage == "SRC")

    // Back on a plain img2img recipe (no references block), the source
    // comes back rather than staying lost.
    let restored = adopted.adopting(recipe(capabilities(sourceImage: .optional)), isNewModel: false)
    #expect(restored.sourceImage == "SRC")
}
