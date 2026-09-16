import Foundation
import Testing

@testable import MoldClient

private func recipe(
    steps: IntegerControl, guidance: FloatControl,
    prompt: PromptRequirement = .required, width: Int = 1024
) -> GenerationRecipe {
    GenerationRecipe(
        id: "r", label: "R",
        defaults: GenerationDefaults(width: width, height: width, steps: steps.default,
                                     guidance: guidance.default, frames: nil, fps: nil,
                                     negativePrompt: nil),
        resolution: ResolutionProfile(domain: .dynamic, alignment: 16, minWidth: 256,
                                      minHeight: 256, maxPixels: nil, maxAxisPixels: nil,
                                      offBucket: nil, aspectGroups: nil),
        steps: steps, guidance: guidance, temporal: nil,
        capabilities: RecipeCapabilities(
            prompt: PromptCapability(mode: prompt, reason: nil), negativePrompt: nil,
            output: nil, referenceImages: nil, supportsStrength: nil, supportsLora: nil,
            supportsIdentity: nil, supportsSequence: nil, supportsExtend: nil,
            supportsAudio: nil, sourceImage: nil)
    )
}

private let wide = IntegerControl(default: 20, min: 1, max: 100, step: 1,
                                  recommended: nil, mode: .adjustable, note: nil)
private let narrow = IntegerControl(default: 4, min: 1, max: 8, step: 1,
                                    recommended: nil, mode: .adjustable, note: nil)
private let guidance = FloatControl(default: 3.5, min: 0, max: 10, step: 0.1,
                                    mode: .adjustable, note: nil)

@Test func switchingModelsTakesTheNewRecipesDefaults() {
    var draft = RenderDraft()
    draft.prompt = "a tin robot"
    draft.steps = 50

    let adopted = draft.adopting(recipe(steps: narrow, guidance: guidance), isNewModel: true)
    #expect(adopted.steps == 4)
    // Changing model is not a reason to lose what you typed.
    #expect(adopted.prompt == "a tin robot")
}

@Test func keepingTheSameModelClampsRatherThanResets() {
    var draft = RenderDraft()
    draft.steps = 50
    let adopted = draft.adopting(recipe(steps: narrow, guidance: guidance), isNewModel: false)
    // 50 is out of range for this recipe, so it lands on the ceiling rather
    // than being silently submitted and refused by the host.
    #expect(adopted.steps == 8)
}

@Test func aFixedControlTakesItsOneValueWhateverWasThere() {
    let fixed = IntegerControl(default: 4, min: 4, max: 4, step: 1,
                               recommended: nil, mode: .fixed, note: nil)
    var draft = RenderDraft()
    draft.steps = 37
    #expect(draft.adopting(recipe(steps: fixed, guidance: guidance), isNewModel: false).steps == 4)
}

@Test func aPromptIgnoredRecipeClearsThePrompt() {
    var draft = RenderDraft()
    draft.prompt = "text no encoder will read"
    let adopted = draft.adopting(
        recipe(steps: wide, guidance: guidance, prompt: .ignored), isNewModel: true)
    #expect(adopted.prompt.isEmpty)
}

@Test func anEmptyPromptIsRefusedOnlyWhereThePromptIsRequired() {
    let draft = RenderDraft()
    #expect(draft.refusal(for: recipe(steps: wide, guidance: guidance)) != nil)
    #expect(draft.refusal(for: recipe(steps: wide, guidance: guidance, prompt: .ignored)) == nil)
    #expect(draft.refusal(for: recipe(steps: wide, guidance: guidance, prompt: .optional)) == nil)
}

@Test func anUnlockedSeedIsLeftToTheHost() {
    var draft = RenderDraft()
    draft.seed = 42
    draft.locksSeed = false
    #expect(draft.request(model: "m").seed == nil)
    draft.locksSeed = true
    #expect(draft.request(model: "m").seed == 42)
}

private let wanTemporal = try! MoldJSON.decoder.decode(TemporalProfile.self, from: Data("""
{"frames":{"default":121,"min":1,"max":257,"step":4,"recommended":[121],"mode":"adjustable"},
 "frame_offset":1,"fps":{"mode":"adjustable","default":24,"min":1,"max":120,"step":1}}
""".utf8))

private func clipRecipe(source: SourceImageCapability? = nil,
                        negative: ControlMode = .hidden) -> GenerationRecipe {
    GenerationRecipe(
        id: "v", label: "V",
        defaults: GenerationDefaults(width: 704, height: 480, steps: 4, guidance: 1,
                                     frames: 121, fps: 24, negativePrompt: nil),
        resolution: ResolutionProfile(domain: .dynamic, alignment: 16, minWidth: 64,
                                      minHeight: 64, maxPixels: nil, maxAxisPixels: nil,
                                      offBucket: nil, aspectGroups: nil),
        steps: wide, guidance: guidance, temporal: wanTemporal,
        capabilities: RecipeCapabilities(
            prompt: nil, negativePrompt: FeatureControl(mode: negative, required: false,
                                                        reason: nil),
            output: nil, referenceImages: nil, supportsStrength: true, supportsLora: nil,
            supportsIdentity: nil, supportsSequence: nil, supportsExtend: nil,
            supportsAudio: nil, sourceImage: source))
}

@Test func adoptingAClipRecipeTakesItsLengthAndRate() {
    let draft = RenderDraft().adopting(clipRecipe(), isNewModel: true)
    #expect(draft.frames == 121)
    #expect(draft.fps == 24)
}

@Test func aFrameCountOffTheGridIsSnappedRatherThanSent() {
    var draft = RenderDraft().adopting(clipRecipe(), isNewModel: true)
    draft.frames = 120
    // Wan's grid is 4k+1, so 120 would be refused outright.
    #expect(draft.adopting(clipRecipe(), isNewModel: false).frames == 121)
}

@Test func movingToAStillModelDropsTheClipLength() {
    var draft = RenderDraft().adopting(clipRecipe(), isNewModel: true)
    draft = draft.adopting(recipe(steps: wide, guidance: guidance), isNewModel: true)
    #expect(draft.frames == nil)
    #expect(draft.fps == nil)
    #expect(draft.request(model: "m").frames == nil)
}

@Test func aSourceImageIsDroppedWhenTheRecipeCannotReadOne() {
    var draft = RenderDraft()
    draft.sourceImage = "AAAA"
    draft.sourceImageName = "a.png"

    let keeps = draft.adopting(clipRecipe(source: .optional), isNewModel: false)
    #expect(keeps.sourceImage == "AAAA")

    // Sending bytes to a recipe with no source path is a refusal, not a render.
    let drops = draft.adopting(clipRecipe(source: .unsupported), isNewModel: false)
    #expect(drops.sourceImage == nil)
    #expect(drops.sourceImageName == nil)
}

@Test func strengthRidesOnlyWithSomethingToApplyItTo() {
    var draft = RenderDraft()
    draft.strength = 0.4
    #expect(draft.request(model: "m").strength == nil)

    draft.sourceImage = "AAAA"
    #expect(draft.request(model: "m").strength == 0.4)
}

@Test func aHiddenNegativePromptIsClearedRatherThanSent() {
    var draft = RenderDraft()
    draft.negativePrompt = "blurry"
    #expect(draft.adopting(clipRecipe(negative: .hidden), isNewModel: false)
        .negativePrompt.isEmpty)
    #expect(draft.adopting(clipRecipe(negative: .adjustable), isNewModel: false)
        .negativePrompt == "blurry")
}

private func metadata(_ json: String) -> OutputMetadata {
    try! MoldJSON.decoder.decode(OutputMetadata.self, from: Data(json.utf8))
}

@Test func reuseRestoresWhatWasRendered() {
    let draft = RenderDraft(reusing: metadata("""
    {"prompt":"a tin robot","negative_prompt":"blurry","model":"flux-dev:q8",
     "seed":42,"steps":28,"guidance":3.5,"width":1024,"height":768}
    """))
    #expect(draft.prompt == "a tin robot")
    #expect(draft.negativePrompt == "blurry")
    #expect(draft.steps == 28)
    #expect(draft.width == 1024)
    #expect(draft.height == 768)
}

@Test func reuseRestoresTheSeedWithoutPinningIt() {
    let draft = RenderDraft(reusing: metadata("""
    {"prompt":"p","model":"m","seed":42,"steps":4,"guidance":0,"width":512,"height":512}
    """))
    // "Like that one, but different" is what reuse usually means; pinning the
    // seed would make every reuse produce the identical picture.
    #expect(draft.seed == 42)
    #expect(draft.locksSeed == false)
    #expect(draft.request(model: "m").seed == nil)
}

@Test func reusingASequenceTakesOnlyItsFirstStage() {
    let draft = RenderDraft(reusing: metadata("""
    {"prompt":"a harbour at dawn\\nthe boats leave\\ngulls circle",
     "output_mode":"sequence","model":"m","seed":1,"steps":4,"guidance":0,
     "width":512,"height":512}
    """))
    // A sequence's recorded prompt is every stage joined. Restoring the whole
    // thing would put three prompts in a box that holds one.
    #expect(draft.prompt == "a harbour at dawn")
}

@Test func reuseTakesTheRenderedSizeNotTheUpscaledOne() {
    let draft = RenderDraft(reusing: metadata("""
    {"prompt":"p","model":"m","seed":1,"steps":4,"guidance":0,
     "width":2048,"height":2048,"generation_width":1024,"generation_height":1024}
    """))
    #expect(draft.width == 1024)
    #expect(draft.height == 1024)
}
