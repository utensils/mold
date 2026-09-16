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
        steps: steps, guidance: guidance,
        capabilities: RecipeCapabilities(
            prompt: PromptCapability(mode: prompt, reason: nil), negativePrompt: nil,
            output: nil, referenceImages: nil, supportsStrength: nil, supportsLora: nil,
            supportsIdentity: nil, supportsSequence: nil, supportsExtend: nil,
            supportsAudio: nil)
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
