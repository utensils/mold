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
            supportsControlnet: nil, supportsIdentity: nil, supportsSequence: nil,
            supportsExtend: nil, supportsAudio: nil, sourceImage: nil, lora: nil,
            controlnet: nil, mask: nil, keyframes: nil, audio: nil, sourceVideo: nil,
            schedulers: nil, wanRecipe: nil),
        requestSelector: nil
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
            supportsControlnet: nil, supportsIdentity: nil, supportsSequence: nil,
            supportsExtend: nil, supportsAudio: nil, sourceImage: source, lora: nil,
            controlnet: nil, mask: nil, keyframes: nil, audio: nil, sourceVideo: nil,
            schedulers: nil, wanRecipe: nil),
        requestSelector: nil)
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
    draft.media.sourceImage = "AAAA"
    draft.media.sourceImageName = "a.png"

    let keeps = draft.adopting(clipRecipe(source: .optional), isNewModel: false)
    #expect(keeps.media.sourceImage == "AAAA")

    // Sending bytes to a recipe with no source path is a refusal, not a render.
    let drops = draft.adopting(clipRecipe(source: .unsupported), isNewModel: false)
    #expect(drops.media.sourceImage == nil)
    #expect(drops.media.sourceImageName == nil)
}

@Test func strengthRidesOnlyWithSomethingToApplyItTo() {
    var draft = RenderDraft()
    draft.strength = 0.4
    #expect(draft.request(model: "m").strength == nil)

    draft.media.sourceImage = "AAAA"
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

private func referenceRecipe(_ mode: ControlMode, relation: ReferenceSourceRelation,
                             maxCount: Int? = 2,
                             source: SourceImageCapability? = .optional) -> GenerationRecipe {
    GenerationRecipe(
        id: "r", label: "R",
        defaults: GenerationDefaults(width: 1024, height: 1024, steps: 20, guidance: 3.5,
                                     frames: nil, fps: nil, negativePrompt: nil),
        resolution: ResolutionProfile(domain: .dynamic, alignment: 16, minWidth: 64,
                                      minHeight: 64, maxPixels: nil, maxAxisPixels: nil,
                                      offBucket: nil, aspectGroups: nil),
        steps: wide, guidance: guidance, temporal: nil,
        capabilities: RecipeCapabilities(
            prompt: nil, negativePrompt: nil, output: nil,
            referenceImages: ReferenceImagesCapability(
                mode: mode, required: false, maxCount: maxCount, primaryIsTarget: false,
                sourceRelation: relation, reason: nil, weight: nil),
            supportsStrength: true, supportsLora: nil, supportsControlnet: nil,
            supportsIdentity: nil, supportsSequence: nil, supportsExtend: nil,
            supportsAudio: nil, sourceImage: source, lora: nil, controlnet: nil, mask: nil,
            keyframes: nil, audio: nil, sourceVideo: nil, schedulers: nil, wanRecipe: nil),
        requestSelector: nil)
}

@Test func referencesAreDroppedWhereTheRecipeHidesThem() {
    var draft = RenderDraft()
    draft.media.editImages = ["A", "B"]
    let adopted = draft.adopting(referenceRecipe(.hidden, relation: .replaces),
                                 isNewModel: false)
    #expect(adopted.media.editImages.isEmpty)
    #expect(adopted.request(model: "m").editImages == nil)
}

@Test func referencesAreTrimmedToWhatTheRecipeAccepts() {
    var draft = RenderDraft()
    draft.media.editImages = ["A", "B", "C", "D"]
    let adopted = draft.adopting(referenceRecipe(.adjustable, relation: .replaces, maxCount: 2),
                                 isNewModel: false)
    #expect(adopted.media.editImages == ["A", "B"])
}

@Test func anExclusiveRecipeCarriesReferencesOrASourceButNotBoth() {
    var draft = RenderDraft()
    draft.media.sourceImage = "SRC"
    draft.media.sourceImageName = "s.png"
    draft.media.editImages = ["A"]
    let adopted = draft.adopting(referenceRecipe(.adjustable, relation: .exclusive),
                                 isNewModel: false)
    // Both wells keep their media; ONE render carries one or the other, and
    // the REQUEST is where that is decided (finding 02#1).
    #expect(adopted.media.editImages == ["A"])
    #expect(adopted.media.sourceImage == "SRC")
    let request = adopted.request(model: "m")
    #expect(!(request.sourceImage != nil && request.editImages != nil))
}

@Test func aReplacesRecipeDropsTheSourceEntirely() {
    var draft = RenderDraft()
    draft.media.sourceImage = "SRC"
    draft.media.editImages = ["A"]
    let adopted = draft.adopting(referenceRecipe(.adjustable, relation: .replaces),
                                 isNewModel: false)
    #expect(adopted.media.sourceImage == nil)
}

@Test func anEmptyReferenceListIsOmittedRatherThanSentEmpty() {
    let draft = RenderDraft().adopting(referenceRecipe(.adjustable, relation: .replaces),
                                       isNewModel: true)
    // An empty array and an absent field are different instructions.
    #expect(draft.request(model: "m").editImages == nil)
}

private func sizedRecipe(_ resolution: ResolutionProfile) -> GenerationRecipe {
    GenerationRecipe(
        id: "r", label: "R",
        defaults: GenerationDefaults(width: 1024, height: 1024, steps: wide.default,
                                     guidance: guidance.default, frames: nil, fps: nil,
                                     negativePrompt: nil),
        resolution: resolution,
        steps: wide, guidance: guidance, temporal: nil,
        capabilities: RecipeCapabilities(
            prompt: nil, negativePrompt: nil, output: nil, referenceImages: nil,
            supportsStrength: nil, supportsLora: nil, supportsControlnet: nil,
            supportsIdentity: nil, supportsSequence: nil, supportsExtend: nil,
            supportsAudio: nil, sourceImage: nil, lora: nil, controlnet: nil, mask: nil,
            keyframes: nil, audio: nil, sourceVideo: nil, schedulers: nil, wanRecipe: nil),
        requestSelector: nil
    )
}

/// A draft kept across a model switch is the one path that carries a size
/// nobody validated against the NEW recipe -- a fresh model already takes
/// its defaults, which are valid by construction. See `RenderDraft.fit`.
@Test func aKeptDraftSnapsToTheNearestBucketOnABucketedRecipe() {
    let buckets = ResolutionProfile(
        domain: .buckets, alignment: nil, minWidth: nil, minHeight: nil,
        maxPixels: nil, maxAxisPixels: nil, offBucket: .reject,
        aspectGroups: [AspectGroup(id: "square", label: "Square", presets: [
            SizePreset(id: "a", width: 832, height: 832, tier: nil),
            SizePreset(id: "b", width: 1216, height: 1216, tier: nil),
        ])])
    var draft = RenderDraft()
    draft.width = 1024
    draft.height = 1024
    // 1024 sits closer to 832 than to 1216.
    let adopted = draft.adopting(sizedRecipe(buckets), isNewModel: false)
    #expect(adopted.width == 832)
    #expect(adopted.height == 832)
}

/// The wan resolution block exactly as plato advertises it
/// (`recipe-wan.json`, `wan22-t2v-a14b:q8`): a BUCKET domain with
/// `off_bucket: warn`, and a real alignment, minimum and pixel budget beside
/// it. Every one of those is non-`Option` on the Rust side and therefore
/// always on the wire -- which is why a test that switched them all off could
/// not have caught what 01#7's first fix broke.
private func wanResolution() throws -> ResolutionProfile {
    let set = try MoldJSON.decoder.decode(
        GenerationProfileSet.self, from: RepoFixtures.fixture("recipe-wan.json"))
    return try #require(set.recipe(named: "default")).resolution
}

/// **Fails today**: `fit` snaps EVERY `.buckets` recipe, so a wan clip
/// rendered at an off-ladder size -- which its host admits with a warning --
/// came back as a different shape on reuse, silently (finding 01#7).
@Test func aWarnedOffBucketSizeIsKeptRatherThanSnapped() throws {
    let wan = try wanResolution()
    #expect(wan.domain == .buckets)
    #expect(wan.offBucket == .warn)

    var draft = RenderDraft()
    draft.width = 1024
    draft.height = 768
    let adopted = draft.adopting(sizedRecipe(wan), isNewModel: false)
    // Already legal for this profile -- on the grid, over the minimum, under
    // the budget -- so it survives, off-ladder though it is.
    #expect(adopted.width == 1024)
    #expect(adopted.height == 768)
}

/// **Fails today**: the `warn` guard returned from the WHOLE `.buckets` arm,
/// so nothing else was applied either. `warn` switches off the bucket
/// MEMBERSHIP check alone -- `validate_resolution`
/// (`generation_profile.rs:1327-1404`) still enforces the grid, the minimums
/// and `max_pixels` for a warned profile, so an off-grid carried size became a
/// hard 422 where the old snap at least rendered.
@Test func aWarnedProfileStillHonoursItsGridAndItsBudget() throws {
    let wan = try wanResolution()
    let alignment = try #require(wan.alignment)
    let maxPixels = try #require(wan.maxPixels)

    // Off the grid (1368 % 16 == 8), carried off a recipe with a finer one.
    var offGrid = RenderDraft()
    offGrid.width = 1368
    offGrid.height = 768
    let aligned = offGrid.adopting(sizedRecipe(wan), isNewModel: false)
    #expect(aligned.width % alignment == 0)
    #expect(aligned.height % alignment == 0)

    // Over the budget, carried off a recipe with a larger one.
    var oversize = RenderDraft()
    oversize.width = 2048
    oversize.height = 1152
    let clamped = oversize.adopting(sizedRecipe(wan), isNewModel: false)
    #expect(clamped.width * clamped.height <= maxPixels)
    #expect(clamped.width % alignment == 0)
    #expect(clamped.height % alignment == 0)
    #expect(clamped.width >= (wan.minWidth ?? 0))
    #expect(clamped.height >= (wan.minHeight ?? 0))
}

/// `wan22-ti2v-5b` is a 32-grid checkpoint: same wire shape, coarser grid.
/// 1360 is a multiple of 16 but not of 32 -- the exact size the regression
/// shipped to a host that refuses it.
@Test func aCoarserWarnedGridStillTakesACarriedSize() {
    let coarse = ResolutionProfile(
        domain: .buckets, alignment: 32, minWidth: 64, minHeight: 64,
        maxPixels: 1_800_000, maxAxisPixels: nil, offBucket: .warn,
        aspectGroups: [AspectGroup(id: "wide", label: "Wide", presets: [
            SizePreset(id: "a", width: 1280, height: 720, tier: nil),
        ])])
    var draft = RenderDraft()
    draft.width = 1360
    draft.height = 768
    let adopted = draft.adopting(sizedRecipe(coarse), isNewModel: false)

    #expect(adopted.width % 32 == 0)
    #expect(adopted.height % 32 == 0)
    // Still off the ladder: `warn` keeps the shape, it does not snap it.
    #expect(adopted.width != 1280)
}

/// Aspect is the one bound that cannot be clamped without changing the shape
/// the size is FOR, so a size outside the band falls back to the ladder --
/// the only legal answer left.
@Test func aWarnedSizeOutsideTheAspectBandFallsBackToTheLadder() {
    let banded = ResolutionProfile(
        domain: .buckets, alignment: 16, minWidth: 64, minHeight: 64,
        maxPixels: 1_800_000, maxAxisPixels: nil,
        minAspectRatio: 0.5, maxAspectRatio: 2.0, offBucket: .warn,
        aspectGroups: [AspectGroup(id: "wide", label: "Wide", presets: [
            SizePreset(id: "a", width: 1280, height: 720, tier: nil),
        ])])
    var draft = RenderDraft()
    draft.width = 1600
    draft.height = 400
    let adopted = draft.adopting(sizedRecipe(banded), isNewModel: false)
    #expect(adopted.width == 1280)
    #expect(adopted.height == 720)
}

/// **Fails today**: alignment rounds to the NEAREST multiple AFTER the pixel
/// budget, which grows both axes back past it -- FLUX's 1,800,000 budget
/// scales 2048x1152 to 1788x1006 and then aligns it to 1792x1008 =
/// 1,806,336, which `validate_resolution` refuses (finding 01#8).
@Test func alignmentNeverGrowsASizeBackPastThePixelBudget() {
    let flux = ResolutionProfile(
        domain: .dynamic, alignment: 16, minWidth: 256, minHeight: 256,
        maxPixels: 1_800_000, maxAxisPixels: nil, offBucket: nil, aspectGroups: nil)
    var draft = RenderDraft()
    draft.width = 2048
    draft.height = 1152
    let adopted = draft.adopting(sizedRecipe(flux), isNewModel: false)

    #expect(adopted.width * adopted.height <= 1_800_000)
    #expect(adopted.width % 16 == 0)
    #expect(adopted.height % 16 == 0)
    // Under budget, so the ordinary nearest-rounding still applies.
    var small = RenderDraft()
    small.width = 1020
    small.height = 1020
    let fitted = small.adopting(sizedRecipe(flux), isNewModel: false)
    #expect(fitted.width == 1024)
    #expect(fitted.height == 1024)
}

@Test func aKeptDraftClampsToTheRangeOnADynamicRecipe() {
    let ranged = ResolutionProfile(
        domain: .dynamic, alignment: 16, minWidth: nil, minHeight: nil,
        maxPixels: nil, maxAxisPixels: 768, offBucket: nil, aspectGroups: nil)
    var draft = RenderDraft()
    draft.width = 1024
    draft.height = 1024
    let adopted = draft.adopting(sizedRecipe(ranged), isNewModel: false)
    #expect(adopted.width == 768)
    #expect(adopted.height == 768)
}

// MARK: - Fan-out

/// `/api/generation-batches` refuses any child whose `batch_size` is not 1
/// (`queue_media_admission.rs:380-386`), so a batch of N is N independent
/// one-output requests sharing a prompt, a title, tags, a collection and one
/// logical `batchId` -- differing only by seed.
@Test func aBatchOfFourIsFourSingleOutputRequests() throws {
    var draft = RenderDraft()
    draft.prompt = "a tin robot"
    draft.title = "Robots"
    draft.tags = ["metal"]
    draft.collectionName = "Robots"
    let requests = draft.requests(model: "m", copies: 4, randomBase: 100)

    #expect(requests.count == 4)
    #expect(requests.allSatisfy { $0.batchSize == 1 })
    #expect(requests.allSatisfy { $0.prompt == "a tin robot" })
    #expect(requests.allSatisfy { $0.title == "Robots" })
    // "Robots" auto-tags as "robots" -- `autoTagTitle` defaults to on, and
    // `ClientTags.compose` is what turns the title into that extra tag.
    #expect(requests.allSatisfy { $0.tags == ["metal", "robots"] })
    #expect(requests.allSatisfy { $0.collection == .named("Robots") })

    let batchId = try #require(requests[0].batchId)
    #expect(requests.allSatisfy { $0.batchId == batchId })
    #expect(requests.map(\.batchIndex) == [1, 2, 3, 4])
    #expect(requests.allSatisfy { $0.batchCount == 4 })
    #expect(requests.map(\.seed) == [100, 101, 102, 103])
}

/// A one-off is not a prepared set: no batch provenance, and with the seed
/// unlocked, no seed either -- the host picks.
@Test func aSingleRenderCarriesNoBatchProvenance() {
    let draft = RenderDraft()
    let requests = draft.requests(model: "m", copies: 1, randomBase: 100)
    #expect(requests.count == 1)
    #expect(requests[0].batchSize == 1)
    #expect(requests[0].batchId == nil)
    #expect(requests[0].batchIndex == nil)
    #expect(requests[0].batchCount == nil)
    #expect(requests[0].seed == nil)
}

/// A seed near the ceiling is legal; a trap is not an answer.
@Test func aLockedSeedAtTheCeilingWrapsRatherThanTrapping() {
    var draft = RenderDraft()
    draft.seed = .max
    draft.locksSeed = true
    let requests = draft.requests(model: "m", copies: 2, randomBase: 999)
    #expect(requests.map(\.seed) == [UInt64.max, 0])
}

/// The count goes in `PlacementRequest.copies`; the request itself always
/// previews one output, however large the draft's own batch is.
@Test func aPlacementRequestIsAlwaysOneOutput() {
    var draft = RenderDraft()
    draft.batchSize = 4
    #expect(draft.placementRequest(model: "m").batchSize == 1)
}

/// Whitespace is not a title.
@Test func aTitleOfOnlyWhitespaceIsNoTitle() {
    var draft = RenderDraft()
    draft.title = "   "
    #expect(draft.request(model: "m").title == nil)
}

/// A filed request names its collection by NAME, never by id -- an id is
/// only ever right on one machine.
@Test func aFiledRequestNamesItsCollectionNeverAnId() {
    var draft = RenderDraft()
    draft.collectionName = "Smurf Village"
    #expect(draft.request(model: "m").collection == .named("Smurf Village"))
    #expect(RenderDraft().request(model: "m").collection == nil)
}

// MARK: - S6c: the draft's media inputs are one value

/// A fresh draft starts with an EMPTY `DraftMedia`, and assigning one
/// wholesale round-trips -- the seam `RenderDraft.media` exists to be a
/// single value, not a bag of independently-defaulted fields.
@Test func theDraftsMediaIsOneValue() {
    #expect(RenderDraft().media == DraftMedia())

    var media = DraftMedia()
    media.sourceImage = "AAAA"
    media.sourceImageName = "a.png"
    media.editImages = ["A", "B"]
    media.identity = IdentityConditioning(photos: [IdentityPhoto(encoded: "x", name: "a")])
    media.loras = [LoraChoice(path: "/x.safetensors", name: "X")]
    media.parked.maskImage = "MASK"

    var draft = RenderDraft()
    draft.media = media
    #expect(draft.media == media)
    #expect(draft.media.sourceImage == "AAAA")
    #expect(draft.media.parked.maskImage == "MASK")
}
