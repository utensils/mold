import Foundation

// Turning a draft into a descriptor and back. Split from the shape purely for
// size.
//
// The one rule worth stating twice: NOTHING about media crosses either way.
// `DraftMedia` is untouched by `apply(to:)`, so a restored draft keeps
// whatever the live session already staged rather than clearing it, and a
// persisted descriptor can never resurrect a picture whose bytes are gone.
public extension DraftDescriptor {
    init(_ draft: RenderDraft, model: String?, family: String?, recipeID: String?) {
        self.model = model
        self.family = family
        self.recipeID = recipeID
        prompt = draft.prompt
        negativePrompt = draft.negativePrompt
        width = draft.width
        height = draft.height
        steps = draft.steps
        guidance = draft.guidance
        batchSize = draft.batchSize
        seed = draft.seed
        locksSeed = draft.locksSeed
        frames = draft.frames
        fps = draft.fps
        pipeline = draft.pipeline
        enableAudio = draft.enableAudio
        preferredAudio = draft.preferredAudio
        hasAudioPreference = draft.preferredAudio != nil
        videoOnly = draft.videoOnly
        strength = draft.strength
        title = draft.title
        tags = draft.tags
        collectionName = draft.collectionName
        autoTagTitle = draft.autoTagTitle
        outputFormat = draft.outputFormat
        upscaleModel = draft.upscaleModel
        savesToGallery = draft.savesToGallery
        canvasIntent = draft.canvasIntent
        sourceFit = draft.media.sourceFit
        scheduler = draft.advanced.scheduler
        cfgPlus = draft.advanced.cfgPlus
        sampleShift = draft.advanced.sampleShift
        distillStrengthHigh = draft.advanced.distillStrengthHigh
        distillStrengthLow = draft.advanced.distillStrengthLow
        stgScale = draft.advanced.stgScale
        stgBlocks = draft.advanced.stgBlocks
        rescaleScale = draft.advanced.rescaleScale
        modalityScale = draft.advanced.modalityScale
        skipStep = draft.advanced.skipStep
    }

    /// Writes this descriptor over a draft, leaving its MEDIA alone.
    ///
    /// The recipe is adopted afterwards by the caller, exactly as a model
    /// choice is: this restores what was asked for, and the recipe decides
    /// what of it is still askable.
    func apply(to draft: inout RenderDraft) {
        draft.prompt = prompt
        draft.negativePrompt = negativePrompt
        draft.width = width
        draft.height = height
        draft.steps = steps
        draft.guidance = guidance
        draft.batchSize = batchSize
        draft.seed = seed
        draft.locksSeed = locksSeed
        draft.frames = frames
        draft.fps = fps
        draft.pipeline = pipeline
        if hasAudioPreference == nil {
            // A descriptor from before capability and preference were split.
            // Its one bool is the only user-state evidence available. In
            // particular, legacy false stays an explicit off: changing an
            // existing person's saved choice to the new default would be a
            // silent migration of authored state.
            draft.preferredAudio = enableAudio
        } else {
            draft.preferredAudio = hasAudioPreference == true ? preferredAudio : nil
        }
        draft.videoOnly = videoOnly
        draft.strength = strength
        draft.title = title
        draft.tags = tags
        draft.collectionName = collectionName
        draft.autoTagTitle = autoTagTitle
        draft.outputFormat = outputFormat
        draft.upscaleModel = upscaleModel
        draft.savesToGallery = savesToGallery
        draft.canvasIntent = canvasIntent
        draft.media.sourceFit = sourceFit
        draft.advanced.scheduler = scheduler
        draft.advanced.cfgPlus = cfgPlus
        draft.advanced.sampleShift = sampleShift
        draft.advanced.distillStrengthHigh = distillStrengthHigh
        draft.advanced.distillStrengthLow = distillStrengthLow
        draft.advanced.stgScale = stgScale
        draft.advanced.stgBlocks = stgBlocks
        draft.advanced.rescaleScale = rescaleScale
        draft.advanced.modalityScale = modalityScale
        draft.advanced.skipStep = skipStep
    }
}
