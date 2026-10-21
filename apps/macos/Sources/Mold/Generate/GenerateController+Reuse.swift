import MoldClient

extension GenerateController {
    /// Reconciles settings restored from a print when that print cannot adopt
    /// its own model. The currently selected recipe still owns every request
    /// capability even though the reused prompt and controls came elsewhere.
    func reconcileReusedDraft(with model: Model) {
        let recipe = recipeID.flatMap { model.generationProfile?.recipe(named: $0) }
            ?? model.defaultRecipe
        guard let recipe else { return }
        draft = draft.adopting(
            recipe, isNewModel: false, family: model.family, model: model.name,
            profile: model.generationProfile, modelSupportsAudio: model.supportsAudio,
            modelSupportsDurationPrediction: model.offersDurationPrediction)
    }
}

extension GenerateController {
    func applyVideoSettings(_ restored: VideoSettingsReuse, legacy: LegacySequenceSettingsReuse?,
                            model: Model, recipe: GenerationRecipe, on host: MoldHost.ID) {
        sequence.startFresh()
        let recipeDraft = legacy?.draft ?? restored.draft
        draft = recipeDraft
        adopt(model: model, on: host, keepingDraft: true, selectedRecipe: recipe)
        // This locates the archived endpoints; requested duration is separate.
        draft.frames = recipeDraft.frames
        videoDuration = restored.duration
        durationSource = nil; durationSourceError = nil
        videoPlanner.clear()
        if let legacy { sequence.restoreSettings(legacy) }
    }
}
