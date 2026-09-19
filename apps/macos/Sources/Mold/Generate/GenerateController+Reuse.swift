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
            profile: model.generationProfile, modelSupportsAudio: model.supportsAudio)
    }
}
