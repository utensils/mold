import MoldClient

extension DraftPersistence {
    /// Restores once for the whole window, whichever destination mounts first.
    func restore(into controller: GenerateController) {
        guard let descriptor = restore(), controller.draft == RenderDraft() else { return }
        var draft = controller.draft
        descriptor.apply(to: &draft)
        controller.draft = draft
        controller.modelFamily = descriptor.family ?? controller.modelFamily
        controller.recipeID = descriptor.recipeID ?? controller.recipeID
    }
}
