import MoldClient

/// A multi-file chooser advances its fence after each accepted file, while
/// edits elsewhere or a destination switch invalidate every remaining file.
@MainActor
final class ReferenceImportSession {
    private var fence: ReferenceImportFence
    private let recipeID: String?
    private let recipe: GenerationRecipe

    init(controller: GenerateController, recipe: GenerationRecipe, media: DraftMedia) {
        self.recipe = recipe
        recipeID = controller.recipeID
        fence = ReferenceImportFence(model: controller.modelName,
            host: controller.machineChoice ?? controller.hostID, recipe: recipe, media: media)
    }

    func isCurrent(controller: GenerateController, media: DraftMedia) -> Bool {
        controller.recipeID == recipeID && media.adoptedReferenceCapabilities == recipe.capabilities
            && fence.isCurrent(model: controller.modelName,
                host: controller.machineChoice ?? controller.hostID, recipe: recipe, media: media)
    }

    func advance(controller: GenerateController, media: DraftMedia) {
        fence = ReferenceImportFence(model: controller.modelName,
            host: controller.machineChoice ?? controller.hostID, recipe: recipe, media: media)
    }
}
