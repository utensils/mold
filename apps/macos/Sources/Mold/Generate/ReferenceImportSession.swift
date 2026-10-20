import MoldClient

/// A multi-file chooser advances its fence after each accepted file, while
/// edits elsewhere or a destination switch invalidate every remaining file.
@MainActor
final class ReferenceImportSession {
    private var fence: ReferenceImportFence
    private let recipeID: String?
    private let recipe: GenerationRecipe
    private let authority: MoldHost?
    private let instance: String?

    init(controller: GenerateController, recipe: GenerationRecipe, media: DraftMedia, authority: MoldHost? = nil) {
        self.recipe = recipe
        self.authority = authority
        self.instance = authority.flatMap { controller.hosts.instanceID(of: $0.id) }
        recipeID = controller.recipeID
        fence = ReferenceImportFence(model: controller.modelName,
            host: controller.machineChoice ?? controller.hostID, recipe: recipe, media: media, authority: authority, instance: instance, videoSource: controller.videoDuration?.draft.source, videoSourceOwner: controller.videoDuration?.sourceOwner)
    }

    func isCurrent(controller: GenerateController, media: DraftMedia) -> Bool {
        controller.recipeID == recipeID && media.adoptedReferenceCapabilities == recipe.capabilities
            && fence.isCurrent(model: controller.modelName,
                host: controller.machineChoice ?? controller.hostID, recipe: recipe, media: media, authority: authority.flatMap { controller.hosts.host($0.id) }, instance: authority.flatMap { controller.hosts.instanceID(of: $0.id) }, videoSource: controller.videoDuration?.draft.source, videoSourceOwner: controller.videoDuration?.sourceOwner)
    }

    func advance(controller: GenerateController, media: DraftMedia) {
        fence = ReferenceImportFence(model: controller.modelName,
            host: controller.machineChoice ?? controller.hostID, recipe: recipe, media: media, authority: authority.flatMap { controller.hosts.host($0.id) }, instance: authority.flatMap { controller.hosts.instanceID(of: $0.id) }, videoSource: controller.videoDuration?.draft.source, videoSourceOwner: controller.videoDuration?.sourceOwner)
    }
}
