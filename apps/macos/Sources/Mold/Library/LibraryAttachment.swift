import Foundation
import MoldClient

enum LibraryAttachmentKind { case source, reference }

enum AttachmentModelAdoption {
    /// Mirrors Generate's first-ready rule: only the model restored with the
    /// authored draft keeps its numbers; a fallback model takes its defaults.
    static func keepsDraft(restoredModel: String?, adopting model: String) -> Bool {
        restoredModel == model
    }
}

struct LibraryAttachmentOffer: Equatable {
    let canUseAsSource: Bool
    let canAddReference: Bool
}

struct AttachmentModelKey: Hashable {
    let host: MoldHost.ID?
    let isUp: Bool
}

/// The UI facts an in-flight authenticated media read was started for.
struct LibraryAttachmentTransaction {
    let version: Int
    let cursor: LibraryCursor.Selection
    let viewed: PrintID?
    let scope: LibraryScope
    let destination: Destination

    func permits(
        version currentVersion: Int, cursor currentCursor: LibraryCursor.Selection,
        viewed currentViewed: PrintID?, scope currentScope: LibraryScope,
        destination currentDestination: Destination, draftIsCurrent: Bool,
        liveEntry: LibraryEntry?, sourceIsUp: Bool
    ) -> Bool {
        version == currentVersion && cursor == currentCursor && viewed == currentViewed
            && scope == currentScope && destination == .library
            && currentDestination == destination && draftIsCurrent
            && liveEntry?.isAttachableRaster == true && sourceIsUp
    }
}

extension LibraryEntry {
    /// A live still the destination well can decode. In particular WAV is
    /// neither video nor mesh, but it is not a picture.
    var isAttachableRaster: Bool {
        guard print.trashedAt == nil, !print.isVideo, !print.isMesh else { return false }
        let raster = Set(["png", "jpeg", "jpg", "webp"])
        if let format = print.format?.lowercased() { return raster.contains(format) }
        let ext = (print.filename as NSString).pathExtension.lowercased()
        return raster.contains(ext)
    }
}

/// Everything that must stay put while authenticated print bytes are fetched.
struct DraftAttachmentFence: Equatable {
    let draft: RenderDraft
    let modelName: String?
    let hostID: MoldHost.ID?
    let machineChoice: MoldHost.ID?
    let recipeID: String?
    let reuseVersion: Int

    init(controller: GenerateController, reuse: ReuseStore) {
        draft = controller.draft
        modelName = controller.modelName
        hostID = controller.hostID
        machineChoice = controller.machineChoice
        recipeID = controller.recipeID
        reuseVersion = reuse.currentFence
    }

    func permits(_ controller: GenerateController, reuse: ReuseStore) -> Bool {
        self == DraftAttachmentFence(controller: controller, reuse: reuse)
    }
}

extension LibraryPane {
    var attachmentOffer: LibraryAttachmentOffer {
        guard let context = attachmentContext else {
            return LibraryAttachmentOffer(canUseAsSource: false, canAddReference: false)
        }
        let layout = ImageConditioningWells.layout(
            recipe: context.recipe, model: context.model, media: generate.draft.media)
        return LibraryAttachmentOffer(
            canUseAsSource: layout.showsSourceWell,
            canAddReference: layout.references?.hasRoom(
                for: generate.draft.media.editImages.count) == true)
    }
}
