import Foundation
import MoldClient

extension GenerateController {
    /// Attach the selected print while retaining the authored model and settings.
    func useAsSource(_ entry: LibraryEntry, role: GenerationImageReferenceRole? = nil) async {
        let fence = ReferenceImportFence(model: modelName, host: target?.id, recipe: recipe, media: draft.media)
        let source = entry.presented(onAnyOf: Set(hosts.upHosts.map(\.id))) ?? entry
        guard source.print.kind == .picture, let host = hosts.host(source.hostID) else { return }
        do {
            let bytes = try await hosts.backend(for: host).media(source.print.filename, trashed: false)
            let picked = try await PictureImport.conforming(bytes, name: source.print.filename,
                                                           accepting: PictureImport.engineReadable)
            guard !Task.isCancelled, fence.isCurrent(model: modelName, host: target?.id, recipe: recipe, media: draft.media) else { return }
            if let caps = recipe?.capabilities,
               caps.mesh?.namedViews?.mode.isVisible == true || caps.generationReferences?.mode.isVisible == true {
                var media = draft.media
                let reference = try GenerationReferenceImporter.image(picked, role: role)
                if let role { media.generationReferences.removeAll { $0.role == role } }
                media.appendGenerationReference(reference)
                if let error = media.generationReferenceError(capabilities: caps, allowIncomplete: true) {
                    throw MoldClientError.unreachable(error)
                }
                draft.media = media
            } else if let caps = recipe?.capabilities, BoundaryFramePolicy.resolve(capabilities: caps) != nil {
                BoundaryFramePolicy.set(first: true, picture: picked, draft: &draft, capabilities: caps, recipe: recipe)
            } else if let capability = recipe?.capabilities.referenceImages(family: model?.family, model: modelName), capability.sourceRelation == .replaces {
                guard capability.hasRoom(for: draft.media.editImages.count) else {
                    throw SourceAttachmentFailure.referencesFull
                }
                DraftPictureAttachment.addReference(picked, to: &draft, capability: capability, recipe: recipe)
            } else {
                DraftPictureAttachment.useAsSource(picked, in: &draft, recipe: recipe)
            }
            retainedReuse.clear()
            saveDraft()
        } catch {
            guard !Task.isCancelled else { return }
            hosts.report(host, doing: String(localized: "fetch that picture"), error)
        }
    }
}

private enum SourceAttachmentFailure: LocalizedError {
    case referencesFull
    var errorDescription: String? { "Remove a reference image before adding another picture." }
}
