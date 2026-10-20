import Foundation
import MoldClient

extension GenerateController {
    /// Attach the selected print while retaining the authored model and settings.
    func useAsSource(_ entry: LibraryEntry, role: GenerationImageReferenceRole? = nil, attachment: SourceMediaAttachment? = nil) async {
        let fence = ReferenceImportFence(model: modelName, host: target?.id, recipe: recipe, media: draft.media, videoSource: videoDuration?.draft.source, videoSourceOwner: videoDuration?.sourceOwner)
        let source = entry.presented(onAnyOf: Set(hosts.upHosts.map(\.id))) ?? entry
        guard source.print.kind == .picture, let host = hosts.host(source.hostID) else { return }
        let sourceInstance = hosts.instanceID(of: host.id)
        let targetRoute = target
        let targetInstance = target.flatMap { hosts.instanceID(of: $0.id) }
        let revisions = mediaRevisions
        func current() -> Bool {
            !Task.isCancelled && hosts.host(host.id) == host && hosts.instanceID(of: host.id) == sourceInstance
                && target == targetRoute && target.flatMap { hosts.instanceID(of: $0.id) } == targetInstance
                && revisions == mediaRevisions && fence.isCurrent(model: modelName, host: target?.id, recipe: recipe, media: draft.media, videoSource: videoDuration?.draft.source, videoSourceOwner: videoDuration?.sourceOwner)
        }
        do {
            let bytes = try await hosts.backend(for: host).media(source.print.filename, trashed: false)
            let picked = try await PictureImport.conforming(bytes, name: source.print.filename,
                                                           accepting: PictureImport.engineReadable)
            guard current() else { return }
            if let attachment, let recipe {
                let member = RetainedSourceMedia.Member(memberId: "", role: "source_image", displayName: source.print.filename, sizeBytes: bytes.count, mediaKind: "image")
                let snapshot = draft
                let staged = try await attachment.applying(data: bytes, member: member, to: snapshot, recipe: recipe)
                guard current() else { return }
                draft.media = staged.media
                if draft.width == snapshot.width { draft.width = staged.width }
                if draft.height == snapshot.height { draft.height = staged.height }
                if draft.frames == snapshot.frames { draft.frames = staged.frames }
                saveDraft()
                return
            }
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
            } else if let recipe, SourceMediaAttachment.acceptsEndpoints(capabilities: recipe.capabilities) {
                SourceMediaAttachment.setEndpoint(first: true, picture: picked, draft: &draft, recipe: recipe)
            } else if let capability = recipe?.capabilities.referenceImages(family: model?.family, model: modelName), capability.sourceRelation == .replaces {
                guard capability.hasRoom(for: draft.media.editImages.count) else {
                    throw SourceAttachmentFailure.referencesFull
                }
                DraftPictureAttachment.addReference(picked, to: &draft, capability: capability, recipe: recipe)
            } else {
                DraftPictureAttachment.useAsSource(picked, in: &draft, recipe: recipe)
            }
            // An authored picture overrides its own role. Other retained
            // roles (audio, identity, control) still belong to this reuse.
            saveDraft()
        } catch {
            guard current() else { return }
            hosts.report(host, doing: String(localized: "fetch that picture"), error)
        }
    }
}

private enum SourceAttachmentFailure: LocalizedError {
    case referencesFull
    var errorDescription: String? { "Remove a reference image before adding another picture." }
}
