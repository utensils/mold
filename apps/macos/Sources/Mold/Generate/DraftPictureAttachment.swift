import MoldClient

/// One write path for a picture entering an authored draft outside a well.
enum DraftPictureAttachment {
    static func useAsSource(
        _ picked: ImportedPicture, in draft: inout RenderDraft, recipe: GenerationRecipe?
    ) {
        let replaced = draft.media.sourceImageOriginal != picked.encoded
        draft.media.sourceImageOriginal = picked.encoded
        draft.media.sourceImageOriginalName = picked.name
        draft.media.sourceImage = picked.encoded
        draft.media.sourceImageName = picked.name
        if let size = PictureImport.pixelSize(of: picked.data) {
            draft.media.sourceImagePixels = SourcePixels(width: size.width, height: size.height)
            draft.attachSourceShape(size, recipe: recipe, replaced: replaced)
        }
        draft.media.lastExclusiveWrite = .source
    }

    static func addReference(
        _ picked: ImportedPicture, to draft: inout RenderDraft,
        capability: ReferenceImagesCapability
    ) {
        guard capability.hasRoom(for: draft.media.editImages.count) else { return }
        draft.media.editImages.append(picked.encoded)
        draft.media.lastExclusiveWrite = .references
    }
}
