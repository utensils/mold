import Foundation

/// One write path for a picture entering an authored draft outside a well.
/// Putting a picture into a draft: as the source, or as the next reference.
/// Shared by the Mac's wells and the iPhone's.
public enum DraftPictureAttachment {
    public static func useAsSource(
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

    /// Appends one reference. On a `last-reference` recipe the new LAST
    /// picture re-derives a canvas that is still the model's own.
    public static func addReference(
        _ picked: ImportedPicture, to draft: inout RenderDraft,
        capability: ReferenceImagesCapability, recipe: GenerationRecipe?
    ) {
        guard capability.hasRoom(for: draft.media.editImages.count) else { return }
        draft.media.editImages.append(picked.encoded)
        draft.media.lastExclusiveWrite = .references
        draft.followLastReference(recipe: recipe)
    }
    public static func replaceReference(
        _ picked: ImportedPicture, at index: Int, in draft: inout RenderDraft,
        recipe: GenerationRecipe?
    ) {
        guard draft.media.editImages.indices.contains(index) else { return }
        draft.media.editImages[index] = picked.encoded
        draft.media.lastExclusiveWrite = .references
        draft.followLastReference(recipe: recipe)
    }

    public static func removeReference(at index: Int, from draft: inout RenderDraft, recipe: GenerationRecipe?) {
        guard draft.media.editImages.indices.contains(index) else { return }
        draft.media.editImages.remove(at: index)
        draft.followLastReference(recipe: recipe)
    }

    public static func moveReference(from index: Int, to destination: Int, in draft: inout RenderDraft,
                                     recipe: GenerationRecipe?) {
        guard draft.media.editImages.indices.contains(index),
              draft.media.editImages.indices.contains(destination), index != destination else { return }
        let value = draft.media.editImages.remove(at: index)
        draft.media.editImages.insert(value, at: destination)
        draft.media.lastExclusiveWrite = .references
        draft.followLastReference(recipe: recipe)
    }
}
