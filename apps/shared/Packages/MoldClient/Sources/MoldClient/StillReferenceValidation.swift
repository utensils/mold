import Foundation

public extension ReferenceImagesCapability {
    var acceptingTypes: Set<String> {
        Set(acceptedFormats.compactMap(PictureImport.typeIdentifier(forFormat:)))
    }

    /// The same single/multiple pixel budget admission applies to each image.
    func stillRefusal(images: [String], retainedImagesAvailable: Bool = false) -> String? {
        if required && images.isEmpty && !retainedImagesAvailable { return "Attach the picture to edit first." }
        if let maxCount, images.count > maxCount { return "This model accepts at most \(maxCount) reference pictures." }
        let limit = images.count > 1 ? maxPixelsMulti : maxPixelsSingle
        for (index, image) in images.enumerated() {
            guard let pixels = ReferenceCanvas.uprightPixels(ofBase64: image) else {
                return "Reference \(index + 1) couldn't be read. Replace that picture."
            }
            if let limit, pixels.width > limit / max(1, pixels.height) {
                return "Reference \(index + 1) exceeds this model's \(limit.formatted()) pixel limit. Choose a smaller picture."
            }
        }
        return nil
    }
}

public extension RenderDraft {
    func stillReferenceRefusal(for recipe: GenerationRecipe, family: String? = nil, model: String? = nil, retainedFields: Set<RetainedSourceMedia.Field> = []) -> String? {
        guard let capability = recipe.capabilities.referenceImages(family: family, model: model) else { return nil }
        // On an exclusive recipe a staged strip may be parked for this request.
        guard capability.required || media.requestConditioning.carriesReferences else { return nil }
        return capability.stillRefusal(images: media.editImages, retainedImagesAvailable: retainedFields.contains(.editImages))
    }
}
