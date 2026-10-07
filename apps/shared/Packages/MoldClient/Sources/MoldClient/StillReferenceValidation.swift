import Foundation

public extension ReferenceImagesCapability {
    var acceptingTypes: Set<String> {
        Set(acceptedFormats.compactMap(PictureImport.typeIdentifier(forFormat:)))
    }

    /// Processing budgets describe engine preparation, not limits on imported originals.
    /// Only the shared reference ingestion envelope can refuse image dimensions.
    func stillRefusal(images: [String], retainedImagesAvailable: Bool = false) -> String? {
        if required && images.isEmpty && !retainedImagesAvailable { return "Attach the picture to edit first." }
        if let maxCount, images.count > maxCount { return "This model accepts at most \(maxCount) reference pictures." }
        for (index, image) in images.enumerated() {
            guard let pixels = ReferenceCanvas.uprightPixels(ofBase64: image) else {
                return "Reference \(index + 1) couldn't be read. Replace that picture."
            }
            if max(pixels.width, pixels.height) > 16_384 || pixels.width > 100_000_000 / max(1, pixels.height) {
                return "Reference \(index + 1) exceeds the safe image input size. Replace that picture."
            }
            if max(pixels.width, pixels.height) > 200 * max(1, min(pixels.width, pixels.height)) {
                return "Reference \(index + 1) has an unsupported aspect ratio. Crop that picture."
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
