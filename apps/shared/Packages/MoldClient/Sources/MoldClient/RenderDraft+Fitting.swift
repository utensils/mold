import Foundation

public extension RenderDraft {
    /// Prepare a snapshot before admission. Fit the original, never a crop
    /// of an earlier crop, and leave references/identity/source-driven inputs alone.
    func fittingSource(recipe: GenerationRecipe?) async throws -> RenderDraft {
        if let recipe, BoundaryFramePolicy.resolve(capabilities: recipe.capabilities) != nil {
            return try await fittingBoundaryFrames(recipe: recipe)
        }
        guard !SourceFitOptions.resolve(recipe: recipe, media: media).isEmpty,
              let encoded = media.sourceImageOriginal ?? media.sourceImage,
              let original = Data(base64Encoded: encoded), width > 0, height > 0 else { return self }
        var fitted = self
        let policy = media.acceptsMask ? media.sourceFit : media.sourceFit.coercedForMaskless()
        let target = (width: width, height: height)
        if let picture = await SourceFitRender.fit(original,
            name: media.sourceImageOriginalName ?? media.sourceImageName ?? "source.png", target: target, policy: policy) {
            fitted.media.sourceImage = picture.encoded
            fitted.media.sourceImageName = picture.name
        } else {
            fitted.media.sourceImage = encoded
            fitted.media.sourceImageName = media.sourceImageOriginalName ?? media.sourceImageName
        }
        try Task.checkCancellation()
        if let transform = SourceFitRender.transform(of: original, target: target, policy: policy), media.acceptsMask,
           let mask = await SourceFitRender.mask(existing: media.maskImage.flatMap { Data(base64Encoded: $0) }, transform: transform, sourceSpace: true) {
            fitted.media.maskImage = mask.base64EncodedString()
        }
        try Task.checkCancellation()
        return fitted
    }

    /// Fit a submission copy of every active endpoint; authoring bytes, frame
    /// indices and parked inputs remain untouched. Endpoints never carry masks.
    func fittingBoundaryFrames(recipe: GenerationRecipe?) async throws -> RenderDraft {
        guard let recipe, let wire = BoundaryFramePolicy.resolve(capabilities: recipe.capabilities),
              !SourceFitOptions.resolve(recipe: recipe, media: media).isEmpty,
              width > 0, height > 0 else { return self }
        try Task.checkCancellation()
        var fitted = self
        let policy = media.sourceFit.coercedForMaskless()
        fitted.media.sourceFit = policy
        fitted.media.maskImage = nil
        let target = (width: width, height: height)
        if wire == "h3-endpoints", media.sourceImage != nil,
           let encoded = media.sourceImageOriginal ?? media.sourceImage,
           let original = Data(base64Encoded: encoded) {
            if let picture = await SourceFitRender.fit(original,
                name: media.sourceImageOriginalName ?? media.sourceImageName ?? "first.png",
                target: target, policy: policy) {
                fitted.media.sourceImage = picture.encoded
                fitted.media.sourceImageName = picture.name
            } else {
                fitted.media.sourceImage = encoded
                fitted.media.sourceImageName = media.sourceImageOriginalName ?? media.sourceImageName
            }
            try Task.checkCancellation()
        }
        for index in media.keyframes.indices {
            try Task.checkCancellation()
            let frame = media.keyframes[index]
            guard let original = Data(base64Encoded: frame.image) else { continue }
            if let picture = await SourceFitRender.fit(original, name: frame.name ?? "frame.png",
                                                      target: target, policy: policy) {
                fitted.media.keyframes[index] = KeyframeCondition(
                    frame: frame.frame, image: picture.encoded, name: picture.name)
            }
            try Task.checkCancellation()
        }
        return fitted
    }

}
