import Foundation

/// Shared native source-fit choices. Mirrors desktop SourceFitRow and
/// studio/lib/sourceFit.ts:124-142; only source bytes actually sent are fitted.
public enum SourceFitOptions {
    public static func policy(for mode: SourceFitMode, supportsMask: Bool) -> SourceFit {
        switch mode {
        case .cropFill: .default
        case .padRepaint: supportsMask ? .padRepaint : .default
        case .padFit: .padFit
        case .lanczosResize: .lanczosResize
        case .upscaleThenFit: .default
        }
    }

    public static func resolve(recipe: GenerationRecipe?, media: DraftMedia) -> [SourceFitMode] {
        guard let recipe, media.extendVideo == nil,
              recipe.resolution.domain != .sourceDriven, recipe.resolution.hasCanvas else { return [] }
        let boundary = BoundaryFramePolicy.resolve(capabilities: recipe.capabilities)
        let carriesBoundary = boundary != nil && (!media.keyframes.isEmpty
            || (boundary == "h3-endpoints" && media.sourceImage != nil))
        guard carriesBoundary || media.requestConditioning.carriesSource else { return [] }
        let modes: [SourceFitMode] = [.cropFill, .padFit, .lanczosResize]
        return !carriesBoundary && recipe.capabilities.acceptsMask && recipe.capabilities.readsSourceImage
            ? [.padRepaint] + modes : modes
    }
}
