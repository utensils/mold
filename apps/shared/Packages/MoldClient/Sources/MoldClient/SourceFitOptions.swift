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
        guard let recipe, media.requestConditioning.carriesSource, media.extendVideo == nil,
              recipe.resolution.domain != .sourceDriven, recipe.resolution.hasCanvas else { return [] }
        let modes: [SourceFitMode] = [.cropFill, .padFit, .lanczosResize]
        return recipe.capabilities.acceptsMask && recipe.capabilities.readsSourceImage
            ? [.padRepaint] + modes : modes
    }
}
