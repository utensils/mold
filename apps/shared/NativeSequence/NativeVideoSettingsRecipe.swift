import MoldClient

/// Settings reuse follows the installed source machine's advertised recipes.
enum NativeVideoSettingsRecipe {
    static func resolve(metadata: OutputMetadata, model: Model) throws -> GenerationRecipe {
        guard let profile = model.generationProfile else {
            throw SequenceDraftError.invalid("Update the original machine or install this video's model before using its settings.")
        }
        if metadata.pipelineRequested == true, let pipeline = metadata.pipeline {
            guard let recipe = profile.recipes.first(where: { $0.makes == .clip && $0.requestSelector?.pipeline == pipeline }) else {
                throw SequenceDraftError.invalid("The original video pipeline is unavailable. Install a compatible recipe before using these settings.")
            }
            return recipe
        }
        guard let recipe = profile.defaultRecipe?.makes == .clip ? profile.defaultRecipe : profile.recipes.first(where: { $0.makes == .clip }) else {
            throw SequenceDraftError.invalid("The original model does not offer a compatible video recipe.")
        }
        return recipe
    }
}
