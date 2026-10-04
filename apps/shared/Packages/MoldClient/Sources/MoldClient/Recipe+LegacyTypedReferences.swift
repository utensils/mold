import Foundation

public extension RecipeCapabilities {
    /// Compatibility for hosts predating the additive contract. Exact reviewed
    /// H3 identities mirror Studio; an advertised block always overrides this.
    func resolvingLegacyReferences(model: String?) -> Self {
        guard let model else { return self }
        let normalized = model.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let parts = normalized.split(separator: ":", maxSplits: 1).map(String.init)
        guard parts.count == 2 else { return self }
        let fl = Set(["comfy-pruned-int8", "comfy-pruned-int8-turbo-8step", "comfy-pruned-int8-turbo-4step-768p", "comfy-pruned-int8-turbo-4step-768p-v1.1", "comfy-pruned-int8-turbo-8step-768p", "comfy-pruned-int8-turbo-4step-768p-r21", "comfy-pruned-int8-turbo-8step-r21"])
        let ref = Set(["comfy-pruned-int8", "comfy-pruned-int8-turbo-4step", "comfy-pruned-int8-turbo-4step-r21"])
        var result = self
        if parts[0] == "minimax-h3-ref2va", ref.contains(parts[1]), generationReferences == nil {
            result.generationReferences = .init(mode: .adjustable, required: true,
                kinds: ["image", "video", "audio"], maxCount: 12, maxImages: 9, maxVideos: 3, maxAudios: 3,
                minDurationMs: 2000, maxDurationMs: 15000, maxVideoDurationMs: 15000,
                maxAudioDurationMs: 15000, maxInlineBytes: 32 * 1024 * 1024, requiresVisual: true)
        }
        if parts[0] == "minimax-h3-fl2va", fl.contains(parts[1]), boundaryFrames == nil {
            result.boundaryFrames = .init(mode: .adjustable, firstRequired: false,
                lastRequired: false, minFrames: 107, wire: "h3-endpoints")
        }
        return result
    }
}

public extension GenerationRecipe {
    func resolvingReferenceCapabilities(family: String? = nil, model: String?) -> Self {
        GenerationRecipe(id: id, label: label, defaults: defaults, resolution: resolution,
            steps: steps, guidance: guidance, temporal: temporal,
            capabilities: capabilities.resolvingLegacyReferences(model: model), requestSelector: requestSelector)
    }
}
