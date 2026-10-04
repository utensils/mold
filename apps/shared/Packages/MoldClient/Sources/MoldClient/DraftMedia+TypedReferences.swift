import Foundation

public extension DraftMedia {
    mutating func appendGenerationReference(_ reference: GenerationReference) {
        generationReferences.append(reference)
    }
    mutating func replaceGenerationReference(at index: Int, with reference: GenerationReference) {
        guard generationReferences.indices.contains(index) else { return }
        generationReferences[index] = reference
    }
    mutating func removeGenerationReference(at index: Int) {
        guard generationReferences.indices.contains(index) else { return }
        generationReferences.remove(at: index)
    }
    mutating func moveGenerationReference(from index: Int, to destination: Int) {
        guard generationReferences.indices.contains(index), generationReferences.indices.contains(destination) else { return }
        generationReferences.insert(generationReferences.remove(at: index), at: destination)
    }
    mutating func reconcileGenerationReferences(capabilities: RecipeCapabilities) {
        let kinds: Set<String>
        if capabilities.generationReferences?.mode.isVisible == true {
            kinds = Set(capabilities.generationReferences?.kinds ?? [])
        } else if capabilities.mesh?.namedViews?.mode.isVisible == true {
            kinds = ["named_image"]
        } else { kinds = [] }
        let unsupported = generationReferences.filter { !kinds.contains($0.kind) }
        parked.generationReferences.append(contentsOf: unsupported)
        generationReferences.removeAll { !kinds.contains($0.kind) }
        let restored = parked.generationReferences.filter { kinds.contains($0.kind) }
        generationReferences.append(contentsOf: restored)
        parked.generationReferences.removeAll { kinds.contains($0.kind) }
    }
    func generationReferenceError(capabilities: RecipeCapabilities, allowIncomplete: Bool = false) -> String? {
        if let named = capabilities.mesh?.namedViews, named.mode.isVisible {
            guard allowIncomplete || generationReferences.count >= named.minCount else { return "Attach at least one named camera view." }
            guard generationReferences.count <= named.maxCount else { return "Attach at most \(named.maxCount) camera views." }
            let roles = generationReferences.compactMap(\.role)
            if roles.count != generationReferences.count || Set(roles).count != roles.count ||
                !roles.allSatisfy({ named.roles.contains($0) }) || generationReferences.contains(where: { $0.kind != "named_image" }) {
                return "Each camera view needs a distinct advertised role."
            }
        } else if let cap = capabilities.generationReferences, cap.mode.isVisible {
            if !allowIncomplete && cap.required && generationReferences.isEmpty { return "Attach at least one reference." }
            if generationReferences.count > cap.maxCount { return "Attach at most \(cap.maxCount) references." }
            if generationReferences.contains(where: { !cap.kinds.contains($0.kind) }) { return "This model cannot use this reference kind." }
            for (kind, maximum) in [("image", cap.maxImages), ("video", cap.maxVideos), ("audio", cap.maxAudios)] {
                if generationReferences.filter({ $0.kind == kind }).count > maximum { return "Attach at most \(maximum) \(kind) references." }
            }
            if !allowIncomplete && cap.requiresVisual && !generationReferences.isEmpty && !generationReferences.contains(where: { ["image", "video"].contains($0.kind) }) {
                return "Audio references require an image or video reference."
            }
            for reference in generationReferences where ["video", "audio"].contains(reference.kind) {
                guard let duration = reference.durationMs, (cap.minDurationMs...cap.maxDurationMs).contains(duration) else { return "Reference clips must be 2–15 seconds long." }
                if reference.kind == "video" && (reference.frameCount ?? 0) <= 0 { return "The video reference needs an exact decoded frame count." }
                if reference.kind == "audio" && (reference.sampleCount ?? 0) <= 0 { return "The audio reference needs an exact decoded sample count." }
            }
            let video = generationReferences.filter { $0.kind == "video" }.reduce(0) { $0 + ($1.durationMs ?? 0) }
            let audio = generationReferences.reduce(0) { $0 + ($1.kind == "audio" ? ($1.durationMs ?? 0) : ($1.audioDurationMs ?? 0)) }
            if video > cap.maxVideoDurationMs || audio > cap.maxAudioDurationMs { return "Total reference video and audio durations must each fit within 15 seconds." }
        } else if !generationReferences.isEmpty { return "This model does not accept typed references." }
        return nil
    }
}
