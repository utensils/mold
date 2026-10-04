import Foundation

/// Boundary anchors use pixel-frame indices; temporal controls count frames.
public enum BoundaryFramePolicy {
    public static func snapIndex(_ requested: Int, temporal: TemporalProfile, frames: Int) -> Int {
        let ceiling = max(frames - 1, 0)
        let step = max(temporal.frames.step, 1)
        let clamped = min(max(requested, 0), ceiling)
        return min(((clamped + step / 2) / step) * step, ceiling)
    }

    public static func resolve(capabilities: RecipeCapabilities) -> String? {
        if let boundary = capabilities.boundaryFrames, boundary.mode.isVisible,
           ["wan-pair", "h3-endpoints"].contains(boundary.wire) { return boundary.wire }
        if capabilities.wanRecipe?.supportsFirstLastFrame == true { return "wan-pair" }
        return nil
    }

    public static func minimumFrames(capabilities: RecipeCapabilities) -> Int {
        capabilities.boundaryFrames?.minFrames
            ?? capabilities.wanRecipe?.firstLastFrameMinFrames ?? 2
    }

    public static func reconcile(media: inout DraftMedia, capabilities: RecipeCapabilities) {
        let old = media.adoptedReferenceCapabilities.flatMap { resolve(capabilities: $0) } ?? "interpolation"
        let next = resolve(capabilities: capabilities) ?? "interpolation"
        guard old != next else { return }
        media.boundaryKeyframes[old] = media.keyframes
        media.keyframes = media.boundaryKeyframes.removeValue(forKey: next) ?? []
    }

    public static func apply(to draft: inout RenderDraft, capabilities: RecipeCapabilities) {
        guard let wire = resolve(capabilities: capabilities) else { return }
        let frames = max(draft.frames ?? 2, minimumFrames(capabilities: capabilities))
        if !draft.media.keyframes.isEmpty {
            draft.frames = frames
            let first = wire == "wan-pair" ? draft.media.keyframes.first { $0.frame == 0 } : nil
            let last = draft.media.keyframes.last { wire == "h3-endpoints" || $0.frame != 0 }
            draft.media.keyframes = [first, last.map { frame in
                var result = frame
                result.frame = frames - 1
                return result
            }].compactMap { $0 }
            if wire == "wan-pair" {
                draft.media.reconcileSourceImage(supported: false)
                draft.media.reconcileMask(supported: false)
            }
        }
    }

    public static func image(first: Bool, draft: RenderDraft, capabilities: RecipeCapabilities) -> String? {
        if first && resolve(capabilities: capabilities) == "h3-endpoints" { return draft.media.sourceImage }
        return draft.media.keyframes.first { first ? $0.frame == 0 : $0.frame != 0 }?.image
    }

    public static func set(
        first: Bool, picture: ImportedPicture?, draft: inout RenderDraft,
        capabilities: RecipeCapabilities
    ) {
        guard let wire = resolve(capabilities: capabilities) else { return }
        if first && wire == "h3-endpoints" {
            draft.media.sourceImage = picture?.encoded
            draft.media.sourceImageName = picture?.name
            draft.media.sourceImageOriginal = picture?.encoded
            draft.media.sourceImageOriginalName = picture?.name
            draft.media.sourceImagePixels = picture.flatMap { ReferenceCanvas.uprightPixels(ofBase64: $0.encoded) }
        } else {
            draft.media.keyframes.removeAll { first ? $0.frame == 0 : $0.frame != 0 }
            if let picture {
                let frame = first ? 0 : max((draft.frames ?? 2) - 1, 1)
                draft.media.addingKeyframe(KeyframeCondition(frame: frame, image: picture.encoded, name: picture.name))
            }
            if wire == "wan-pair" {
                draft.media.reconcileSourceImage(supported: false)
                draft.media.reconcileMask(supported: false)
            }
        }
        apply(to: &draft, capabilities: capabilities)
    }

    public static func refusal(draft: RenderDraft, capabilities: RecipeCapabilities) -> String? {
        guard let wire = resolve(capabilities: capabilities) else { return nil }
        let first = image(first: true, draft: draft, capabilities: capabilities) != nil
        let last = image(first: false, draft: draft, capabilities: capabilities) != nil
        if wire == "wan-pair", first != last { return "Choose both the first and last frame." }
        if capabilities.boundaryFrames?.firstRequired == true, !first { return "Choose the first frame." }
        if capabilities.boundaryFrames?.lastRequired == true, !last { return "Choose the last frame." }
        return nil
    }
}
