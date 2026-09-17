import MoldClient

/// What the Length control may offer, and where a render of that length goes.
///
/// ONE answer, resolved from the recipe and the checkpoint, so the slider's
/// ceiling and the router's decision can never disagree -- a slider offering
/// a length submit would turn away is the whole class of bug this replaces.
struct ClipRouting {
    /// What the slider may reach.
    let bounds: ClipLengthBounds
    /// Where a render of the CURRENT length goes.
    let decision: ChainRouting.Decision

    /// The sentence under the slider: the recipe's own note about its ceiling,
    /// or -- once the length is past one clip -- how many clips this will be.
    var note: String? {
        if case let .chain(_, _, stageCount) = decision {
            return "Rendered as \(stageCount) clips and stitched into one video."
        }
        if case let .reject(reason) = decision { return reason }
        return bounds.note
    }

    /// Whether Generate would be turned away right now.
    var refusal: String? {
        if case let .reject(reason) = decision { return reason }
        return nil
    }
}

extension ClipRouting {
    /// Resolves both halves from the same facts.
    ///
    /// The Length ceiling is LIFTED past the single-clip size wherever the
    /// router would chain: an earlier lane capped the slider at the clip and
    /// said so, which was right only for the models that cannot be chained at
    /// all. Where chaining is allowed, the ceiling is the longest chain
    /// `MAX_CHAIN_STAGES` permits, and the refusal a text-only wan or a legacy
    /// LTX-Video earns is the SERVER's sentence rather than a shorter slider.
    /// `limits` is the HOST's own answer for this model
    /// (`/api/capabilities/chain-limits`) and outranks every constant this app
    /// carries. `nil` is a host too OLD to publish the route, which is where
    /// the ported constants belong -- and nowhere else.
    static func resolve(
        recipe: GenerationRecipe, model: Model?, draft: RenderDraft,
        limits: ChainLimits? = nil
    ) -> ClipRouting? {
        guard let temporal = recipe.temporal else { return nil }
        let fps = draft.fps ?? temporal.fps.value
        let sourceImage = recipe.capabilities.sourceImage
        let single = temporal.lengthBounds(
            fps: fps, family: model?.family, model: model?.name, sourceImage: sourceImage)
        let decision = limits.map {
            ChainRouting.decide(
                frames: draft.frames, family: model?.family, model: model?.name ?? "",
                limits: $0, sourceImage: sourceImage)
        } ?? ChainRouting.decide(
            frames: draft.frames, family: model?.family, model: model?.name ?? "",
            sourceImage: sourceImage, tierDefault: temporal.frames.default,
            advertisedMaxFrames: temporal.durationCappedMaxFrames(fps: fps))
        return ClipRouting(bounds: chainedBounds(single, temporal: temporal, model: model,
                                                 sourceImage: sourceImage, limits: limits),
                           decision: decision)
    }

    /// The slider's ceiling once chaining is taken into account.
    ///
    /// A model the router will never chain keeps the single-clip ceiling and
    /// its sentence. A chainable one may reach the longest chain the stage cap
    /// allows -- but never past the recipe's own advertised maximum, which is
    /// a resource guard the host still enforces per clip and per request.
    private static func chainedBounds(
        _ single: ClipLengthBounds, temporal: TemporalProfile,
        model: Model?, sourceImage: SourceImageCapability?, limits: ChainLimits?
    ) -> ClipLengthBounds {
        let family = ChainRouting.canonical(model?.family)
        guard ChainRouting.autoChainCapableFamilies.contains(family) else { return single }
        if let limits, !limits.supportsSequence { return single }
        // A wan tier that hands nothing across a seam has no chain to become,
        // so its clip size IS its ceiling and the note explains why.
        if family == "wan", sourceImage == .unsupported { return single }
        let clip = limits?.framesPerClipRecommended ?? (family == "wan"
            ? ClipLengthBounds.wanRoutingClipFrames(
                model: model?.name ?? "", tierDefault: temporal.frames.default)
            : ChainRouting.ltx2DefaultClipFrames)
        let tail = family == "wan"
            ? (ChainRouting.wanCarriesContext(sourceImage)
                ? ChainRouting.wanHandoffDuplicatedFrames : 0)
            : ChainRouting.defaultMotionTail
        guard tail < clip else { return single }
        let stages = limits?.maxStages ?? ChainRouting.maxChainStages
        var longest = clip + (stages - 1) * (clip - tail)
        if let total = limits?.maxTotalFrames { longest = Swift.min(longest, total) }
        let ceiling = Swift.min(temporal.snapDown(longest), temporal.frames.max)
        guard ceiling > single.max else { return single }
        return ClipLengthBounds(min: single.min, max: ceiling, note: nil)
    }
}
