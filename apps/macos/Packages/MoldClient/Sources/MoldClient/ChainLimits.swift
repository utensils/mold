import Foundation

/// What `GET /api/capabilities/chain-limits?model=…&fps=…` advertises.
///
/// The HOST's answer for one model, and it outranks every constant this app
/// carries. `ChainRouting`'s numbers are a mirror of
/// `crates/mold-cli/src/commands/chain.rs`, which is the right fallback for a
/// host too old to publish this -- and the wrong answer wherever the machine
/// itself knows better. `framesPerClipCap` in particular "must be read rather
/// than recomputed from either half alone" (`chain_limits.rs:33-42`).
public struct ChainLimits: Decodable, Hashable, Sendable {
    public let model: String
    /// The largest clip this model renders as ONE generation at `fps`.
    public let framesPerClipCap: Int
    public let fps: Int?
    /// The clip size the router splits work into -- the model's own default,
    /// snapped to its grid.
    public let framesPerClipRecommended: Int
    public let maxStages: Int
    public let maxTotalFrames: Int
    public let supportsSequence: Bool
    public let sequenceUnsupportedReason: String?
}

public extension ChainRouting {
    /// The routing decision, with the host's own limits in hand.
    ///
    /// The advertised numbers replace the ported constants one for one: the
    /// clip size, the stage cap and the single-request ceiling. Everything
    /// else -- whether the family can chain at all, and the two refusals for
    /// the families that hand nothing across a seam -- stays where it is,
    /// because those are contracts about the CHECKPOINT that no limits block
    /// answers.
    static func decide(
        frames: Int?, family: String?, model: String, limits: ChainLimits,
        sourceImage: SourceImageCapability? = nil,
        motionTail: Int = defaultMotionTail
    ) -> Decision {
        // A host that says this model has no sequence path at all is the
        // authority, and its own sentence is the one to show.
        if !limits.supportsSequence, let frames, frames > limits.framesPerClipCap {
            return .reject(limits.sequenceUnsupportedReason
                ?? "Model '\(model)' does not support chained video generation. "
                + "Reduce frames to \(limits.framesPerClipCap) or less.")
        }
        return decide(frames: frames, family: family, model: model, motionTail: motionTail,
                      sourceImage: sourceImage,
                      tierDefault: limits.framesPerClipRecommended,
                      advertisedMaxFrames: limits.framesPerClipCap,
                      maxStages: limits.maxStages, maxTotalFrames: limits.maxTotalFrames,
                      routingClipFrames: limits.framesPerClipRecommended)
    }
}
