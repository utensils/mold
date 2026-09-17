import Foundation

/// Whether a clip is one denoise or an automatic chain of them.
///
/// Port of `studio/lib/chainRouting.ts`, whose own constants mirror
/// `crates/mold-cli/src/commands/chain.rs` and `ChainRequest::normalise`.
///
/// An auto-chained one-shot is NOT an authored sequence: there is no scenes
/// UI, no timeline, and the result is ONE print. It exists only because the
/// checkpoint cannot render that many frames in one pass.
public enum ChainRouting {
    public enum Decision: Equatable, Sendable {
        /// One denoise. `preserved` names the options that forced an otherwise
        /// automatic chain to stay one render so their semantics survive.
        case single(preserved: [AutoChainField] = [])
        case chain(clipFrames: Int, motionTail: Int, stageCount: Int)
        case reject(String)
    }

    /// The ROUTING clip size, not LTX-2's ceiling: its real single-request
    /// limit is a runtime budget that moves with fps, and 97 is simply the
    /// clip that fits comfortably on one consumer GPU (`chainRouting.ts:6-9`).
    public static let ltx2DefaultClipFrames = 97
    public static let maxChainStages = 16
    /// 17 pixel frames become three LTX-2 latent frames of carryover under the
    /// VAE's 8x causal temporal compression (`chainRouting.ts:17-20`).
    public static let defaultMotionTail = 17
    /// Wan's seam re-renders exactly the one frame it was seeded with
    /// (`chainRouting.ts:266-273`).
    public static let wanHandoffDuplicatedFrames = 1

    /// Families that may turn a one-shot into a context-preserving chain.
    /// Legacy LTX-Video is deliberately absent: its one-shot router stays
    /// SINGLE up to the engine ceiling (`chainRouting.ts:146-152`).
    public static let autoChainCapableFamilies: Set<String> = ["ltx2", "wan"]
    /// Families where latent context crosses a seam for EVERY checkpoint. Wan
    /// is deliberately absent -- its handoff is last-frame IMAGE conditioning,
    /// which only an image-conditioned checkpoint accepts (#783), so its
    /// answer comes from the advertised contract (`chainRouting.ts:154-164`).
    public static let familiesWithContextHandoff: Set<String> = ["ltx2"]

    public static func canonical(_ family: String?) -> String {
        let normalized = (family ?? "").trimmingCharacters(in: .whitespaces).lowercased()
        return normalized == "ltx-2" ? "ltx2" : normalized
    }

    /// Whether a wan checkpoint carries context across a clip boundary.
    /// Mirrors `mold_inference::chain::wan_carryover`: `required` is the A14B
    /// I2V concat and `optional` the TI2V-5B latent inpaint; `unsupported` is
    /// text-to-video only and an unclassified contract is UNKNOWN, never an
    /// assumed handoff (`chainRouting.ts:166-178`).
    public static func wanCarriesContext(_ sourceImage: SourceImageCapability?) -> Bool {
        sourceImage == .required || sourceImage == .optional
    }

    /// The routing decision. `advertisedMaxFrames` is the HOST's own
    /// single-request ceiling and outranks anything this module knows.
    public static func decide(
        frames: Int?, family: String?, model: String,
        motionTail: Int = defaultMotionTail,
        sourceImage: SourceImageCapability? = nil,
        tierDefault: Int? = nil, advertisedMaxFrames: Int? = nil
    ) -> Decision {
        guard let frames, frames > 0 else { return .single() }
        let normalized = canonical(family)
        guard autoChainCapableFamilies.contains(normalized) else {
            // A non-chainable model still gets its own advertised ceiling.
            let cap = advertisedMaxFrames ?? ltx2DefaultClipFrames
            guard frames > cap else { return .single() }
            return .reject("Model '\(model)' does not support chained video generation. "
                + "Reduce frames to \(cap) or less.")
        }

        let isWan = normalized == "wan"
        let clipFrames = isWan
            ? ClipLengthBounds.wanRoutingClipFrames(model: model, tierDefault: tierDefault)
            : ltx2DefaultClipFrames
        guard frames > clipFrames else { return .single() }

        // A family that carries nothing across a seam cannot be auto-chained
        // into a longer video -- it would render the same clip again. The
        // sentence is `mold_core::chain::text_only_auto_chain_refusal`'s, so
        // this app, the CLI and the server's own 422 read the same.
        if let refusal = textOnlyRefusal(family: normalized, model: model,
                                         sourceImage: sourceImage, totalFrames: frames,
                                         clipFrames: clipFrames) {
            return .reject(refusal)
        }

        let effectiveTail = isWan
            ? (wanCarriesContext(sourceImage) ? wanHandoffDuplicatedFrames : 0)
            : (familiesWithContextHandoff.contains(normalized) ? motionTail : 0)
        guard effectiveTail < clipFrames else {
            return .reject("motion tail (\(effectiveTail)) must be strictly less than "
                + "clip frames (\(clipFrames)).")
        }

        // The first clip emits `clipFrames`; each continuation contributes the
        // clip minus its trimmed motion tail.
        let effective = clipFrames - effectiveTail
        let remainder = frames - clipFrames
        let stageCount = 1 + Int((Double(remainder) / Double(effective)).rounded(.up))
        guard stageCount <= maxChainStages else {
            let maxFrames = clipFrames + (maxChainStages - 1) * effective
            return .reject("Chained video supports at most \(maxFrames) frames "
                + "(\(maxChainStages) clips) for this model. Reduce the frame count.")
        }
        return .chain(clipFrames: clipFrames, motionTail: effectiveTail, stageCount: stageCount)
    }
}
