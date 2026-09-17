import Foundation

/// How a recipe lays out image conditioning: which wells are drawn, and
/// whether one of them parks while the other is in use.
///
/// Projected from the advertised `capabilities.reference_images` block, and
/// never from a model name at a call site. Port of
/// `studio/lib/generationCapabilities.ts:78-112` (`SourceImageMode` and
/// `sourceImageModeForReferences`), which is the ONE client authority that
/// turns the advertised relation into a layout.
public enum SourceImageMode: String, Hashable, Sendable, CaseIterable {
    /// No reference protocol at all: the source well alone.
    case single
    /// `replaces` with the first image as the edit TARGET (Qwen-Image-Edit).
    case qwenEdit
    /// `replaces`: the references ARE the conditioning -- no strength, no
    /// mask, no source well.
    case references
    /// `exclusive`: both wells draw, and whichever holds media parks the
    /// other. ONE render carries a source image OR references, never both.
    case singleOrReferences
    /// `combines` (IP-Adapter on SD1.5/SDXL): the reference is an image
    /// PROMPT injected alongside the text conditioning, so it rides WITH the
    /// source image, its strength, its mask, ControlNet and a LoRA in the
    /// same pass. Both wells are live and NEITHER parks -- which is exactly
    /// why it cannot borrow the exclusive layout.
    case singleAndReferences

    /// The layout a resolved reference contract projects onto. `nil` is a
    /// recipe with no reference protocol -- a `hidden` block or an older host
    /// whose legacy rule answered nothing.
    public init(references: ReferenceImagesCapability?) {
        guard let references else { self = .single; return }
        if references.primaryIsTarget { self = .qwenEdit; return }
        switch references.sourceRelation {
        case .replaces: self = .references
        case .exclusive: self = .singleOrReferences
        case .combines: self = .singleAndReferences
        case .unknown:
            // A relation added after this build. The host is NEWER, not
            // older, so its references are real -- drawing the additive
            // layout offers both wells and lets admission answer a pairing
            // this build cannot reason about, where hiding the strip would
            // make the whole protocol unreachable (the qwen-edit lesson in
            // finding 01#4).
            self = .singleAndReferences
        }
    }

    /// Whether the strip TAKES the source path outright, so a staged source
    /// image has nowhere to go and is parked. True only for `replaces`;
    /// `exclusive` keeps both wells and parks only for the length of one
    /// request (`RequestConditioning`).
    public var replacesSourceImage: Bool { self == .references || self == .qwenEdit }

    /// Whether this layout draws the reference strip at all.
    public var showsReferenceStrip: Bool { self != .single }

    /// Whether this layout draws the source well -- subject to the recipe's
    /// own `readsSourceImage`, which the caller folds in.
    public var showsSourceWell: Bool { !replacesSourceImage }
}

public extension ReferenceImagesCapability {
    /// The pre-profile reference rule, for a host that advertises no
    /// `capabilities.reference_images` block AT ALL.
    ///
    /// Absence of the block means an OLDER SERVER, never a refusal, so this
    /// is consulted only when the field is `nil` -- a `hidden` block is the
    /// server SAYING NO and answers `nil` on its own. Port of
    /// `studio/lib/legacyRecipeRules.ts:104-136` (`legacyReferenceImages`),
    /// the one sanctioned family sniff in the whole contract.
    ///
    /// FLUX.2 [klein] deliberately answers `nil`: Klein's reference protocol
    /// shipped WITH the wire contract, so a host old enough to omit the block
    /// has no Klein reference engine and the wells would promise a render it
    /// refuses.
    static func legacy(family: String?, model: String?) -> ReferenceImagesCapability? {
        let normalizedFamily = (family ?? "").trimmingCharacters(in: .whitespaces).lowercased()
        let normalizedModel = (model ?? "").trimmingCharacters(in: .whitespaces).lowercased()
        if normalizedFamily == "qwen-image-edit" {
            return ReferenceImagesCapability(
                mode: .adjustable, required: true, maxCount: nil, primaryIsTarget: true,
                // An older host advertises no block at all, so it can never
                // have told us about an adapter strength. `nil` renders no
                // slider.
                sourceRelation: .replaces, reason: nil, weight: nil)
        }
        if normalizedModel.contains("flux2-dev") || normalizedModel.contains("flux.2-dev") {
            return ReferenceImagesCapability(
                mode: .adjustable, required: false,
                maxCount: Self.legacyFlux2MaxReferenceImages, primaryIsTarget: false,
                sourceRelation: .replaces, reason: nil, weight: nil)
        }
        return nil
    }

    /// FLUX.2 Dev's reference ceiling before the profile advertised one;
    /// mirrors `mold_core::validation::FLUX2_MAX_REFERENCE_IMAGES`.
    static let legacyFlux2MaxReferenceImages = 4
}
