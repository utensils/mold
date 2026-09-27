import Foundation

// The single entry point `RenderDraft+Recipe.swift`'s `adopting` calls into:
// every conditioning field, reconciled against one recipe's capabilities in
// the order the fields depend on each other. Split from `DraftMedia+Park.swift`
// and `+ParkClip.swift` (which hold the per-field methods this calls) purely
// for size.
public extension DraftMedia {
    /// Reconciles every conditioning field against `capabilities` in one
    /// pass -- conditioning the recipe cannot currently take is PARKED
    /// rather than dropped, so it comes back if the next recipe can read it
    /// again (decision 4 in the M4 design).
    /// `family` and `model` are only ever read by the LEGACY reference rule,
    /// for a host that advertises no `reference_images` block at all
    /// (`RecipeCapabilities.referenceImages(family:model:)`). Both default to
    /// nil so a caller with no model in hand -- a test, or a recipe switch
    /// where the advertised block is present anyway -- reads exactly the
    /// advertised contract.
    mutating func reconcile(
        for capabilities: RecipeCapabilities, family: String? = nil, model: String? = nil
    ) {
        // Keyframes and an extend continuation reconcile FIRST: both can
        // park or restore the source image below, and an extend also pins
        // the request's video-only reading (`RenderDraft+Audio.swift`).
        reconcileKeyframes(supported: capabilities.acceptsKeyframes)
        reconcileExtend(supported: capabilities.supportsExtend == true)
        reconcileAudioFile(supported: capabilities.acceptsSourceAudio)
        reconcileSourceVideo(supported: capabilities.acceptsSourceVideo)
        // Keyframes and an extend are mutually exclusive on ONE request
        // (`validation.rs:1851-1853`); `addingKeyframe`/`settingExtend` keep
        // that true while a person edits, but the two live in SEPARATE parks
        // and a recipe switch can restore both at once. Extend wins -- it
        // already parks the source image below, the stronger claim.
        if extendVideo != nil, !keyframes.isEmpty {
            parked.keyframes = keyframes
            keyframes = []
        }

        // Edit images are reconciled next so the exclusive/replaces check
        // below reads the post-truncation list, matching the order this
        // logic ran in before parking existed.
        let references = capabilities.referenceImages(family: family, model: model)
        sourceMode = SourceImageMode(references: references)
        let referencesVisible = references != nil
        reconcileEditImages(supported: referencesVisible, maxCount: references?.maxCount)
        // The weight rides with the WEIGHT CONTROL, not with the strip: a
        // recipe can advertise references and no `weight` block at all (every
        // `replaces` recipe does, and so does any host predating the field),
        // and a value carried onto one would be a number nothing on screen
        // explains. Parked, never dropped -- `reference_images.weight` is a
        // `FloatControl` whose range travels WITH the capability
        // (`generation_profile.rs:470-484`), so the honest reading of an
        // absent control is "not offered here", not "reset to zero".
        Self.reconcile(&referenceWeight, &parked.referenceWeight,
                       supported: referencesVisible && references?.weight?.mode.isVisible == true)
        if !referencesVisible {
            lastExclusiveWrite = nil
        }

        // `replaces` means the references ARE the conditioning: no strength,
        // no mask, no source path at all, so a staged source image is PARKED.
        // `exclusive` is deliberately NOT parked here -- it keeps both wells,
        // and `requestConditioning` decides which one ships (finding 02#1).
        // Parking it would empty a well the layout draws. `readsSourceImage`
        // is the CORRECTED reading of an
        // absent `sourceImage` block: absence means the recipe reads one
        // (fact 1 in the M4 design, `manifest.rs:265-270`), not that there is
        // no source path. An extend is a third claimant, and the strongest
        // one -- it pins the continuation's first frames from the source
        // clip's own tail (`validation.rs:1845-1849`).
        let takenByReferences = !editImages.isEmpty && sourceMode.replacesSourceImage
        reconcileSourceImage(
            supported: capabilities.readsSourceImage && !takenByReferences && extendVideo == nil
        )

        // The mask needs BOTH the recipe's own permission and a surviving
        // source image -- an orphaned mask over no source is meaningless
        // (`validation.rs:3101-3107`).
        acceptsMask = capabilities.acceptsMask && capabilities.readsSourceImage
        reconcileMask(supported: capabilities.acceptsMask && sourceImage != nil)
        // `pad-repaint` on a recipe with no mask path would pad the source
        // with bands the model can never repaint -- and this app would then
        // write a white-band mask into a draft that cannot carry one.
        // `coerceSourceFitForMaskless` is applied "both when entering such a
        // family and defensively on submit" (`sourceFit.ts:261-266`); this is
        // the first half, and `applySourceFit` is the second.
        if !acceptsMask { sourceFit = sourceFit.coercedForMaskless() }
        // This app has no client-side upscale to run first, so a restored or
        // reused `upscale-then-fit` is normalised to the fit it will ACTUALLY
        // perform. Leaving it would bind the Fit picker to a selection
        // matching no row, caption it "Enhances a small picture first…", and
        // then record a policy the pixels never got.
        if sourceFit.mode == .upscaleThenFit { sourceFit = sourceFit.effective }

        // Identity is positive-only: `supportsIdentity != true` means the
        // well is not drawn and the staged photo is held, because sending it
        // to an unqualified checkpoint is a refusal, not a silent ignore
        // (`identity.rs:1011-1013`).
        reconcileIdentity(supported: capabilities.supportsIdentity == true)

        // ControlNet: `controlNet` is nil for a `hidden` block and for no
        // block at all (fact 3 in the M4 design) -- there is a real recipe
        // gate, so this never asks whether an adapter happens to be
        // installed, only whether THIS recipe would read one.
        reconcileControl(supported: capabilities.controlNet != nil)

        reconcileLoras(supported: capabilities.loraStack != nil, maxCount: capabilities.loraStack?.maxCount)
    }
}
