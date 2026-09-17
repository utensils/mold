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
    mutating func reconcile(for capabilities: RecipeCapabilities) {
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
        let references = capabilities.referenceImages
        let referencesVisible = references?.mode.isVisible == true
        reconcileEditImages(supported: referencesVisible, maxCount: references?.maxCount)
        if !referencesVisible { referenceWeight = nil }

        // `exclusive`/`replaces` mean ONE render carries a source image OR
        // references, never both. Keeping whichever was added last would be
        // guessing, so references win -- they are the more specific
        // instruction. `readsSourceImage` is the CORRECTED reading of an
        // absent `sourceImage` block: absence means the recipe reads one
        // (fact 1 in the M4 design, `manifest.rs:265-270`), not that there is
        // no source path. An extend is a third claimant, and the strongest
        // one -- it pins the continuation's first frames from the source
        // clip's own tail (`validation.rs:1845-1849`).
        let takenByReferences = !editImages.isEmpty
            && (references?.sourceRelation == .exclusive || references?.sourceRelation == .replaces)
        reconcileSourceImage(
            supported: capabilities.readsSourceImage && !takenByReferences && extendVideo == nil
        )

        // The mask needs BOTH the recipe's own permission and a surviving
        // source image -- an orphaned mask over no source is meaningless
        // (`validation.rs:3101-3107`).
        reconcileMask(supported: capabilities.acceptsMask && sourceImage != nil)

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
