import Foundation

// The numeric and adapter controls a request carries. Split from
// `RenderRequest.swift`, which was already at the 150-line lint.
extension RenderRequest {
    /// `control_image` and `control_model` are a symmetric pair
    /// (`validation.rs:3079-3090`): either alone is refused. The Refine
    /// group's picker and picture well can each be filled in before the
    /// other, so a draft with only one half sends NEITHER rather than a
    /// request the server would 422.
    static func applyControl(_ draft: RenderDraft, to request: inout GenerateRequest) {
        guard let image = draft.media.control?.image,
              let model = draft.media.control?.model else { return }
        request.controlImage = image
        request.controlModel = model
        request.controlScale = Swift.max(draft.media.control?.scale ?? Control.defaultScale, 0)
    }

    /// The sampler controls, every one of them absent unless it was moved.
    ///
    /// There is no capability gate HERE and there must not be one: everything
    /// this recipe does not advertise was already PARKED when the recipe was
    /// adopted (`AdvancedControls+Park.swift`), so what is live is by
    /// construction what this recipe offers. A second gate at request time
    /// would be a second opinion that could disagree with the controls on
    /// screen -- the mistake `requestConditioning` exists to prevent for the
    /// picture wells.
    ///
    /// A value the wire cannot carry contributes nothing, exactly as
    /// `guidanceOverridesToWire` drops one (`guidanceOverrides.ts:164-185`);
    /// `AdvancedControls.refusal` is what the pane shows first, so this is
    /// never the user's only feedback.
    static func applyAdvanced(_ draft: RenderDraft, to request: inout GenerateRequest) {
        let advanced = draft.advanced
        request.scheduler = advanced.scheduler
        // `true` or nothing. Absence IS false to the server, so an explicit
        // `false` would record a choice nobody made.
        request.cfgPlus = advanced.cfgPlus ? true : nil
        if AdvancedControls.sampleShiftRefusal(advanced.sampleShift) == nil {
            request.sampleShift = advanced.sampleShift
        }
        if AdvancedControls.distillRefusal(advanced.distillStrengthHigh, "High-noise") == nil {
            request.distillStrengthHigh = advanced.distillStrengthHigh
        }
        if AdvancedControls.distillRefusal(advanced.distillStrengthLow, "Low-noise") == nil {
            request.distillStrengthLow = advanced.distillStrengthLow
        }
        request.guidanceOverrides = guidanceOverrides(advanced)
    }

    /// The crop policy, as PROVENANCE. Only alongside a source image that
    /// really ships: the value describes what was done to those bytes, and on
    /// a render carrying none it would describe nothing.
    static func applySourceFit(
        _ draft: RenderDraft, to request: inout GenerateRequest, carriesSource: Bool
    ) {
        guard carriesSource else { request.sourceFit = nil; return }
        // The defensive half. The adopt above has already coerced it, so
        // this can only ever agree -- which is the point of a belt.
        request.sourceFit = draft.media.acceptsMask
            ? draft.media.sourceFit : draft.media.sourceFit.coercedForMaskless()
    }

    /// LTX-2's overrides, or NOTHING. An empty object is refused outright
    /// ("guidance_overrides must set at least one field; omit it to keep
    /// pipeline defaults", `validation.rs:1728-1733`), so the absent case has
    /// to be absence and not `{}`.
    private static func guidanceOverrides(
        _ advanced: AdvancedControls
    ) -> Ltx2GuidanceOverrides? {
        var overrides = Ltx2GuidanceOverrides()
        if AdvancedControls.scaleRefusal(
            advanced.stgScale, "STG scale", AdvancedControls.maxGuidanceScale) == nil {
            overrides.stgScale = advanced.stgScale
        }
        overrides.stgBlocks = advanced.parsedStgBlocks
        if AdvancedControls.scaleRefusal(advanced.rescaleScale, "CFG rescale", 1) == nil {
            overrides.rescaleScale = advanced.rescaleScale
        }
        if AdvancedControls.scaleRefusal(
            advanced.modalityScale, "Modality scale", AdvancedControls.maxGuidanceScale) == nil {
            overrides.modalityScale = advanced.modalityScale
        }
        // A `u32` on the wire: a fractional or out-of-band value fails JSON
        // deserialization outright, which comes back as an opaque body-parse
        // failure rather than a field-named 422 (`guidanceOverrides.ts:108-121`).
        if AdvancedControls.skipStepRefusal(advanced.skipStep) == nil {
            overrides.skipStep = advanced.skipStep
        }
        return overrides.wire
    }
}
