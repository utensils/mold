import MoldClient
import MoldStyle
import SwiftUI

// LTX-2's `guidance_overrides`: the second half of the Sampler group, split
// purely for size.
//
// Each of these replaces exactly ONE constant the chosen pipeline pins for its
// stage, and only for guiders the pipeline already runs -- an override never
// switches one on (`types.rs:2598-2600`). That is why `modalityScale` is
// ABSENT on an audio-only pipeline rather than offered and refused: there is
// no video modality for it to guide against, and admission says so
// (`validation.rs:3302-3312`).
extension SamplerGroup {
    @ViewBuilder var guidanceRows: some View {
        if offered.guidance {
            OptionalNumberRow(
                title: "STG scale", placeholder: "Pipeline default",
                value: $draft.advanced.stgScale,
                refusal: AdvancedControls.scaleRefusal(
                    draft.advanced.stgScale, "STG scale", AdvancedControls.maxGuidanceScale))
            stgBlocksRow
            OptionalNumberRow(
                title: "CFG rescale", placeholder: "Pipeline default",
                value: $draft.advanced.rescaleScale,
                refusal: AdvancedControls.scaleRefusal(
                    draft.advanced.rescaleScale, "CFG rescale", 1))
            if offered.modalityScale {
                OptionalNumberRow(
                    title: "Modality scale", placeholder: "Pipeline default",
                    value: $draft.advanced.modalityScale,
                    refusal: AdvancedControls.scaleRefusal(
                        draft.advanced.modalityScale, "Modality scale",
                        AdvancedControls.maxGuidanceScale))
            }
            OptionalStepRow(
                title: "Guidance skip", placeholder: "Pipeline default",
                value: $draft.advanced.skipStep,
                refusal: AdvancedControls.skipStepRefusal(draft.advanced.skipStep))
        }
    }

    /// The perturbed blocks, as the comma-separated list every mold surface
    /// takes. Free text on purpose: the ceiling is the resolved checkpoint's
    /// own transformer depth, which no client knows, so a list this app
    /// cannot fault is still the server's to refuse.
    @ViewBuilder private var stgBlocksRow: some View {
        LabeledSection("STG blocks") {
            VStack(alignment: .leading, spacing: 3) {
                TextField("Pipeline default", text: $draft.advanced.stgBlocks)
                    .textFieldStyle(.roundedBorder)
                    .frame(maxWidth: 140)
                    .accessibilityLabel("STG blocks")
                if let refusal = draft.advanced.stgBlocksRefusal {
                    Text(refusal).font(.caption).foregroundStyle(.secondary)
                }
            }
        }
    }
}
