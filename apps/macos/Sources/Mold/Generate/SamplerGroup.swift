import MoldClient
import MoldStyle
import SwiftUI

/// How the picture is SAMPLED rather than what is in it: the solver, CFG++,
/// and wan's own flow shift and distill strengths. LTX-2's guidance overrides
/// are the same group's second half (`SamplerGroup+Guidance.swift`).
///
/// Every row exists only where the recipe advertises it, and the whole group
/// is absent when it would have nothing in it -- `AdvancedControlsOffered`
/// resolves that once, and both this and the parking rule read the same
/// answer.
struct SamplerGroup: View {
    let offered: AdvancedControlsOffered
    @Binding var draft: RenderDraft

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            solverRow
            cfgPlusRow
            wanRows
            guidanceRows
        }
        .rowActionMenu(GenerateMenus.sampler(touched: draft.advanced.touchedCount > 0),
                       perform: perform)
    }

    private func perform(_ action: GenerateAction) {
        guard action == .resetSampler else { return }
        draft.advanced.reset()
    }

    @ViewBuilder private var solverRow: some View {
        if !offered.schedulers.isEmpty {
            LabeledSection("Solver") {
                Picker("Solver", selection: solver) {
                    // "Default" means OMIT the field, which is how the server
                    // is asked for its own choice -- never a spelling.
                    Text("Default").tag(String?.none)
                    ForEach(offered.schedulers, id: \.self) { name in
                        Text(Self.label(for: name)).tag(String?.some(name))
                    }
                }
                .labelsHidden()
                .fixedSize()
            }
        }
    }

    private var solver: Binding<String?> {
        Binding(get: { draft.advanced.scheduler }, set: { draft.advanced.scheduler = $0 })
    }

    @ViewBuilder private var cfgPlusRow: some View {
        if offered.cfgPlus {
            LabeledSection("Guidance style") {
                Toggle("CFG++", isOn: $draft.advanced.cfgPlus)
                    .help("A different guidance update; the same step count.")
            }
        }
    }

    @ViewBuilder private var wanRows: some View {
        if offered.sampleShift {
            OptionalNumberRow(
                title: "Flow shift", placeholder: "Tier default",
                value: $draft.advanced.sampleShift,
                refusal: AdvancedControls.sampleShiftRefusal(draft.advanced.sampleShift))
        }
        if offered.distillStrength {
            OptionalNumberRow(
                title: "High-noise distill", placeholder: "1",
                value: $draft.advanced.distillStrengthHigh,
                refusal: AdvancedControls.distillRefusal(
                    draft.advanced.distillStrengthHigh, "High-noise"))
            OptionalNumberRow(
                title: "Low-noise distill", placeholder: "1",
                value: $draft.advanced.distillStrengthLow,
                refusal: AdvancedControls.distillRefusal(
                    draft.advanced.distillStrengthLow, "Low-noise"))
        }
    }
}

extension SamplerGroup {
    /// Display names for every solver mold has, so this app and the browser
    /// word them the same way. Port of `SCHEDULER_LABELS`
    /// (`generationCapabilities.ts:247-259`); an unlisted name -- a solver
    /// added after this build -- is shown AS ADVERTISED rather than hidden,
    /// because the recipe offering it is newer than this app, not broken.
    static func label(for scheduler: String) -> String {
        switch scheduler {
        case "ddim": "DDIM"
        case "euler-ancestral": "Euler ancestral"
        case "uni-pc": "UniPC"
        case "edm-dpm-pp-2m": "EDM DPM++ 2M"
        case "euler": "Euler"
        case "dpm-pp": "DPM++"
        default: scheduler
        }
    }

    static func isShown(_ offered: AdvancedControlsOffered) -> Bool { offered.offersAnything }
}
