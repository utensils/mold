import MoldClient
import MoldStyle
import SwiftUI

/// How a source picture that is not the canvas's shape is mapped onto it.
///
/// A CLIENT policy with no capability behind it -- the fitting happens on this
/// Mac and the server records the choice verbatim as provenance
/// (`types.rs:3268-3273`). So the gate is not "does the host advertise this"
/// but "is there a source picture that will actually ship", which is
/// `requestConditioning`'s answer and never `sourceImage != nil`.
struct SourceFitRow: View {
    let modes: [SourceFitMode]
    @Binding var draft: RenderDraft

    var body: some View {
        LabeledSection("Fit") {
            VStack(alignment: .leading, spacing: 3) {
                Picker("Fit", selection: mode) {
                    ForEach(modes, id: \.self) { Text($0.label).tag($0) }
                }
                .labelsHidden()
                .fixedSize()
                Text(draft.media.sourceFit.mode.help)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .rowActionMenu(
            GenerateMenus.sourceFit(isAtDefault: draft.media.sourceFit == .default),
            perform: perform)
    }

    private func perform(_ action: GenerateAction) {
        guard action == .resetSourceFit else { return }
        draft.media.sourceFit = .default
    }

    /// Port of `sourceFitPolicyForMode` (`sourceFit.ts:124-142`): a compact
    /// mode control builds the COMPLETE policy, centred, and `pad-repaint`
    /// degrades to a centred crop where the recipe has no mask to repaint
    /// through.
    private var mode: Binding<SourceFitMode> {
        Binding(
            get: { draft.media.sourceFit.mode },
            set: { picked in
                draft.media.sourceFit = Self.policy(
                    for: picked, supportsMask: modes.contains(.padRepaint))
            })
    }

    static func policy(for mode: SourceFitMode, supportsMask: Bool) -> SourceFit {
        switch mode {
        case .cropFill: .default
        case .padRepaint: supportsMask ? .padRepaint : .default
        case .padFit: .padFit
        case .lanczosResize: .lanczosResize
        // Never authored here -- see `resolve` below, which does not offer it.
        case .upscaleThenFit: .default
        }
    }
}

extension SourceFitRow {
    /// The modes this recipe may offer, or an empty list for no row at all.
    ///
    /// `pad-repaint` is dropped where the recipe has no mask path: it would
    /// paint bands the model can never repaint, which is exactly what
    /// `coerceSourceFitForMaskless` exists to prevent (`sourceFit.ts:261-280`).
    /// `upscale-then-fit` is not offered because this app has no client-side
    /// upscale to run first -- the policy still round-trips, so a print made
    /// elsewhere keeps its provenance rather than being rewritten.
    static func resolve(
        recipe: GenerationRecipe?, media: DraftMedia
    ) -> [SourceFitMode] {
        guard let recipe, media.requestConditioning.carriesSource else { return [] }
        // A recipe whose canvas comes FROM the source has nothing to fit onto.
        guard recipe.resolution.domain != .sourceDriven, recipe.resolution.hasCanvas else {
            return []
        }
        let maskless: [SourceFitMode] = [.cropFill, .padFit, .lanczosResize]
        guard RefineGroup.maskCapable(recipe.capabilities) else { return maskless }
        return [.padRepaint] + maskless
    }
}
