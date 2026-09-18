import MoldClient
import SwiftUI

/// The capsule's last row (M8 decision 10): what the placement preview says
/// at the leading edge, then how many more renders are waiting, Stop while
/// one is busy, and Generate at the trailing edge -- a sheet's own button
/// row. Generate never turns into Stop (M8 decision 8), a second press just
/// admits another batch; and it never MOVES either: it is the trailing item
/// of a row that is always there, so Stop appearing, the hint changing or a
/// control wrapping shifts nothing under the pointer.
extension PromptPanel {
    /// Whether Stop is a plain button or a split menu over "Stop All
    /// Queued" -- pure, so the capsule's busiest row is tested without a
    /// view.
    enum StopControl: Equatable { case button, menu }

    static func stopControl(queued: Int) -> StopControl {
        queued > 0 ? .menu : .button
    }

    func actions(_ recipe: GenerationRecipe) -> some View {
        HStack(spacing: 10) {
            // workstation's own "infeasible" answer names every GPU and runs to
            // hundreds of characters: flexible and truncating, so it takes
            // whatever the buttons leave and never widens the capsule (the
            // buttons are `fixedSize`, so they are never the ones squeezed).
            PlacementHint(placement: controller.probe.placement, error: controller.probe.error)
                .lineLimit(1)
                .truncationMode(.tail)
                .frame(maxWidth: .infinity, alignment: .leading)
            if controller.queuedCount > 0 {
                Text("\(controller.queuedCount) more queued")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            if controller.chain.active?.isPaused == true {
                // A host restart PARKS an ephemeral chain rather than losing
                // it: its manifest, its finished clips and its tail cache are
                // all still there, so this continues rather than re-renders.
                Button("Resume") {
                    controller.chain.resume(backend: { controller.hosts.backend(for: $0) })
                }
                .controlSize(.large)
                .help("Continue this clip where the machine parked it")
            }
            if controller.run.isBusy {
                stopButton
            }
            Button(action: submit) {
                HStack(spacing: 6) {
                    Text("Generate")
                    Text("⌘↩").foregroundStyle(.secondary)
                }
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
            .keyboardShortcut(.return, modifiers: .command)
            .disabled(submitRefusal != nil)
            .help(controller.run.isBusy
                  ? "Queue another render"
                  : (submitRefusal ?? "Render this"))
            .fixedSize()
        }
    }

    /// Why Generate is not offered. The draft's own refusal, and -- for a
    /// clip past what this model can chain -- the ROUTING's, which used to be
    /// computed for the caption under the slider and then only discovered
    /// after the press, as a failure on the canvas.
    private var submitRefusal: String? {
        guard let recipe else { return nil }
        if let refusal = draft.refusal(for: recipe) { return refusal }
        return ClipRouting.resolve(recipe: recipe, model: model, draft: draft,
                                   limits: chainLimits)?.refusal
    }

    @ViewBuilder private var stopButton: some View {
        switch Self.stopControl(queued: controller.queuedCount) {
        case .button:
            Button("Stop", role: .destructive, action: cancel)
                .controlSize(.large)
                .fixedSize()
        case .menu:
            Menu {
                Button("Stop All Queued", role: .destructive, action: stopAll)
            } label: {
                Text("Stop")
            } primaryAction: {
                cancel()
            }
            .controlSize(.large)
            .fixedSize()
        }
    }
}
