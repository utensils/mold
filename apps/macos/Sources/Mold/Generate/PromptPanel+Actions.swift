import MoldClient
import SwiftUI

/// The button cluster pinned to the controls row's trailing edge (M8
/// decision 1): what the placement preview says, how many more renders are
/// waiting, Stop while one is busy, and Generate itself -- which never turns
/// into Stop (M8 decision 8), a second press just admits another batch.
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
            // plato's own "infeasible" answer names every GPU and runs to
            // hundreds of characters. A fixed ceiling, not `.infinity`, is
            // what keeps the capsule -- and the window's minimum width
            // behind it -- from being dragged past `Self.maxWidth` and off
            // the screen; it now sits in a pinned trailing group rather than
            // claiming half the row.
            PlacementHint(placement: controller.placement, error: controller.placementError)
                .lineLimit(1)
                .truncationMode(.tail)
                .frame(maxWidth: 260, alignment: .trailing)
            if controller.queuedCount > 0 {
                Text("\(controller.queuedCount) more queued")
                    .font(.caption)
                    .foregroundStyle(.secondary)
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
            .disabled(draft.refusal(for: recipe) != nil)
            .help(controller.run.isBusy
                  ? "Queue another render"
                  : (draft.refusal(for: recipe) ?? "Render this"))
            .fixedSize()
        }
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
