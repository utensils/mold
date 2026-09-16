import MoldClient
import MoldStyle
import SwiftUI

/// The floating panel: what to make, how to make it, and the button.
struct PromptPanel: View {
    let recipe: GenerationRecipe?
    @Binding var draft: RenderDraft
    let model: Model?

    @Environment(GenerateController.self) private var controller
    @FocusState private var promptFocused: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            if let recipe {
                prompt(recipe)
                Divider()
                HStack(alignment: .bottom, spacing: 12) {
                    ControlsRow(recipe: recipe, draft: $draft)
                    actions(recipe)
                }
            } else {
                Text("Pick a model to see its controls.")
                    .foregroundStyle(.secondary)
            }
        }
        .padding(16)
        .panel(.floating)
        .frame(maxWidth: 760)
    }

    @ViewBuilder private func prompt(_ recipe: GenerationRecipe) -> some View {
        switch recipe.capabilities.promptRequirement {
        case .ignored:
            // No text encoder anywhere in this family. A prompt box here would
            // be furniture -- the recipe's own words say why.
            Text(recipe.capabilities.prompt?.reason ?? "This model doesn't read a prompt.")
                .font(.callout)
                .foregroundStyle(.secondary)
        default:
            TextField("Describe a picture…", text: $draft.prompt, axis: .vertical)
                .textFieldStyle(.plain)
                .font(.body)
                .lineLimit(2...6)
                .focused($promptFocused)
        }
    }

    private func actions(_ recipe: GenerationRecipe) -> some View {
        HStack(spacing: 10) {
            PlacementHint(
                placement: controller.placement,
                error: controller.placementError
            )
            Button {
                // Submission lands in the next milestone; the request is
                // already built and validated against the host.
            } label: {
                HStack(spacing: 6) {
                    Text("Generate")
                    Text("⌘↩").foregroundStyle(.secondary)
                }
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
            .disabled(draft.refusal(for: recipe) != nil)
            .help(draft.refusal(for: recipe) ?? "Render this")
        }
        .fixedSize()
    }
}

/// What the host says about a render before it is asked for.
struct PlacementHint: View {
    let placement: PlacementPreview?
    let error: String?

    var body: some View {
        Group {
            if let error {
                Label(error, systemImage: "exclamationmark.triangle")
                    .lineLimit(1)
            } else if let candidate = placement?.candidate,
                      let duration = candidate.predictedDuration {
                // A low-confidence estimate is stated as approximate. Showing
                // a guess as a measurement is how a progress bar starts lying.
                Label(
                    "about \(duration.formatted(.units(allowed: [.minutes, .seconds])))",
                    systemImage: candidate.setupKind == "cold" ? "snowflake" : "bolt"
                )
                .help(candidate.estimateConfidence == "low"
                      ? "A rough estimate — this model hasn't run here recently."
                      : "Estimated from recent runs on this machine.")
            } else if let reason = placement?.reason {
                Label(reason, systemImage: "exclamationmark.triangle").lineLimit(1)
            }
        }
        .font(.caption)
        .foregroundStyle(.secondary)
    }
}
