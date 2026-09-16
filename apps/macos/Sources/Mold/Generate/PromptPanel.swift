import MoldClient
import MoldStyle
import SwiftUI

/// The floating panel: what to make, how to make it, and the button.
struct PromptPanel: View {
    let recipe: GenerationRecipe?
    @Binding var draft: RenderDraft
    let model: Model?
    let submit: () -> Void
    let cancel: () -> Void
    let maxBatch: Int

    @Environment(GenerateController.self) private var controller
    @FocusState private var promptFocused: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            if let steps = controller.run.steps {
                StepSegments(done: steps.done, total: steps.total)
            }
            if let recipe {
                prompt(recipe)
                Divider()
                HStack(alignment: .bottom, spacing: 12) {
                    ControlsRow(recipe: recipe, maxBatch: maxBatch, draft: $draft)
                    Spacer(minLength: 12)
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
            HStack(alignment: .top, spacing: 12) {
                VStack(alignment: .leading, spacing: 6) {
                    TextField(placeholder(recipe), text: $draft.prompt, axis: .vertical)
                        .textFieldStyle(.plain)
                        .font(.body)
                        .lineLimit(2...6)
                        .focused($promptFocused)
                    if recipe.capabilities.negativePrompt?.isAvailable == true {
                        TextField("Avoid…", text: $draft.negativePrompt, axis: .vertical)
                            .textFieldStyle(.plain)
                            .font(.callout)
                            .foregroundStyle(.secondary)
                            .lineLimit(1...3)
                    }
                }
                if recipe.capabilities.sourceImage?.isSupported == true {
                    SourceImageWell(draft: $draft)
                }
            }
        }
    }

    /// A clip model is not making "a picture", and saying so is the cheapest
    /// way to tell someone what they are about to get.
    private func placeholder(_ recipe: GenerationRecipe) -> String {
        recipe.temporal == nil ? "Describe a picture…" : "Describe a clip…"
    }

    private func actions(_ recipe: GenerationRecipe) -> some View {
        HStack(spacing: 10) {
            PlacementHint(
                placement: controller.placement,
                error: controller.placementError
            )
            if controller.run.isBusy {
                Button("Stop", role: .destructive, action: cancel)
                    .controlSize(.large)
            } else {
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
                .help(draft.refusal(for: recipe) ?? "Render this")
            }
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
