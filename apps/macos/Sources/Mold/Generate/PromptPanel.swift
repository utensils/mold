import MoldClient
import MoldStyle
import SwiftUI

/// The floating panel: what to make, how to make it, and the button.
struct PromptPanel: View {
    /// The capsule's ceiling. `PromptLip` matches it so the lip sits flush
    /// under the capsule it came from.
    static let maxWidth: CGFloat = 760

    let recipe: GenerationRecipe?
    @Binding var draft: RenderDraft
    let model: Model?
    let host: MoldHost?
    @Binding var destination: Destination
    let submit: () -> Void
    let cancel: () -> Void
    /// Stops every batch this pane has admitted, not just the one on screen
    /// (M8 decision 8). Neither this nor `cancel` takes a backend any more --
    /// each batch resolves its own machine.
    let stopAll: () -> Void
    let maxBatch: Int
    /// This machine's own chain limits for the chosen model.
    let chainLimits: ChainLimits?

    /// Not `private`: `PromptPanel+Actions`, an extension in another file,
    /// reads the run and the queue depth to build the trailing button group.
    @Environment(GenerateController.self) var controller
    @FocusState private var promptFocused: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            if let steps = controller.run.steps {
                StepSegments(done: steps.done, total: steps.total)
            }
            if let recipe {
                prompt(recipe)
                promptTools(recipe)
                Divider()
                ControlsRow(recipe: recipe, model: model, maxBatch: maxBatch,
                            chainLimits: chainLimits, draft: $draft)
                actions(recipe)
            } else {
                Text("Pick a model to see its controls.")
                    .foregroundStyle(.secondary)
            }
        }
        .padding(16)
        .panel(.floating)
        .frame(maxWidth: Self.maxWidth)
        // Published the same way the Library's title and tag fields already
        // do, so `ResultStrip`'s arrow-key shortcuts stand down for a caret
        // here exactly as they do for one there.
        .focusedValue(\.editingText, promptFocused ? true : nil)
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
                            // Shares the one focus state with the prompt
                            // field above: a bare `Bool` binding answers
                            // "is either of these two typing", which is
                            // exactly what a caret's claim on an arrow key
                            // needs.
                            .focused($promptFocused)
                    }
                }
                ImageConditioningWells(recipe: recipe, model: model, draft: $draft)
            }
        }
    }

    /// The wand's split button and, once a rewrite has been accepted, the
    /// way back out of it -- under the prompt rather than a glyph pinned to
    /// its corner (M8 decision 4).
    @ViewBuilder private func promptTools(_ recipe: GenerationRecipe) -> some View {
        if recipe.capabilities.promptRequirement != .ignored, let host {
            HStack(spacing: 10) {
                PromptWand(recipe: recipe, host: host, draft: $draft, destination: $destination)
                if controller.canRevertExpansion {
                    Button("\(undoLabel) · Undo") { controller.revertExpansion() }
                        .buttonStyle(.plain)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
            }
        }
    }

    /// A clip model is not making "a picture", and saying so is the cheapest
    /// way to tell someone what they are about to get.
    private func placeholder(_ recipe: GenerationRecipe) -> String {
        recipe.temporal == nil ? "Describe a picture…" : "Describe a clip…"
    }

    /// Absence of `sourceImage` means YES -- raw `sourceImage?.isSupported` had it backwards.
    ///
    /// The recipe's own permission only; whether the well is DRAWN is
    /// `ImageConditioningWells.layout`'s decision, which folds this together
    /// with the reference relation.
    static func showsSourceWell(for recipe: GenerationRecipe) -> Bool { recipe.capabilities.readsSourceImage }

    /// What `canRevertExpansion`'s affordance says was just done to the
    /// prompt -- the operation the accepted choice actually carried out.
    private var undoLabel: String {
        draft.promptTransform?.operation == .remix ? "remixed" : "expanded"
    }
}
