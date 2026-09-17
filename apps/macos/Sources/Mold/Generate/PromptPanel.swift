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
                        .overlay(alignment: .bottomTrailing) { wand(recipe) }
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
                    if controller.canRevertExpansion {
                        Button("\(undoLabel) · Undo") { controller.revertExpansion() }
                            .buttonStyle(.plain)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
                if let references = recipe.capabilities.referenceImages,
                   references.mode.isVisible {
                    ReferenceStrip(capability: references, draft: $draft)
                } else if Self.showsSourceWell(for: recipe) {
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

    /// Absence of `sourceImage` means YES -- raw `sourceImage?.isSupported` had it backwards.
    static func showsSourceWell(for recipe: GenerationRecipe) -> Bool { recipe.capabilities.readsSourceImage }

    @ViewBuilder private func wand(_ recipe: GenerationRecipe) -> some View {
        if let host {
            PromptWand(recipe: recipe, host: host, draft: $draft, destination: $destination)
                .padding(6)
        }
    }

    /// What `canRevertExpansion`'s affordance says was just done to the
    /// prompt -- the operation the accepted choice actually carried out.
    private var undoLabel: String {
        draft.promptTransform?.operation == .remix ? "remixed" : "expanded"
    }

    private func actions(_ recipe: GenerationRecipe) -> some View {
        HStack(spacing: 10) {
            // The flexible member: plato's own "infeasible" answer names
            // every GPU and runs to hundreds of characters. Letting THIS
            // absorb the row's width (and truncate) is what keeps the
            // capsule -- and the window's minimum width behind it -- from
            // being dragged past `Self.maxWidth` and off the screen.
            PlacementHint(placement: controller.placement, error: controller.placementError)
                .frame(maxWidth: .infinity, alignment: .trailing)
            if controller.run.isBusy {
                Button("Stop", role: .destructive, action: cancel)
                    .controlSize(.large)
                    .fixedSize()
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
                .fixedSize()
            }
        }
    }
}
