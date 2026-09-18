import MoldClient
import MoldStyle
import SwiftUI

/// The prompt field's own control for rewriting itself: a click asks for
/// alternatives, an option-click asks for the same idea said differently.
///
/// The button's SHAPE never changes with what the machine can do -- only its
/// tint and its `.help()` do. A click always goes through
/// `GenerateController.expand`/`remix`, which already knows whether that means
/// a real rewrite, the family guide's own advice, or "pull this model first";
/// this view never re-derives that decision, it only decides whether to ask.
struct PromptWand: View {
    let recipe: GenerationRecipe
    let host: MoldHost
    @Binding var draft: RenderDraft
    @Binding var destination: Destination

    @Environment(GenerateController.self) private var controller
    @Environment(ExpandStore.self) private var expansions
    @Environment(HostStore.self) private var hosts
    @State private var optionHeld = false

    var body: some View {
        let offer = ExpansionOffer.resolve(recipe: recipe, capabilities: hosts.capabilities(of: host))
        let visibility = Visibility.resolve(
            offer: offer, promptMode: recipe.capabilities.promptRequirement, prompt: draft.prompt)

        Group {
            if isWorking {
                HStack(spacing: 6) {
                    ProgressView().controlSize(.small)
                    Text("Rewriting…").font(.caption).foregroundStyle(.secondary)
                }
            } else if case .hidden = visibility {
                EmptyView()
            } else {
                splitButton(offer: offer, visibility: visibility)
            }
        }
        .onModifierKeysChanged(mask: .option, initial: false) { _, new in
            optionHeld = new.contains(.option)
        }
        .popover(isPresented: showsPopover) {
            PromptWandPopover(expansion: expansions.expansion, host: host, destination: $destination)
        }
    }

    private var isWorking: Bool {
        if case .working = expansions.expansion { return true }
        return false
    }

    /// The split button under the prompt (M8 decision 4): a click asks for
    /// whatever the primary action means right now; the menu spells out both
    /// choices by name for anyone who wants to pick rather than hold ⌥.
    private func splitButton(offer: ExpansionOffer, visibility: Visibility) -> some View {
        Menu {
            Button("Rewrite This Prompt", action: expand)
            if visibility.canRemix {
                Button("Suggest Other Ways to Say This", action: remix)
            }
        } label: {
            Label(optionHeld && visibility.canRemix ? "Remix" : "Expand",
                  systemImage: optionHeld ? "wand.and.rays" : "wand.and.sparkles")
        } primaryAction: {
            tap(offer: offer, visibility: visibility)
        }
        .menuStyle(.button)
        .controlSize(.regular)
        .fixedSize()
        // The "pull this model" press must stay live to reveal itself --
        // only a truly empty prompt with nothing to reveal is disabled.
        .disabled(!visibility.isReady && !isNeedsModel(offer))
        .help(help(for: visibility))
        .accessibilityLabel(help(for: visibility))
    }

    private func isNeedsModel(_ offer: ExpansionOffer) -> Bool {
        if case .needsModel = offer { return true }
        return false
    }

    private var showsPopover: Binding<Bool> {
        Binding(
            get: {
                switch expansions.expansion {
                case .offering, .advised, .refused, .needsModel: true
                case .idle, .working: false
                }
            },
            set: { if !$0 { expansions.dismiss() } }
        )
    }

    private func help(for visibility: Visibility) -> String {
        switch visibility {
        case .hidden: ""
        case let .disabled(reason): reason
        case let .ready(canRemix):
            Self.wantsRemix(optionHeld: optionHeld, canRemix: canRemix)
                ? "Suggest other ways to say this" : "Rewrite this prompt"
        }
    }

    private func tap(offer: ExpansionOffer, visibility: Visibility) {
        switch visibility {
        case .hidden:
            return
        case .disabled:
            // Only the "pull this model" case has anything to reveal; an
            // empty prompt has nothing to explain, so pressing it is inert.
            guard case .needsModel = offer else { return }
            expand()
        case let .ready(canRemix):
            if Self.wantsRemix(optionHeld: optionHeld, canRemix: canRemix) {
                remix()
            } else {
                expand()
            }
        }
    }

    private func expand() {
        Task { await expansions.expand(controller, on: host, backend: hosts.backend(for: host)) }
    }

    private func remix() {
        Task { await expansions.remix(controller, on: host, backend: hosts.backend(for: host)) }
    }
}
