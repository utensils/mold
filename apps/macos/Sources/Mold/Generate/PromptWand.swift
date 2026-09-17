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
    @Environment(HostStore.self) private var hosts
    @State private var optionHeld = false

    var body: some View {
        let offer = controller.expansionOffer(for: recipe, on: host)
        let visibility = Visibility.resolve(
            offer: offer, promptMode: recipe.capabilities.promptRequirement, prompt: draft.prompt)

        Group {
            if isWorking {
                ProgressView().controlSize(.small)
            } else if case .hidden = visibility {
                EmptyView()
            } else {
                button(offer: offer, visibility: visibility)
            }
        }
        .onModifierKeysChanged(mask: .option, initial: false) { _, new in
            optionHeld = new.contains(.option)
        }
        .popover(isPresented: showsPopover) {
            PromptWandPopover(expansion: controller.expansion, host: host, destination: $destination)
        }
    }

    private var isWorking: Bool {
        if case .working = controller.expansion { return true }
        return false
    }

    private func button(offer: ExpansionOffer, visibility: Visibility) -> some View {
        Button { tap(offer: offer, visibility: visibility) } label: {
            Image(systemName: optionHeld ? "wand.and.rays" : "wand.and.sparkles")
        }
        .buttonStyle(.borderless)
        .controlSize(.small)
        .opacity(visibility.isReady ? 1 : 0.4)
        .help(help(for: visibility))
        // A symbol-only button has no name of its own; VoiceOver would read
        // the glyph's identifier. The help text is the name.
        .accessibilityLabel(help(for: visibility))
    }

    private var showsPopover: Binding<Bool> {
        Binding(
            get: {
                switch controller.expansion {
                case .offering, .advised, .refused, .needsModel: true
                case .idle, .working: false
                }
            },
            set: { if !$0 { controller.dismissExpansion() } }
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
            Task { await controller.expand(on: host, backend: hosts.backend(for: host)) }
        case let .ready(canRemix):
            Task {
                if Self.wantsRemix(optionHeld: optionHeld, canRemix: canRemix) {
                    await controller.remix(on: host, backend: hosts.backend(for: host))
                } else {
                    await controller.expand(on: host, backend: hosts.backend(for: host))
                }
            }
        }
    }
}

extension PromptWand {
    /// What the button shows, decided once from the same three questions a
    /// view would otherwise ask itself: whether there is a wand at all,
    /// whether the machine can do anything with a click right now, and
    /// whether that click could also mean a remix.
    enum Visibility: Equatable {
        case hidden
        case disabled(reason: String)
        case ready(canRemix: Bool)

        var isReady: Bool { if case .ready = self { true } else { false } }

        var canRemix: Bool {
            if case let .ready(canRemix) = self { return canRemix }
            return false
        }

        static func resolve(offer: ExpansionOffer, promptMode: PromptRequirement, prompt: String) -> Visibility {
            guard promptMode != .ignored else { return .hidden }
            switch offer {
            case .hidden:
                return .hidden
            case let .needsModel(model):
                return .disabled(reason: "Pull \(model) to expand prompts on this machine.")
            case let .wand(canRemix):
                guard !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                    return .disabled(reason: "Write a prompt to expand it.")
                }
                return .ready(canRemix: canRemix)
            }
        }
    }

    /// Whether a click, with ⌥ held, asks for a remix rather than an expand --
    /// only where the machine offers one at all. Holding ⌥ on a host with no
    /// remix still expands, it just never says "remix" while doing it.
    static func wantsRemix(optionHeld: Bool, canRemix: Bool) -> Bool {
        optionHeld && canRemix
    }
}
