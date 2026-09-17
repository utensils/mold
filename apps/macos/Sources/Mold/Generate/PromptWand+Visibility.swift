import MoldClient

/// `PromptWand`'s pure gate, split out purely for size.
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
