import Foundation

/// The seed control both native apps draw: `Random | Fixed`, and when Fixed an
/// editable full-range number plus New seed. Randomness is an explicit choice;
/// a remembered number is only sent while Fixed.
public enum SeedChoice {
    public enum Mode: CaseIterable, Hashable, Sendable {
        case random, fixed

        public var title: String {
            switch self {
            case .random: "Random"
            case .fixed: "Fixed"
            }
        }

        public static func resolve(_ draft: RenderDraft) -> Self {
            draft.locksSeed && draft.seed != nil ? .fixed : .random
        }
    }

    public static let help = "Random picks a new seed each render. Fixed repeats a result with the same model and settings. In a batch, each result uses the next seed number."
    public static let newSeedHelp = "Choose a different random seed and keep it for future renders. This does not start a render."

    public static func select(_ mode: Mode, in draft: inout RenderDraft,
                              newSeed: @autoclosure () -> UInt64 = randomSeed()) {
        draft.locksSeed = mode == .fixed
        if mode == .fixed, draft.seed == nil { draft.seed = newSeed() }
    }

    public static func pickNewSeed(in draft: inout RenderDraft,
                                   newSeed: @autoclosure () -> UInt64 = randomSeed()) {
        draft.seed = newSeed()
        draft.locksSeed = true
    }

    /// Exact unsigned decimal: no sign, space, separator or overflow.
    public static func parse(_ text: String) -> UInt64? {
        guard !text.isEmpty, text.utf8.allSatisfy({ $0 >= 48 && $0 <= 57 }) else { return nil }
        return UInt64(text)
    }

    /// Applies typed text. Empty or partial text keeps Fixed and the last valid
    /// seed -- it never flips the draft to Random. Returns whether it applied.
    @discardableResult
    public static func edit(_ text: String, in draft: inout RenderDraft) -> Bool {
        guard let value = parse(text) else { return false }
        draft.seed = value
        draft.locksSeed = true
        return true
    }

    public static func text(for draft: RenderDraft) -> String { draft.seed.map(String.init) ?? "" }

    public static func randomSeed() -> UInt64 { UInt64.random(in: 0...UInt64(UInt32.max)) }
}
