import MoldClient
import SwiftUI

/// Make randomness an explicit choice; only show a number when it will be used.
struct SeedControl: View {
    @Binding var draft: RenderDraft

    var body: some View {
        if Mode.resolve(draft) == .fixed, let seed = draft.seed {
            ViewThatFits(in: .horizontal) {
                HStack(spacing: 8) {
                    modePicker
                    seedNumber(seed)
                    newSeedButton
                }
                VStack(alignment: .leading, spacing: 6) {
                    HStack(spacing: 8) {
                        modePicker
                        newSeedButton
                    }
                    seedNumber(seed)
                }
            }
        } else {
            modePicker
        }
    }

    private var modePicker: some View {
        Picker("Seed mode", selection: Binding(
            get: { Mode.resolve(draft) },
            set: { Self.select($0, in: &draft) }
        )) {
            Text("Random each time").tag(Mode.random)
            Text("Keep this seed").tag(Mode.fixed)
        }
        .labelsHidden()
        .pickerStyle(.menu)
        .fixedSize()
        .help("Random each time makes a new variation. Keep this seed reuses the starting number to help repeat a result with the same model and settings. In a batch, each result uses the next seed number.")
    }

    private func seedNumber(_ seed: UInt64) -> some View {
        Text(String(seed))
            .monospacedDigit()
            .fixedSize()
            .textSelection(.enabled)
            .accessibilityLabel("Starting seed \(seed)")
            .help("The starting seed for the next render. A batch uses this number, then adds one for each additional result.")
    }

    private var newSeedButton: some View {
        Button("New seed") { Self.pickNewSeed(in: &draft) }
            .buttonStyle(.accessoryBar)
            .fixedSize()
            .help("Choose a different random seed and keep it for future renders. This does not start a render.")
    }

    enum Mode: Hashable {
        case random, fixed

        static func resolve(_ draft: RenderDraft) -> Self {
            draft.locksSeed && draft.seed != nil ? .fixed : .random
        }
    }

    static func select(_ mode: Mode, in draft: inout RenderDraft,
                       newSeed: @autoclosure () -> UInt64 = UInt64.random(in: 0...UInt64(UInt32.max))) {
        draft.locksSeed = mode == .fixed
        if mode == .fixed, draft.seed == nil { draft.seed = newSeed() }
    }

    static func pickNewSeed(in draft: inout RenderDraft,
                            newSeed: @autoclosure () -> UInt64 = UInt64.random(in: 0...UInt64(UInt32.max))) {
        draft.seed = newSeed()
        draft.locksSeed = true
    }
}
