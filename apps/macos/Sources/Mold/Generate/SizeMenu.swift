import MoldClient
import SwiftUI

// `SizeMenu` (one buried "1024 × 1024" menu) is retired -- `ShapeControl.swift`
// replaces it with aspect + size, two menus (M8 decision 3). This file keeps
// its name; only `SeedControl` below still lives here.

/// The seed, and whether it is held still between renders.
struct SeedControl: View {
    @Binding var draft: RenderDraft

    var body: some View {
        HStack(spacing: 6) {
            Text(draft.seed.map(String.init) ?? "Random")
                .monospacedDigit()
                .lineLimit(1)
                // A seed is up to ten digits; without a floor the label
                // collapses to "R..." as soon as the row is tight.
                .frame(minWidth: 62, alignment: .leading)
                .foregroundStyle(draft.locksSeed ? .primary : .secondary)
            Button {
                draft.seed = UInt64.random(in: 0...UInt64(UInt32.max))
                draft.locksSeed = true
            } label: {
                Image(systemName: "shuffle")
            }
            .buttonStyle(.accessoryBar)
            .help("Pick a new seed")
            Toggle(isOn: $draft.locksSeed) {
                Image(systemName: draft.locksSeed ? "lock" : "lock.open")
            }
            .toggleStyle(.button)
            .buttonStyle(.accessoryBar)
            .help(draft.locksSeed ? "Reusing this seed" : "A new seed each render")
        }
    }
}
