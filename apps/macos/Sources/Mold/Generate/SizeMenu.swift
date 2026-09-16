import MoldClient
import SwiftUI

/// Sizes the recipe advertises, grouped by aspect.
struct SizeMenu: View {
    let resolution: ResolutionProfile
    @Binding var draft: RenderDraft

    var body: some View {
        Menu {
            ForEach(resolution.aspectGroups ?? []) { group in
                Section(group.label) {
                    ForEach(group.presets) { preset in
                        Button {
                            draft.width = preset.width
                            draft.height = preset.height
                        } label: {
                            Text(preset.label)
                        }
                    }
                }
            }
        } label: {
            Text("\(draft.width) × \(draft.height)").monospacedDigit()
        }
        .menuStyle(.button)
        .buttonStyle(.accessoryBar)
        .fixedSize()
    }
}

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
