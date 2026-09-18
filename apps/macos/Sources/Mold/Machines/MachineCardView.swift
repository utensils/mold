import MoldClient
import MoldStyle
import SwiftUI

/// One machine on the fleet overview.
///
/// A renderer and nothing else: every word, every figure and every absence is
/// `MachineCard`'s answer (`MachineCard.swift`), so what a card says can be
/// tested without a window. A row whose figure is `nil` is not drawn at all --
/// a card is read at a glance, and a column of em dashes is a column of
/// nothing.
struct MachineCardView: View {
    let card: MachineCard
    let perform: (MachineCardActions.Kind) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            heading
            if let status = card.status {
                Text(status).font(.callout).foregroundStyle(.secondary).lineLimit(1)
            }
            if let explanation = card.explanation {
                Text(explanation).font(.caption).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if let gpus = card.gpus { figure(gpus, trailing: card.gpuLoad) }
            if let memory = card.gpuMemory { bar(memory) }
            if let memory = card.systemMemory { bar(memory) }
            if let work = card.work { figure(work) }
            if let models = card.models { figure(models) }
            Text(card.address)
                .font(.caption).foregroundStyle(.tertiary).lineLimit(1)
                .padding(.top, 2)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .panel(.inset)
        // Dimmed, and still in its place: moving a machine because it stopped
        // answering is how you lose the one you were looking for.
        .opacity(card.isDimmed ? 0.6 : 1)
        .contentShape(.rect)
        .onTapGesture { perform(.open) }
        .rowActionMenu(MachineCardActions.offered(isThisMac: card.isThisMac,
                                                  isDefault: card.isDefault),
                       perform: perform)
        .accessibilityElement(children: .combine)
        .accessibilityLabel(card.name)
        .accessibilityAddTraits(.isButton)
    }

    private var heading: some View {
        HStack(spacing: 8) {
            HostStatusDot(reachability: card.reachability)
            Text(card.name).font(.headline).lineLimit(1)
            // This Mac is marked because it behaves differently -- it cannot
            // be edited, removed or paired with -- not as decoration.
            if card.isThisMac { badge("This Mac") }
            if card.isDefault { badge("Default") }
            Spacer(minLength: 0)
        }
    }

    private func badge(_ text: String) -> some View {
        Text(text)
            .font(.caption2)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(.quaternary, in: Capsule())
    }

    /// A line of the card: a sentence, and optionally one figure at the
    /// trailing edge.
    private func figure(_ text: String, trailing: String? = nil) -> some View {
        HStack(spacing: 10) {
            Text(text).font(.caption).lineLimit(1)
            Spacer(minLength: 8)
            if let trailing {
                Text(trailing).font(.caption).monospacedDigit().foregroundStyle(.secondary)
            }
        }
    }

    private func bar(_ memory: MachineCard.MemoryFigure) -> some View {
        HStack(spacing: 10) {
            Text(memory.text).font(.caption).foregroundStyle(.secondary).lineLimit(1)
            Spacer(minLength: 8)
            MemoryBar(reading: memory.reading)
        }
    }
}
