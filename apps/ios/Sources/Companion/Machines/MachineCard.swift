import MoldClient
import SwiftUI

/// One machine, the way the Mac's fleet card says it (DESIGN.md §5.4): dot,
/// name and Default; "Ready · 0.31.0" and the hardware; video memory; what is
/// waiting and installed; the address. A figure nothing reported is simply
/// not drawn. A machine that is not answering keeps its place and says why.
struct MachineCard: View {
    @Environment(HostStore.self) private var hosts
    @Environment(\.dynamicTypeSize) private var size
    let host: MoldHost

    var body: some View {
        let state = hosts.reachability(of: host)
        VStack(alignment: .leading, spacing: 8) {
            header(state)
            if case let .up(status) = state {
                figures(status)
            } else if let words = hosts.silence(of: host) ?? state.summary {
                Text(words).foregroundStyle(.secondaryText)
            }
            Text(HostAddress.displayString(for: host.baseURL))
                .font(.footnote.monospaced())
                .foregroundStyle(.secondaryText)
                .textSelection(.enabled)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(16)
        .background(.background.secondary, in: .rect(cornerRadius: 8))
        .accessibilityElement(children: .combine)
    }

    private func header(_ state: HostStore.Reachability) -> some View {
        let stacked = RowAxis.for(size) == .vertical
        let layout = stacked ? AnyLayout(VStackLayout(alignment: .leading, spacing: 4))
                             : AnyLayout(HStackLayout(alignment: .firstTextBaseline, spacing: 8))
        return layout {
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                StatusDot(reachability: state)
                Text(host.name).font(.headline)
            }
            if !stacked { Spacer(minLength: 0) }
            if hosts.defaultMachine == host.id {
                Text("Default")
                    .font(.caption.weight(.semibold))
                    .padding(.horizontal, 8)
                    .padding(.vertical, 2)
                    .background(.tint.opacity(0.15), in: .capsule)
                    .foregroundStyle(.tint)
            }
        }
    }

    @ViewBuilder private func figures(_ status: ServerStatus) -> some View {
        Text([HostStore.Reachability.up(status).summary, status.hardware]
            .compactMap(\.self).joined(separator: " · "))
        if let memory = VideoMemory(status) {
            VStack(alignment: .leading, spacing: 4) {
                AdaptiveRow {
                    Text("Video memory")
                } value: {
                    Text(memory.words).monospacedDigit()
                }
                Gauge(value: memory.fraction) { Text("Video memory") }
                    .gaugeStyle(.accessoryLinearCapacity)
                    .tint(.accentColor)
                    .accessibilityValue(memory.words)
            }
        }
        if let line = workLine(status) {
            Text(line).foregroundStyle(.secondaryText).monospacedDigit()
        }
    }

    private func workLine(_ status: ServerStatus) -> String? {
        let parts = [
            status.queueDepth.map { String(localized: "\($0) waiting") },
            hosts.installed[host.id].map { String(localized: "\($0) installed") },
        ].compactMap(\.self)
        return parts.isEmpty ? nil : parts.joined(separator: " · ")
    }
}

/// Every GPU's video memory, summed: the card's one bar. Absent when no card
/// reports both figures -- a 0-of-0 bar would read as full.
struct VideoMemory {
    let used: UInt64
    let total: UInt64

    init?(_ status: ServerStatus) {
        let cards = (status.gpus ?? []).filter { ($0.vramTotalBytes ?? 0) > 0 && $0.vramUsedBytes != nil }
        guard !cards.isEmpty else { return nil }
        used = cards.reduce(0) { $0 + ($1.vramUsedBytes ?? 0) }
        total = cards.reduce(0) { $0 + ($1.vramTotalBytes ?? 0) }
    }

    var fraction: Double { total == 0 ? 0 : min(1, Double(used) / Double(total)) }
    var words: String { "\(DeviceWords.bytes(used)) / \(DeviceWords.bytes(total))" }
}
