import MoldClient
import SwiftUI

/// One card: what it is, what it is doing with itself, and the one control
/// this machine will honour -- which is `DeviceControl`'s answer and never
/// this view's guess.
struct DeviceRow: View {
    @Environment(HostStore.self) private var hosts
    @Environment(MachineStore.self) private var machines
    let device: DeviceInfo
    let host: MoldHost
    /// The live 1 Hz figures, once a snapshot has arrived. Absent before the
    /// first frame, which is what makes the row fall back to the card's own.
    let sample: GpuSample?

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 8) {
                Text(title)
                Spacer(minLength: 8)
                control
            }
            HStack(spacing: 10) {
                Text(status).font(.caption).foregroundStyle(.secondary)
                Spacer(minLength: 8)
                // No reading, no bar: a 0-of-0 bar reads as a full one.
                if let reading = MemoryReading(used: used, total: total) {
                    MemoryBar(reading: reading)
                }
            }
            ForEach(captions, id: \.self) { line in
                Text(line).font(.caption).foregroundStyle(.secondary)
            }
        }
        .padding(.vertical, 2)
    }

    /// A MIG slice has no ordinal worth printing -- it is a partition of a
    /// card, and the number would name the wrong thing.
    private var title: String {
        guard let ordinal = device.ordinal, device.deviceKind != .mig else { return device.name }
        return "\(device.name)  #\(ordinal)"
    }

    private var used: UInt64? { sample?.vramUsed ?? device.memory.usedBytes }
    private var total: UInt64? { sample.map(\.vramTotal) ?? device.memory.totalBytes }

    private var status: String {
        [DeviceWords.state(device),
         DeviceWords.utilization(sample?.gpuUtilization ?? device.telemetry.utilizationPercent),
         DeviceWords.memory(used: used, total: total,
                            mold: sample?.vramUsedByMold ?? device.memory.moldUsedBytes)]
            .compactMap(\.self).joined(separator: " · ")
    }

    /// The lines a card only sometimes has. Health is here only when it is
    /// NOT healthy, and `unschedulableReason` is the machine's own sentence,
    /// printed as written -- paraphrasing it would be paraphrasing a diagnosis.
    private var captions: [String] {
        [DeviceWords.holding(device.loadedModels),
         DeviceWords.health(device.health),
         device.unschedulableReason,
         device.adminState == .startupExcluded ? DeviceWords.startupExcluded : nil,
         device.needsRestart ? DeviceWords.needsRestart : nil]
            .compactMap(\.self)
    }

    @ViewBuilder private var control: some View {
        switch DeviceControl.resolve(device, on: hosts.capabilities(of: host),
                                     isChanging: machines.isChanging(device)) {
        case let .live(isOn, isEnabled):
            // The getter reads the store, never a local copy, so the switch
            // shows what the machine ANSWERED rather than what was asked --
            // a disable is a drain, and settling instantly into "off" would
            // be the opposite of what happened.
            //
            // An EMPTY label with an accessibility name, rather than a named
            // label under `.labelsHidden()`: hiding the label takes the
            // switch's `AXPress` action with it, leaving a control no
            // assistive technology can operate. Measured on this row.
            //
            // `set:` is spelled out rather than passed `flip` by name --
            // converting a MainActor method into a `Binding`'s setter crashes
            // swift-frontend in IRGen.
            Toggle("", isOn: Binding(get: { isOn }, set: { flip($0) }))
                .toggleStyle(.switch)
                .disabled(!isEnabled)
                .accessibilityLabel("Use \(title)")
        case .enableAtRestart:
            Button("Enable at next restart") { flip(true) }
                .buttonStyle(.link)
        case .readOnly:
            EmptyView()
        }
    }

    private func flip(_ enabled: Bool) {
        Task { await machines.setDevice(device, enabled: enabled, on: host.id) }
    }
}
