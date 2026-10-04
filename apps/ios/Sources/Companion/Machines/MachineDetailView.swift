import MoldClient
import SwiftUI

/// One machine's page (DESIGN.md §5.4): what it is, its cards with what each
/// holds and a switch where the machine will honour one, its disk, and its
/// address -- then Edit, Default and Remove.
struct MachineDetailView: View {
    @Environment(HostStore.self) private var hosts
    @Environment(\.dismiss) private var dismiss
    let id: MoldHost.ID
    @State private var devices: [DeviceInfo] = []
    @State private var editing = false
    @State private var confirmRemove = false

    var body: some View {
        if let host = hosts.host(id) {
            List {
                overview(host)
                if !devices.isEmpty { cards(host) }
                storage(host)
                Section {
                    NavigationLink(value: ModelsRoute(host: id)) {
                        AdaptiveRow { Text("Models") } value: {
                            if let count = hosts.installed[id] { Text("\(count) installed") }
                        }
                    }
                    .accessibilityIdentifier("machine-models")
                    NavigationLink(value: QueueRoute(host: id)) { Text("Queue") }
                }
                Section {
                    Button("Edit…") { editing = true }
                    Button("Set as Default") { hosts.makeDefault(id) }
                        .disabled(hosts.defaultMachine == id)
                    Button("Remove…", role: .destructive) { confirmRemove = true }
                }
            }
            .accessibilityIdentifier("machine-details")
            .navigationTitle(host.name)
            .refreshable { await reload(host) }
            .task { await reload(host) }
            .sheet(isPresented: $editing) { EditMachineSheet(host: host) }
            .confirmationDialog("Remove \(host.name)?", isPresented: $confirmRemove, titleVisibility: .visible) {
                Button("Remove", role: .destructive) { hosts.remove(id); dismiss() }
            } message: {
                Text("Its key is removed from this iPhone too. Its prints stay on the machine.")
            }
        } else {
            EmptyState(title: String(localized: "Removed"), symbol: "server.rack",
                       message: String(localized: "This machine is no longer in the list."))
        }
    }

    private func reload(_ host: MoldHost) async {
        await hosts.refresh(host)
        guard hosts.capabilities[id]?.canSeeDevices == true else { devices = []; return }
        devices = (try? await hosts.backend(for: host).devices().devices) ?? []
    }

    private func overview(_ host: MoldHost) -> some View {
        let state = hosts.reachability(of: host)
        return Section {
            Label {
                Text(hosts.silence(of: host) ?? state.sentence ?? String(localized: "Not checked yet"))
                    .fixedSize(horizontal: false, vertical: true)
            } icon: { StatusDot(reachability: state) }
            if case let .up(status) = state {
                if let depth = status.queueDepth {
                    AdaptiveRow { Text("Waiting") } value: { Text("\(depth)").monospacedDigit() }
                }
                if let memory = status.memoryStatus {
                    AdaptiveRow { Text("Memory") } value: { Text(memory) }
                }
            }
        }
    }

    private func cards(_ host: MoldHost) -> some View {
        let canSwitch = hosts.capabilities[id].map { $0.canChangeDeviceLifecycle && $0.dispatchIsAuthoritative } ?? false
        return Section("Graphics") {
            ForEach(devices) { device in
                DeviceRow(device: device, canSwitch: canSwitch) { enabled in
                    Task {
                        do {
                            _ = try await hosts.backend(for: host).setDevice(device.id, enabled: enabled)
                            await reload(host)
                        } catch {
                            hosts.report(host, doing: String(localized: "switch \(device.name)"), error)
                        }
                    }
                }
            }
        }
    }

    @ViewBuilder private func storage(_ host: MoldHost) -> some View {
        Section("Address") {
            Text(HostAddress.displayString(for: host.baseURL))
                .font(.body.monospaced())
                .contextMenu {
                    Button("Copy Address") { UIPasteboard.general.string = HostAddress.displayString(for: host.baseURL) }
                }
            if case let .up(status) = hosts.reachability(of: host), let disk = status.modelsDisk {
                AdaptiveRow { Text("Models disk") } value: {
                    Text("\(DeviceWords.bytes(disk.freeBytes)) free of \(DeviceWords.bytes(disk.totalBytes))")
                        .monospacedDigit()
                }
            }
        }
    }
}

/// One graphics card: its name and state, what it holds, its memory, and a
/// switch only where the machine will honour it.
struct DeviceRow: View {
    let device: DeviceInfo
    let canSwitch: Bool
    let toggle: (Bool) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            if canSwitch {
                Toggle(isOn: Binding(get: { device.desiredEnabled }, set: toggle)) { title }
            } else {
                title
            }
            Text(DeviceWords.memory(used: device.memory.usedBytes, total: device.memory.totalBytes,
                                    mold: device.memory.moldUsedBytes))
                .font(.subheadline).foregroundStyle(.secondaryText).monospacedDigit()
            if let total = device.memory.totalBytes, total > 0, let used = device.memory.usedBytes {
                Gauge(value: min(1, Double(used) / Double(total))) { Text("Memory") }
                    .gaugeStyle(.accessoryLinearCapacity)
            }
            if let holding = DeviceWords.holding(device.loadedModels) {
                Text(holding).font(.subheadline).foregroundStyle(.secondaryText)
            }
            if device.needsRestart {
                Text(DeviceWords.needsRestart).font(.subheadline).foregroundStyle(.orange)
            }
        }
        .padding(.vertical, 4)
    }

    private var title: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(device.name).font(.headline)
            Text([DeviceWords.state(device), DeviceWords.health(device.health),
                  DeviceWords.utilization(device.telemetry.utilizationPercent)]
                .compactMap(\.self).joined(separator: " · "))
                .font(.subheadline).foregroundStyle(.secondaryText)
        }
    }
}
