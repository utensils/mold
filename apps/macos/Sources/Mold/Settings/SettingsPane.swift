import MoldClient
import SwiftUI

/// The three whole-pane states every machine-scoped Settings tab shares --
/// extracted here (S4a) once Generation and Expansion needed exactly what
/// Advanced already built for itself in S3, rather than a second copy of
/// the same switch in every curated pane.
enum SettingsPaneState: Equatable { case noMachines, unavailable, loading, loaded }

enum SettingsPane {
    /// Pure: which whole-pane state to draw, from the fleet and one
    /// machine's own answer -- askable with no view, the same split
    /// `DiscoverRow.resolve` uses for one cell.
    static func resolve(hosts: HostStore, store: ConfigStore, machine: MoldHost.ID?) -> SettingsPaneState {
        guard !hosts.hosts.isEmpty, let machine else { return .noMachines }
        if store.unavailable.contains(machine) { return .unavailable }
        if !store.hasLoaded(on: machine) { return .loading }
        return .loaded
    }
}

/// The machine picker every machine-scoped Settings tab shares -- the same
/// `selectedMachine` binding `AccountsSettings` builds.
struct SettingsMachineHeader: View {
    let hosts: HostStore
    @Binding var selectedMachine: String

    var body: some View {
        Picker("Machine", selection: selectedHostID) {
            ForEach(hosts.hosts) { host in
                Text(host.name).tag(MoldHost.ID?.some(host.id))
            }
        }
        .labelsHidden()
        .frame(width: 220)
    }

    private var selectedHostID: Binding<MoldHost.ID?> {
        Binding(
            get: { hosts.machine(selected: selectedMachine)?.id },
            set: { selectedMachine = $0?.uuidString ?? "" }
        )
    }
}

/// Every `SettingKey` in one section, filtered to what the machine's own
/// listing carries -- a key this build curates that an older host has never
/// heard of is not drawn (`SettingRow.resolve`). Shared by Generation and
/// Expansion so neither repeats the store lookups `SettingRow` needs.
struct SettingRowList: View {
    let settings: [SettingKey]
    let machine: MoldHost
    @Environment(ConfigStore.self) private var store

    var body: some View {
        ForEach(settings) { setting in
            if let entry = store.entry(setting.key, on: machine.id) {
                SettingRow(
                    setting: setting, entry: entry,
                    refusal: store.refusal(for: setting.key, on: machine.id)?.sentence,
                    onSet: { value in await store.set(setting.key, to: value, on: machine.id) })
            }
        }
    }
}

/// The whole-pane, no-machines and unavailable states for a curated pane --
/// shared with `AdvancedSettings.content`'s own switch so `.loaded` is the
/// only branch each pane's own file has to write.
struct SettingsPaneBody<Loaded: View>: View {
    let hosts: HostStore
    let store: ConfigStore
    let machine: MoldHost?
    @ViewBuilder let loaded: (MoldHost) -> Loaded

    var body: some View {
        switch SettingsPane.resolve(hosts: hosts, store: store, machine: machine?.id) {
        case .noMachines:
            Text("Add a machine in Settings ▸ Machines first.")
                .foregroundStyle(.secondary)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        case .unavailable:
            if let machine {
                Text("""
                     \(machine.name)'s settings database is switched off, so \
                     there is nothing here to read or change.
                     """)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .padding()
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        case .loading:
            ProgressView().frame(maxWidth: .infinity, maxHeight: .infinity)
        case .loaded:
            if let machine { loaded(machine) }
        }
    }
}
