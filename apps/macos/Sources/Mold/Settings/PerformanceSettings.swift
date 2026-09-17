import MoldClient
import SwiftUI

/// Settings ▸ Performance -- the port a machine's server listens on, and the
/// scheduler's three timing knobs. Curated from `SettingKeys.performance`
/// (`SettingKeys+Performance.swift`), pinned to the engine by
/// `SettingKeysContractTests`. Each `scheduler.*` row draws its own "Needs a
/// restart" caption straight from `entry.needsRestart` -- the server's own
/// answer (`SettingRow.swift`), never a client-authored list.
struct PerformanceSettings: View {
    @Environment(HostStore.self) private var hosts
    @Environment(ConfigStore.self) private var store
    @AppStorage("selectedMachine", store: AppStorageSuite.defaults) private var selectedMachine = ""

    private var machine: MoldHost? { hosts.machine(selected: selectedMachine) }

    var body: some View {
        SettingsPaneBody(hosts: hosts, store: store, machine: machine) { machine in
            Form {
                SettingsMachineHeader(hosts: hosts, selectedMachine: $selectedMachine)
                Section("Server") {
                    SettingRowList(settings: SettingKeys.performanceServer, machine: machine)
                }
                Section("Scheduler") {
                    SettingRowList(settings: SettingKeys.performanceScheduler, machine: machine)
                }
            }
            .formStyle(.grouped)
        }
        .task(id: machine?.id) {
            guard let id = machine?.id, !store.hasLoaded(on: id) else { return }
            await store.refresh(on: id)
        }
    }
}
