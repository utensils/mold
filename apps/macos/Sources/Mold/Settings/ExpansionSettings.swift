import MoldClient
import SwiftUI

/// Settings ▸ Expansion -- the eight `expand.*` keys
/// (`SettingKeys+Expansion.swift`), pinned to the engine by
/// `SettingKeysContractTests`. The footer names the expander this machine
/// would use locally when it has not been pulled yet -- read straight off
/// `Capabilities.expanderModelToPull` (the M3 gate's own answer), never
/// re-derived here.
struct ExpansionSettings: View {
    @Environment(HostStore.self) private var hosts
    @Environment(ConfigStore.self) private var store
    @AppStorage("selectedMachine", store: AppStorageSuite.defaults) private var selectedMachine = ""

    private var machine: MoldHost? { hosts.machine(selected: selectedMachine) }

    var body: some View {
        SettingsPaneBody(hosts: hosts, store: store, machine: machine) { machine in
            Form {
                SettingsMachineHeader(hosts: hosts, selectedMachine: $selectedMachine)
                Section {
                    SettingRowList(settings: SettingKeys.expansion, machine: machine)
                } footer: {
                    if let model = hosts.capabilities(of: machine)?.expanderModelToPull {
                        Text("""
                             \(machine.name) would expand prompts locally with \(model), \
                             but hasn't pulled it yet.
                             """)
                        .foregroundStyle(.secondary)
                    }
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
