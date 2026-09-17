import MoldClient
import SwiftUI

/// Settings ▸ Generation -- render defaults every model starts from unless
/// its own "Use as default for this model" (M3) overrides them, plus what
/// happens to a saved picture. Curated from `SettingKeys.generationRendering`
/// / `.generationSaving` (`SettingKeys+Generation.swift`), pinned to the
/// engine by `SettingKeysContractTests`.
struct GenerationSettings: View {
    @Environment(HostStore.self) private var hosts
    @Environment(ConfigStore.self) private var store
    @AppStorage("selectedMachine", store: AppStorageSuite.defaults) private var selectedMachine = ""

    private var machine: MoldHost? { hosts.machine(selected: selectedMachine) }

    var body: some View {
        SettingsPaneBody(hosts: hosts, store: store, machine: machine) { machine in
            Form {
                SettingsMachineHeader(hosts: hosts, selectedMachine: $selectedMachine)
                Section("Rendering") {
                    SettingRowList(settings: SettingKeys.generationRendering, machine: machine)
                }
                Section("When a picture is saved") {
                    SettingRowList(settings: SettingKeys.generationSaving, machine: machine)
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
