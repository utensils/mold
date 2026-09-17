import MoldClient
import SwiftUI

/// Settings ▸ Library -- how long a trashed print or a held queue row
/// survives, and the gallery's storage-version switch. Curated from
/// `SettingKeys.library` (`SettingKeys+Library.swift`), pinned to the
/// engine by `SettingKeysContractTests`.
struct LibrarySettings: View {
    @Environment(HostStore.self) private var hosts
    @Environment(ConfigStore.self) private var store
    @Environment(LibraryStore.self) private var library
    @AppStorage("selectedMachine", store: AppStorageSuite.defaults) private var selectedMachine = ""

    private var machine: MoldHost? { hosts.machine(selected: selectedMachine) }

    var body: some View {
        SettingsPaneBody(hosts: hosts, store: store, machine: machine) { machine in
            Form {
                SettingsMachineHeader(hosts: hosts, selectedMachine: $selectedMachine)
                Section {
                    SettingRowList(settings: SettingKeys.library, machine: machine)
                } footer: {
                    // The store's own number, not a second fetch -- Library
                    // already lists this machine's trash for its own pane.
                    Text(trashCaption(for: machine))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            .formStyle(.grouped)
        }
        .task(id: machine?.id) {
            guard let id = machine?.id, !store.hasLoaded(on: id) else { return }
            await store.refresh(on: id)
        }
    }

    private func trashCaption(for machine: MoldHost) -> String {
        let count = library.trashPerHost[machine.id]?.count ?? 0
        return switch count {
        case 0: "\(machine.name)'s trash is empty."
        case 1: "\(machine.name)'s trash holds 1 print."
        default: "\(machine.name)'s trash holds \(count) prints."
        }
    }
}
