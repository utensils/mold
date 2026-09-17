import MoldClient
import SwiftUI

// The machine picker and the downloads button. Split from the pane purely
// for size, the same reason `LibraryPane+Toolbar.swift` is its own file.
extension ModelsPane {
    @ToolbarContentBuilder var toolbar: some ToolbarContent {
        // Drawn only where there are two scopes to switch between -- never
        // a segmented control with one segment (decision 8, M5).
        if availableScopes.count > 1 {
            ToolbarItem {
                Picker("Scope", selection: scope) {
                    ForEach(availableScopes, id: \.self) { Text($0.label).tag($0) }
                }
                .pickerStyle(.segmented)
            }
        }
        ToolbarItem {
            Picker("Machine", selection: selectedHostID) {
                ForEach(hosts.hosts) { host in
                    Text(host.name).tag(MoldHost.ID?.some(host.id))
                }
            }
        }
        if let host {
            ToolbarItem { DownloadsButton(host: host) }
        }
    }

    /// `selectedMachine` read and written the way `HostStore.machine(selected:)`
    /// expects: a `Binding<MoldHost.ID?>` over the stored `uuidString`, shared
    /// with the sidebar and `MachinesPane` rather than a picker of its own.
    var selectedHostID: Binding<MoldHost.ID?> {
        Binding(
            get: { hosts.machine(selected: selectedMachine)?.id },
            set: { selectedMachine = $0?.uuidString ?? "" }
        )
    }
}
