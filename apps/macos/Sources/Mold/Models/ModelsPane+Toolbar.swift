import MoldClient
import SwiftUI

// The machine picker and the downloads button. Split from the pane purely
// for size, the same reason `LibraryPane+Toolbar.swift` is its own file.
extension ModelsPane {
    @ToolbarContentBuilder var toolbar: some ToolbarContent {
        ToolbarItem {
            Picker("Machine", selection: selectedHostID) {
                ForEach(hosts.hosts) { host in
                    Text(host.name).tag(MoldHost.ID?.some(host.id))
                }
            }
        }
        // The Discover scope lands in S6 alongside the catalog browser it
        // has something to show; a one-segment picker in the meantime would
        // be a control with nothing to switch.
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
