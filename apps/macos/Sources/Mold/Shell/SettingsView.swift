import MoldClient
import SwiftUI

/// The Settings scene (⌘,).
struct SettingsView: View {
    var body: some View {
        TabView {
            MachinesSettings()
                .tabItem { Label("Machines", systemImage: "server.rack") }
        }
        .frame(width: 560, height: 380)
    }
}

/// Adding, editing and removing the servers the app talks to.
struct MachinesSettings: View {
    @Environment(HostStore.self) private var hosts
    @State private var selection: MoldHost.ID?
    @State private var isAdding = false

    var body: some View {
        VStack(spacing: 0) {
            List(selection: $selection) {
                ForEach(hosts.hosts) { host in
                    HostSettingsRow(host: host, reachability: hosts.reachability(of: host))
                        .tag(host.id)
                }
            }
            Divider()
            HStack(spacing: 8) {
                Button { isAdding = true } label: { Image(systemName: "plus") }
                    .help("Add a machine")
                Button { removeSelected() } label: { Image(systemName: "minus") }
                    .disabled(selection == nil)
                    .help("Remove the selected machine")
                Spacer()
                Button("Check all") { Task { await hosts.refreshAll() } }
            }
            .buttonStyle(.borderless)
            .padding(8)
        }
        .sheet(isPresented: $isAdding) { HostEditor { hosts.add(name: $0, url: $1, apiKey: $2) } }
        .sheet(item: editing) { host in
            HostEditor(host: host) { name, url, key in
                hosts.update(MoldHost(id: host.id, name: name, baseURL: url, apiKey: key))
            }
        }
        .task { await hosts.refreshAll() }
    }

    /// Double-click opens an editor for that row.
    @State private var editingID: MoldHost.ID?

    private var editing: Binding<MoldHost?> {
        Binding(
            get: { hosts.hosts.first { $0.id == editingID } },
            set: { editingID = $0?.id }
        )
    }

    private func removeSelected() {
        guard let host = hosts.hosts.first(where: { $0.id == selection }) else { return }
        hosts.remove(host)
        selection = nil
    }
}
