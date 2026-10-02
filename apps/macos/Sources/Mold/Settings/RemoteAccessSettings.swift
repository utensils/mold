import MoldClient
import SwiftUI

/// A discoverable home for phone pairing and the address a remote client uses.
struct RemoteAccessSettings: View {
    @Environment(HostStore.self) private var hosts
    @Environment(PairingStore.self) private var pairing
    @AppStorage("selectedMachine", store: AppStorageSuite.defaults) private var selectedMachine = ""
    @State private var showingCode = false

    static func canPair(hostID: MoldHost.ID) -> Bool {
        hostID != MoldEngine.localHostID
    }

    private var selected: MoldHost? { hosts.machine(selected: selectedMachine) }

    var body: some View {
        Form {
            Section("Remote Access") {
                Text("Connect to a machine from your phone or another computer.")
                    .foregroundStyle(.secondary)
                if hosts.hosts.isEmpty {
                    Text("Add a machine in Settings ▸ Machines first.")
                } else {
                    SettingsMachineHeader(hosts: hosts, selectedMachine: $selectedMachine)
                }
            }
            if let host = selected {
                Section("Connection Address") {
                    LabeledContent("Address") {
                        Text(host.baseURL.absoluteString).textSelection(.enabled)
                        CopyButton(what: "Address", value: host.baseURL.absoluteString)
                    }
                    Text("For access outside your local network, add the machine’s public HTTPS relay address in Settings ▸ Machines, then select it here. A pairing code does not create a tunnel.")
                        .foregroundStyle(.secondary)
                    Link("Set Up Remote Access", destination: URL(string: "https://utensils.io/mold/deployment/relay")!)
                }
                if Self.canPair(hostID: host.id) {
                    Section("Phone Pairing") {
                        if showingCode {
                            PairingSheet(host: host, showsDone: false).id(host.id)
                                .frame(maxWidth: .infinity)
                            Button("Hide Code") { showingCode = false }
                        } else {
                            Button("Show Pairing Code") { showingCode = true }
                        }
                    }
                    PairingSection(host: host)
                } else {
                    Section("This Mac") {
                        Text("This Mac’s built-in engine is private and uses a loopback address. To share a machine, set up an authenticated server and relay, then add its public address in Machines.")
                            .foregroundStyle(.secondary)
                    }
                }
            }
        }
        .formStyle(.grouped)
        .onChange(of: selected?.id) { showingCode = false }
        .task(id: selected?.id) {
            guard let host = selected, Self.canPair(hostID: host.id) else { return }
            await pairing.load(on: host.id)
        }
    }
}
