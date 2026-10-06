import MoldClient
import SwiftUI

/// A discoverable home for phone pairing and the address a remote client uses.
struct RemoteAccessSettings: View {
    @Environment(HostStore.self) private var hosts
    @Environment(PairingStore.self) private var pairing
    @Environment(RemotePairingStore.self) private var remote
    @AppStorage("selectedMachine", store: AppStorageSuite.defaults) private var selectedMachine = ""
    @State private var pairingHost: MoldHost?

    static func canPair(_ host: MoldHost) -> Bool {
        guard MoldEngine.isPairable(host) else { return false }
        let name = host.baseURL.host?.lowercased() ?? ""
        return !["localhost", "127.0.0.1", "::1", "[::1]"].contains(name)
            && !name.hasPrefix("127.")
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
                Button("Pair your phone") { pairingHost = selected }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                    .disabled(selected == nil || (selected?.id == MoldEngine.localHostID && remote.isStopping))
                    .accessibilityHint("Show a QR code to connect your phone to the selected machine")
            }
            if let host = selected {
                Section("Connection Address") {
                    LabeledContent("Address") {
                        Text(host.baseURL.absoluteString).textSelection(.enabled)
                        CopyButton(what: "Address", value: host.baseURL.absoluteString)
                    }
                    Text("Paired clients follow this machine across local network, Tailscale and configured HTTPS relay routes. Keep the machine connected while this app learns its addresses; no new pairing is needed when you leave the network.")
                        .foregroundStyle(.secondary)
                    if let routes = host.connectionEndpoints, !routes.isEmpty {
                        ForEach(routes, id: \.url) { route in
                            LabeledContent(route.kind == .lan ? "Local Network" : route.kind == .tailscale ? "Tailscale" : "HTTPS Relay", value: route.url)
                        }
                        if !routes.contains(where: { $0.kind == .relay }) {
                            Text("A public relay address has not been configured on this machine yet.")
                                .foregroundStyle(.secondary)
                        }
                    } else {
                        Text("Connection routes have not been learned yet. Automatic roaming requires a paired client and a newer authenticated server. Manually entered API keys use their saved address.")
                            .foregroundStyle(.secondary)
                    }
                    Link("Set Up Remote Access", destination: URL(string: "https://utensils.io/mold/deployment/relay")!)
                }
                if Self.canPair(host) {
                    PairingSection(host: host, showsPairButton: false)
                } else if host.id == MoldEngine.localHostID {
                    Section("This Mac") {
                        Text("Pair your phone to connect through Mold proxy. This Mac’s engine stays on a private local address.")
                            .foregroundStyle(.secondary)
                        if remote.canStopRemoteAccess {
                            Button("Stop Remote Access") { Task { await remote.disable() } }
                                .disabled(remote.isStopping)
                        }
                        if remote.enabled {
                            PairingSection(host: host, showsPairButton: false)
                        }
                        if case let .failed(reason) = remote.state { Text(reason).foregroundStyle(.secondary) }
                    }
                } else {
                    Section("This Mac") {
                        Text("This Mac’s built-in engine is private and uses a loopback address. To share a machine, set up an authenticated server and relay, then add its public address in Machines.")
                            .foregroundStyle(.secondary)
                    }
                }
            }
        }
        .formStyle(.grouped)
        .sheet(item: $pairingHost) { PhonePairingSheet(host: $0) }
        .onChange(of: selected?.id) { pairingHost = nil }
        .task(id: selected?.id) {
            guard let host = selected else { return }
            await pairing.load(on: host.id)
        }
    }
}
