import MoldClient
import SwiftUI

/// What this machine can see nearby that we do not already have.
///
/// Shown only when the machine can browse at all -- `canBrowsePeers` is what
/// says so. One fetch when the machine is selected, plus a Refresh button in
/// the header, deliberately not a poll: the server keeps its own DNS-SD
/// browse cache, so asking again is the whole cost of "did anything new show
/// up".
struct PeerSection: View {
    @Environment(HostStore.self) private var hosts
    @Environment(MachineStore.self) private var machines
    let host: MoldHost
    /// The peer "Add…" is currently open for. A `DiscoveryPeer` rather than a
    /// name/address pair, so `action(for:)` can be asked again once the sheet
    /// is actually presented instead of carrying a second, stale answer.
    @State private var addingPeer: DiscoveryPeer?

    var body: some View {
        if hosts.capabilities(of: host)?.canBrowsePeers == true {
            Section {
                if offered.isEmpty {
                    Text("Nothing found on this machine's local network.")
                        .foregroundStyle(.secondary)
                } else {
                    ForEach(offered) { row($0) }
                }
            } header: {
                HStack {
                    Text("Nearby")
                    Spacer()
                    Button("Refresh") { Task { await machines.refreshPeers(on: host.id) } }
                        .buttonStyle(.borderless)
                }
            }
            .task(id: host.id) { await machines.refreshPeers(on: host.id) }
            .sheet(item: $addingPeer) { peer in
                if case let .edit(name, address) = action(for: peer) {
                    HostEditor(adding: name, at: address) { name, url, key in
                        hosts.add(name: name, url: url, apiKey: key)
                    }
                }
            }
        }
    }

    /// Peers that are neither this machine nor one we already talk to.
    /// "Already have" is asked twice inside `PeerAction.resolve` -- by origin
    /// and by fleet identity -- which is what catches the same box found
    /// again at a second address.
    private var offered: [DiscoveryPeer] {
        (machines.peers[host.id] ?? []).filter { action(for: $0) != .skip }
    }

    private func action(for peer: DiscoveryPeer) -> PeerAction {
        PeerAction.resolve(
            peer,
            known: { hosts.host(at: $0) != nil },
            knownInstance: { id in hosts.hosts.contains { hosts.instanceID(of: $0.id) == id } }
        )
    }

    @ViewBuilder private func row(_ peer: DiscoveryPeer) -> some View {
        LabeledContent {
            add(peer)
        } label: {
            VStack(alignment: .leading, spacing: 2) {
                Text(peer.name)
                Text(peer.url).font(.caption).foregroundStyle(.secondary)
            }
        }
        .contentShape(Rectangle())
        // The row's one button a second way -- the SAME `action(for:)`, so
        // the ellipsis (a sheet for a peer that wants a key) is drawn in both
        // places or neither. `offered` has already dropped every `.skip`, so
        // this is never an empty menu in practice.
        .contextMenu { add(peer) }
    }

    @ViewBuilder private func add(_ peer: DiscoveryPeer) -> some View {
        switch action(for: peer) {
        case let .add(name, url):
            Button("Add") { hosts.add(name: name, url: url, apiKey: nil) }
        case .edit:
            Button("Add…") { addingPeer = peer }
        case .skip:
            EmptyView()
        }
    }
}
