import MoldClient
import SwiftUI

/// What the fleet can see that is not in it yet.
///
/// The same answer a machine's own page gives (`PeerSection.swift`), asked of
/// every machine that can browse at all and merged: a peer is a machine on
/// THAT machine's local network, so workstation sees a different room from this Mac,
/// and the overview is where both rooms belong. `PeerAction` is the one
/// decision about what a found machine is worth offering -- asked here too
/// rather than re-derived, so a peer already in the list is silent on both
/// surfaces.
struct NearbyMachines: View {
    let hosts: HostStore
    let machines: MachineStore
    /// The peer an "Add…" sheet is open for. The peer itself rather than a
    /// name/address pair, so `action(for:)` is asked again once the sheet is
    /// actually presented instead of carrying a second, stale answer.
    @State private var addingPeer: DiscoveryPeer?

    var body: some View {
        // Absent on a fleet where nothing can browse: a "Nearby" heading over
        // a permanent blank is a promise the app cannot keep.
        if !browsers.isEmpty {
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text("Nearby").font(.headline)
                    Spacer()
                    Button("Refresh") { Task { await refresh() } }
                        .buttonStyle(.borderless)
                }
                if offered.isEmpty {
                    Text("Nothing new on the networks these machines can see.")
                        .font(.caption).foregroundStyle(.secondary)
                } else {
                    ForEach(offered, id: \.id) { row($0) }
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .task { await refresh() }
            .sheet(item: $addingPeer) { peer in
                if case let .edit(name, address) = action(for: peer) {
                    HostEditor(adding: name, at: address) { name, url, key in
                        hosts.add(name: name, url: url, apiKey: key)
                    }
                }
            }
        }
    }

    /// The machines that can browse at all -- `canBrowsePeers` is what says
    /// so, and a machine that does not advertise it is never asked.
    private var browsers: [MoldHost] {
        hosts.hosts.filter { hosts.capabilities(of: $0)?.canBrowsePeers == true }
    }

    /// One fetch per browsing machine, not a poll: each server keeps its own
    /// DNS-SD browse cache, so asking again is the whole cost of "did anything
    /// new show up".
    private func refresh() async {
        for host in browsers {
            await machines.refreshPeers(on: host.id)
        }
    }

    /// Peers no machine in the list already is, deduplicated by address --
    /// two machines seeing the same third one is one row, not two.
    private var offered: [DiscoveryPeer] {
        var seen: Set<String> = []
        var rows: [DiscoveryPeer] = []
        for host in browsers {
            for peer in machines.peers[host.id] ?? [] where action(for: peer) != .skip {
                let key = HostAddress.normalize(peer.url)?.absoluteString ?? peer.url
                guard seen.insert(key).inserted else { continue }
                rows.append(peer)
            }
        }
        return rows
    }

    private func action(for peer: DiscoveryPeer) -> PeerAction {
        PeerAction.resolve(
            peer,
            known: { hosts.host(at: $0) != nil },
            knownInstance: { id in hosts.hosts.contains { hosts.instanceID(of: $0.id) == id } }
        )
    }

    private func row(_ peer: DiscoveryPeer) -> some View {
        HStack(spacing: 10) {
            VStack(alignment: .leading, spacing: 2) {
                Text(peer.name)
                Text(peer.url).font(.caption).foregroundStyle(.secondary)
            }
            Spacer(minLength: 8)
            if let item = menu(for: peer).first {
                Button(item.title) { accept(peer) }
            }
        }
        .contentShape(.rect)
        // The row's one button a second way -- the SAME `action(for:)`, so the
        // ellipsis (a sheet for a peer that wants a key) is drawn in both
        // places or neither.
        .rowActionMenu(menu(for: peer)) { _ in accept(peer) }
    }

    /// `PeerAction`'s own answer, the same one a machine's page draws
    /// (`PeerSection.swift`).
    private func menu(for peer: DiscoveryPeer) -> [RowAction<String>] {
        action(for: peer).offered(for: peer)
    }

    private func accept(_ peer: DiscoveryPeer) {
        switch action(for: peer) {
        case let .add(name, url): hosts.add(name: name, url: url, apiKey: nil)
        case .edit: addingPeer = peer
        case .skip: break
        }
    }
}
