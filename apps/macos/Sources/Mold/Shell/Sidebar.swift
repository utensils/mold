import MoldClient
import SwiftUI

struct Sidebar: View {
    @Environment(HostStore.self) private var hosts
    @Binding var destination: Destination

    var body: some View {
        List(selection: $destination) {
            Section {
                ForEach(Destination.allCases) { item in
                    Label(item.title, systemImage: item.symbol).tag(item)
                }
            }

            Section("Machines") {
                ForEach(hosts.hosts) { host in
                    HostRow(host: host, reachability: hosts.reachability(of: host))
                }
            }
        }
        .listStyle(.sidebar)
        .refreshable { await hosts.refreshAll() }
    }
}

private struct HostRow: View {
    let host: MoldHost
    let reachability: HostStore.Reachability

    var body: some View {
        HStack(spacing: 8) {
            HostStatusDot(reachability: reachability)
            VStack(alignment: .leading, spacing: 1) {
                Text(host.name)
                if let detail = reachability.summary {
                    Text(detail)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }
            }
        }
        .help(HostAddress.displayString(for: host.baseURL))
    }
}
