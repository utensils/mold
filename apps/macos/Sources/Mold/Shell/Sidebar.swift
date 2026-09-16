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
            Image(systemName: "circle.fill")
                .font(.system(size: 7))
                .foregroundStyle(tint)
            VStack(alignment: .leading, spacing: 1) {
                Text(host.name)
                if let detail {
                    Text(detail)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }
            }
        }
        .help(host.baseURL.absoluteString)
    }

    /// `.green`/`.red` here are status semantics, not brand color -- the same
    /// meaning the system uses in its own connection indicators.
    private var tint: Color {
        switch reachability {
        case .unknown, .checking: .secondary
        case .up: .green
        case .down: .red
        }
    }

    private var detail: String? {
        switch reachability {
        case .unknown: nil
        case .checking: "Checking…"
        case let .up(status):
            status.busy ? "Busy · \(status.version)" : "Ready · \(status.version)"
        case let .down(reason): reason
        }
    }
}
