import MoldClient
import SwiftUI

/// Work in flight, on every machine.
struct QueuePane: View {
    @Environment(HostStore.self) private var hosts
    @State private var byHost: [MoldHost.ID: [QueueEntry]] = [:]
    @State private var isLoading = false

    var body: some View {
        Group {
            if rows.isEmpty {
                ContentUnavailableView(
                    isLoading ? "Checking each machine…" : "Nothing queued",
                    systemImage: "list.bullet.indent",
                    description: Text(isLoading ? "" : "Renders you start appear here.")
                )
            } else {
                List {
                    ForEach(hosts.hosts) { host in
                        let entries = byHost[host.id] ?? []
                        if !entries.isEmpty {
                            Section(host.name) {
                                ForEach(entries) { QueueRow(entry: $0) }
                            }
                        }
                    }
                }
                .listStyle(.inset)
            }
        }
        .navigationTitle("Queue")
        .navigationSubtitle(subtitle)
        .toolbar {
            ToolbarItem {
                Button { Task { await load() } } label: {
                    Label("Refresh", systemImage: "arrow.clockwise")
                }
                .disabled(isLoading)
            }
        }
        .task { await load() }
    }

    private var rows: [QueueEntry] { byHost.values.flatMap(\.self) }

    private var subtitle: String {
        let live = rows.filter(\.state.isLive).count
        return live == 0 ? "Idle" : "\(live) waiting or running"
    }

    private func load() async {
        isLoading = true
        defer { isLoading = false }
        await hosts.refreshAll()
        await withTaskGroup(of: (MoldHost.ID, [QueueEntry]).self) { group in
            for host in hosts.hosts {
                let client = hosts.backend(for: host)
                group.addTask {
                    // Merged by id: a live-only row and a durable row for the
                    // same job are one job.
                    (host.id, (try? await client.queue())?.merged ?? [])
                }
            }
            for await (id, entries) in group { byHost[id] = entries }
        }
    }
}
