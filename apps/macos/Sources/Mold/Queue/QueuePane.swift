import MoldClient
import SwiftUI

/// Work in flight, on every machine, with the controls to act on it.
struct QueuePane: View {
    @Environment(HostStore.self) private var hosts
    @Environment(QueueStore.self) private var queue

    var body: some View {
        Group {
            if queue.all.isEmpty {
                ContentUnavailableView(
                    queue.isLoading ? "Checking each machine…" : "Nothing queued",
                    systemImage: "list.bullet.indent",
                    description: Text(queue.isLoading ? "" : "Renders you start appear here.")
                )
            } else {
                List {
                    ForEach(hosts.hosts) { host in
                        let entries = queue.entries(on: host.id)
                        if !entries.isEmpty {
                            Section(host.name) {
                                ForEach(entries) { entry in
                                    QueueRow(entry: entry, act: { act($0, on: entry, host: host) })
                                }
                            }
                        }
                    }
                }
                .listStyle(.inset)
                .alert("That didn't work", isPresented: .constant(queue.failure != nil)) {
                    Button("OK") { }
                } message: {
                    Text(queue.failure ?? "")
                }
            }
        }
        .navigationTitle("Queue")
        .navigationSubtitle(subtitle)
        .toolbar { toolbar }
        .task { await load() }
        .focusedSceneValue(\.refreshAction) { Task { await load() } }
    }

    private var subtitle: String {
        let live = queue.all.filter(\.state.isLive).count
        return live == 0 ? "Idle" : "\(live) waiting or running"
    }

    @ToolbarContentBuilder private var toolbar: some ToolbarContent {
        ToolbarItem {
            Button { Task { await load() } } label: {
                Label("Refresh", systemImage: "arrow.clockwise")
            }
            .disabled(queue.isLoading)
        }
    }

    private func act(_ action: QueueRow.Action, on entry: QueueEntry, host: MoldHost) {
        let backend = hosts.backend(for: host)
        Task {
            switch action {
            case .cancel: await queue.cancel(entry, on: host.id, backend: backend)
            case .pause: await queue.pause(entry, on: host.id, backend: backend)
            case .resume: await queue.resume(entry, on: host.id, backend: backend)
            case .retry: await queue.retry(entry, on: host.id, backend: backend)
            }
            await load()
        }
    }

    private func load() async {
        await hosts.refreshAll()
        await queue.refresh(hosts: hosts.hosts) { hosts.backend(for: $0) }
    }
}
