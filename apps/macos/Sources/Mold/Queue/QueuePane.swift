import MoldClient
import SwiftUI

/// Work in flight, on every machine, with the controls to act on it.
struct QueuePane: View {
    // Not `private`: `QueuePane+Toolbar.swift` reads both, the same reason
    // `LibraryPane.swift`'s `hosts`/`library` aren't private either.
    @Environment(HostStore.self) var hosts
    @Environment(QueueStore.self) var queue
    /// Not `private`, and deliberately: the toolbar button that raises this
    /// lives in `QueuePane+Toolbar.swift`, another file.
    @State var pendingDestruction: Destruction?

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
                                if queue.queuePaused[host.id] == true {
                                    Label("This machine's queue is paused.", systemImage: "pause.circle")
                                        .foregroundStyle(.secondary)
                                }
                                rows(host: host, entries: entries)
                            }
                        }
                    }
                }
                .listStyle(.inset)
            }
        }
        .failureBanner(hosts)
        .navigationTitle("Queue")
        .navigationSubtitle(QueueSummary.sentence(queue.all))
        .toolbar { toolbar }
        .destructionDialog($pendingDestruction)
        .task { await load() }
        .focusedSceneValue(\.refreshAction) { Task { await load() } }
    }

    /// One host's rows. `.onMove` is attached only when the machine
    /// advertises reorder -- SwiftUI draws no drag affordance without it,
    /// so a duplicated `ForEach` is what keeps the modifier truly absent
    /// rather than present-but-inert.
    @ViewBuilder
    private func rows(host: MoldHost, entries: [QueueEntry]) -> some View {
        let groups = queue.groups(on: host.id)
        let canReorder = hosts.capabilities[host.id]?.canReorderQueue == true
        if canReorder {
            ForEach(groups) { row($0, host: host, entries: entries, canReorder: true) }
                .onMove { source, destination in
                    let calls = QueuePane.reorderCalls(
                        source: source, destination: destination, groups: groups, entries: entries)
                    Task { await queue.reorder(calls, on: host.id) }
                }
        } else {
            ForEach(groups) { row($0, host: host, entries: entries, canReorder: false) }
        }
    }

    @ViewBuilder
    private func row(
        _ group: QueueGroup, host: MoldHost, entries: [QueueEntry], canReorder: Bool
    ) -> some View {
        if group.isExpandable {
            QueueBatchRow(
                group: group,
                rowAct: { action, entry in act(action, on: entry, host: host) },
                groupAct: { action in Task { await queue.act(action, onLiveChildrenOf: group, host: host.id) } })
        } else {
            let entry = group.rows[0]
            let reorderable = canReorder && entry.state.isReorderable
            QueueRow(
                entry: entry, isReorderable: reorderable,
                canMoveUp: reorderable && QueueRow.canMove(entry.id, .up, in: entries),
                canMoveDown: reorderable && QueueRow.canMove(entry.id, .down, in: entries),
                moveUp: { move(entry.id, .up, host: host, entries: entries) },
                moveDown: { move(entry.id, .down, host: host, entries: entries) },
                act: { act($0, on: entry, host: host) })
        }
    }

    private func move(_ id: String, _ direction: QueueRow.MoveDirection, host: MoldHost, entries: [QueueEntry]) {
        guard let call = QueueRow.moveCall(id, direction, in: entries) else { return }
        Task { await queue.reorder([call], on: host.id) }
    }

    private func act(_ action: QueueRow.Action, on entry: QueueEntry, host: MoldHost) {
        Task {
            switch action {
            case .cancel: await queue.cancel(entry, on: host.id)
            case .pause: await queue.pause(entry, on: host.id)
            case .resume: await queue.resume(entry, on: host.id)
            case .retry: await queue.retry(entry, on: host.id)
            }
            await load()
        }
    }

    // Not `private`: `QueuePane+Toolbar.swift`'s Refresh button calls this
    // too.
    func load() async {
        await queue.refresh()
    }
}
