import MoldClient
import SwiftUI

/// Work in flight, on every machine, with the controls to act on it.
struct QueuePane: View {
    // Not `private`: `QueuePane+Toolbar.swift` reads both, the same reason
    // `LibraryPane.swift`'s `hosts`/`library` aren't private either.
    @Environment(HostStore.self) var hosts
    @Environment(QueueStore.self) var queue
    @Environment(TransferStore.self) var transfers
    /// For `pullThenRetry(_:entry:host:)`'s own `QueueHoldRow.pullThenRetry` call.
    @Environment(DownloadStore.self) var downloads
    /// Not `private`, and deliberately: the toolbar button that raises this
    /// lives in `QueuePane+Toolbar.swift`, another file.
    @State var pendingDestruction: Destruction?
    /// The `List` selection -- a `QueueGroup.id` or a batch child's
    /// `QueueEntry.id`, one string space. `QueuePane+Commands.swift` reads
    /// it for the Queue menu's `FocusedValue`.
    @State var selection: String?

    var body: some View {
        Group {
            if queue.all.isEmpty {
                ContentUnavailableView(
                    queue.isLoading ? "Checking each machine…" : "Nothing queued",
                    systemImage: "list.bullet.indent",
                    description: Text(queue.isLoading ? "" : "Renders you start appear here.")
                )
            } else {
                List(selection: $selection) {
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
        .transferCaption(transfers.summary)
        .navigationTitle("Queue")
        .navigationSubtitle(QueueSummary.sentence(queue.all))
        .toolbar { toolbar }
        .destructionDialog($pendingDestruction)
        .task { await load() }
        .focusedSceneValue(\.refreshAction) { Task { await load() } }
        .focusedSceneValue(\.queueSelection, queueSelection)
    }

    /// Not `private`: `QueuePane+Commands.swift`'s Move Up/Down items call
    /// this too, and `private` does not cross a file boundary.
    func move(_ id: String, _ direction: QueueRow.MoveDirection, host: MoldHost, entries: [QueueEntry]) {
        guard let call = QueueRow.moveCall(id, direction, in: entries) else { return }
        Task { await queue.reorder([call], on: host.id) }
    }

    /// Not `private`: `QueuePane+Commands.swift`'s Pause/Resume/Try
    /// Again/Cancel items call this too.
    func act(_ action: QueueRow.Action, on entry: QueueEntry, host: MoldHost) {
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
    // too. Seeds once; `isSeeded` then short-circuits `refresh`'s own
    // `poll`/`hydrate` (`QueueStore+Fixture.swift`), so a later Refresh press
    // cannot overwrite the fixture with a real machine's own answer.
    func load() async {
        guard !queue.isSeeded else { return }
        if let fixture = Self.fixtureIfRequested() {
            queue.seed(from: fixture)
            return
        }
        await queue.refresh()
    }
}
