import MoldClient
import SwiftUI

/// Work in flight, on every machine, with the controls to act on it.
struct QueuePane: View {
    // Not `private`: `QueuePane+Toolbar.swift` reads both, the same reason
    // `LibraryPane.swift`'s `hosts`/`library` aren't private either.
    @Environment(HostStore.self) var hosts
    @Environment(QueueStore.self) var queue
    @Environment(TransferStore.self) var transfers
    /// What the machines are doing that never becomes a queue row, and the
    /// clip upscales this app started -- both drawn under Also Running.
    @Environment(ActivityStore.self) var activity
    @Environment(UpscaleStore.self) var upscales
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
            if queue.all.isEmpty, alsoRunning.isEmpty, pausedMachines.isEmpty {
                ContentUnavailableView(
                    queue.isLoading ? "Checking each machine…" : "Nothing queued",
                    systemImage: "list.bullet.indent",
                    description: Text(queue.isLoading ? "" : "Renders you start appear here.")
                )
            } else {
                List(selection: $selection) {
                    // A paused machine says so whether or not it has rows:
                    // an empty queue behind a closed gate looks exactly like
                    // an idle machine, and it is not one.
                    ForEach(pausedMachines) { host in
                        Label(QueueGateOffer.pausedSentence(machine: host.name),
                              systemImage: "pause.circle")
                            .foregroundStyle(.secondary)
                    }
                    ForEach(hosts.hosts) { host in
                        let entries = queue.entries(on: host.id)
                        if !entries.isEmpty {
                            Section(host.name) {
                                rows(host: host, entries: entries)
                            }
                        }
                        alsoRunningSection(host)
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
        // The clip upscales already running on each machine. Not the
        // Library's alone: this pane is where they are DRAWN, and it must
        // not depend on somebody having opened the Library first. One
        // listing per machine, and idempotent.
        .task { await upscales.recover() }
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
