import MoldClient
import SwiftUI

/// Work in flight, on every machine, with the controls to act on it.
struct QueuePane: View {
    // Not `private`: `QueuePane+Toolbar.swift` reads both, the same reason
    // `LibraryPane.swift`'s `hosts`/`library` aren't private either.
    @Environment(HostStore.self) var hosts
    @Environment(QueueStore.self) var queue
    @Environment(GenerateController.self) var generate
    @Environment(\.sourceMediaOwner) private var openVideo
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
    @State var detailTarget: QueueDetailTarget?
    /// The video rows' recovery sheet, Delete confirmation and details.
    @State var videoPresenter = NativeVideoQueuePresenter()

    var body: some View {
        Group {
            if queue.all.isEmpty, alsoRunning.isEmpty, pausedMachines.isEmpty, generate.sequence.jobs.isEmpty, !generate.videoJobs.hasQueueContent, generate.sequence.historyNotice == nil, generate.videoRestarts.pending.isEmpty, generate.videoRestarts.error == nil {
                ContentUnavailableView(
                    queue.isLoading ? "Checking each machine…" : "Nothing queued",
                    systemImage: "list.bullet.indent",
                    description: Text(queue.isLoading ? "" : "Renders you start appear here.")
                )
            } else {
                List(selection: $selection) {
                    NativeVideoQueueNotices(queue: videoQueue, presenter: videoPresenter)
                        .selectionDisabled(true)
                    // A paused machine says so whether or not it has rows:
                    // an empty queue behind a closed gate looks exactly like
                    // an idle machine, and it is not one.
                    ForEach(pausedMachines) { host in
                        Label(QueueGateOffer.pausedSentence(machine: host.name),
                              systemImage: "pause.circle")
                            .foregroundStyle(.secondary)
                    }
                    // A status line is not a row to select: this `List` is
                    // selection-bound, so without this each sentence is
                    // click-highlightable and arrow-key reachable.
                    .selectionDisabled(true)
                    ForEach(hosts.hosts) { host in
                        let entries = queue.entries(on: host.id)
                        let videos = videoQueue.items(on: host.id)
                        if !entries.isEmpty || !videos.isEmpty {
                            Section(host.name) {
                                ForEach(videos) { item in
                                    VideoQueueRow(item: item, queue: videoQueue) { videoPresenter.inspecting = item.id }
                                        .selectionDisabled(true)
                                }
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
        .navigationSubtitle(QueueSummary.sentence(queue.all, videos: videoQueue.liveSymbols()))
        .toolbar { toolbar }
        .sheet(item: $detailTarget) { target in
            QueueDetailSheet(entry: target.entry, host: target.host)
        }
        .sheet(item: Binding(get: { videoPresenter.inspecting.map(VideoDetailID.init) },
                             set: { videoPresenter.inspecting = $0?.id })) { target in
            VideoQueueDetailSheet(queue: videoQueue, id: target.id)
        }
        .destructionDialog($pendingDestruction)
        .task { await load(includeVideoParents: false); await generate.discoverSequences() }
        .task(id: durationDiscoveryMembership) {
            await generate.recoverDurationJobs(activity: activity.byHost, discover: true)
        }
        .task {
            while !Task.isCancelled {
                await queue.pollProgress()
                try? await Task.sleep(for: .seconds(1))
            }
        }
        // The per-video owner re-check that used to need a Check queue press.
        .task {
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(15))
                await videoQueue.recheck()
            }
        }
        // The clip upscales already running on each machine. Not the
        // Library's alone: this pane is where they are DRAWN, and it must
        // not depend on somebody having opened the Library first. One
        // listing per machine, and idempotent.
        .task { await upscales.recover() }
        .focusedSceneValue(\.refreshAction) { Task { await load() } }
        .focusedSceneValue(\.queueSelection, queueSelection)
    }

    /// The video parents and authored sequences, drawn as ordinary rows.
    var videoQueue: NativeVideoQueue {
        NativeVideoQueue(sequence: generate.sequence, videos: generate.videoJobs, restarts: generate.videoRestarts,
            targets: hosts.hosts.compactMap { host in
                guard let instance = hosts.instanceID(of: host.id) else { return nil }
                return VideoRestartTarget(host: host, instanceId: instance)
            }, backend: { target in
                guard recoveryCurrent(target) else { return nil }
                return hosts.backend(for: target.host.id)
            }, current: recoveryCurrent, capabilities: { hosts.capabilities[$0] },
            open: { host, filename in openVideo(PrintID(host: host.id, filename: filename)) }, changed: {
                Task { await generate.discoverSequences(); await generate.recoverDurationJobs(discover: true) }
            }, presenter: videoPresenter)
    }

    private struct VideoDetailID: Identifiable { let id: String }

    private func recoveryCurrent(_ target:VideoRestartTarget)->Bool {
        hosts.host(target.host.id)==target.host && hosts.instanceID(of:target.host.id)==target.instanceId
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
    func load(includeVideoParents: Bool = true) async {
        guard !queue.isSeeded else { return }
        if let fixture = Self.fixtureIfRequested() {
            queue.seed(from: fixture)
            return
        }
        await queue.refresh()
        if includeVideoParents {
            await generate.recoverDurationJobs(activity: activity.byHost, discover: true)
            await videoQueue.recheck(force: true)
        }
    }

    private var durationDiscoveryMembership: [VideoJobDiscoveryMembership] {
        hosts.hosts.sorted { $0.id.uuidString < $1.id.uuidString }.compactMap { host in
            guard hosts.capabilities[host.id]?.canCreateVideoJob == true,
                  let owner = generate.durationOwner(host) else { return nil }
            return VideoJobDiscoveryMembership(owner: owner, activity: activity.byHost[host.id])
        }
    }
}
