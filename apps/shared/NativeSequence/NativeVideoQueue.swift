import MoldClient
import SwiftUI

/// One Queue row's worth of a duration parent or an authored sequence, in the
/// shared `VideoJobRow` words. Each app draws it with its ORDINARY row chrome
/// inside the machine's own section; this type only says what the row shows
/// and offers, so the two apps cannot drift.
struct NativeVideoQueueItem: Identifiable {
    enum Source {
        case video(ObservedVideoJob)
        case sequence(NativeSequenceSession.Job)
    }

    enum Action: Hashable {
        case stop, resume, startAgain, delete, dismiss, openInLibrary
    }

    let id: String
    let host: MoldHost
    let source: Source
    let headline: String
    let prompt: String?
    let row: VideoJobRow
    /// The one line worth reading beside the caption: an error, or what
    /// happened to the saved video. Everything else is in the details.
    let note: String?
    let notes: [String]
    let actions: [Action]
    /// Whether this device may act on it now (its owner predicate holds).
    let isCurrent: Bool
    let isBusy: Bool
    let operation: String

    static let liveSymbols: Set<VideoJobRow.Symbol> = [.waiting, .rendering, .joining, .stopping, .paused]
    var isLive: Bool { Self.liveSymbols.contains(row.symbol) }
    var isRendering: Bool { [.rendering, .joining, .stopping].contains(row.symbol) }

    func offers(_ action: Action) -> Bool { actions.contains(action) }

    func title(_ action: Action, stopRequested: Bool = false) -> String {
        switch action {
        case .stop: stopRequested ? "Check Stop" : "Stop"
        case .resume: "Resume"
        case .startAgain: "Start again…"
        case .delete: "Delete job…"
        case .dismiss: "Dismiss from Queue"
        case .openInLibrary: "Open in Library"
        }
    }

    static func symbol(_ action: Action) -> String {
        switch action {
        case .stop: "stop.circle"
        case .resume: "play"
        case .startAgain: "arrow.counterclockwise"
        case .delete: "trash"
        case .dismiss: "xmark"
        case .openInLibrary: "photo.on.rectangle"
        }
    }

    static func isDestructive(_ action: Action) -> Bool { action == .stop || action == .delete }

    /// The contextual menu: the same list the row's buttons and swipes draw,
    /// destructive last behind a divider (`RowAction.ordered`).
    func menu(stopRequested: Bool = false) -> [RowAction<Action>] {
        RowAction.ordered(actions.map {
            RowAction(kind: $0, title: title($0, stopRequested: stopRequested), isDestructive: Self.isDestructive($0))
        })
    }
}

/// Recovery presentation shared by every row: the Start again / discard sheet
/// and the one Delete confirmation, anchored to the exact row and owner.
@MainActor @Observable
final class NativeVideoQueuePresenter {
    var showingRecovery = false
    var discarding: NativeSequenceSession.Job?
    var deleting: ObservedVideoJob?
    var inspecting: String?
    init() {}
}

/// Both native queues use the same explicit parent protocol and recovery sheet.
/// Not a view: the panes ask it for rows per machine and draw them with their
/// own chrome, and `NativeVideoQueueNotices` carries what is not a row.
@MainActor
struct NativeVideoQueue {
    let sequence: NativeSequenceSession
    let videos: VideoJobCoordinator
    let restarts: VideoRestartController
    let targets: [VideoRestartTarget]
    let backend: (VideoRestartTarget) -> (any MoldBackend)?
    let current: (VideoRestartTarget) -> Bool
    let capabilities: (UUID) -> Capabilities?
    let open: (MoldHost, String) -> Void
    var changed: (() -> Void)?
    let presenter: NativeVideoQueuePresenter

    // MARK: Rows

    /// Finished and stopped parents are not drawn: like every other row they
    /// leave the Queue on their own (`ObservedVideoJob.belongsInQueue`).
    func items(on host: MoldHost.ID) -> [NativeVideoQueueItem] {
        videos.jobs.filter { $0.owner.host.id == host && $0.belongsInQueue }.map(item)
            + sequence.queueJobs.filter { $0.host.id == host }.map(item)
    }

    func item(id: String) -> NativeVideoQueueItem? {
        videos.jobs.first { $0.id == id }.map(item) ?? sequence.jobs.first { $0.id == id }.map(item)
    }

    /// Live parents per machine -- what Queue headers and badges add to the
    /// ordinary rows. Their clips are never ordinary `/api/queue` rows (chain
    /// stages are plan work items; H3 children are internal), so nothing is
    /// counted twice.
    func liveSymbols(on host: MoldHost.ID? = nil) -> [VideoJobRow.Symbol] {
        Self.liveSymbols(videos: videos, sequence: sequence, on: host)
    }

    func liveCount(on host: MoldHost.ID? = nil) -> Int { liveSymbols(on: host).count }

    /// The same count for a surface that is not the Queue (badges, Machines).
    static func liveSymbols(videos: VideoJobCoordinator, sequence: NativeSequenceSession,
                            on host: MoldHost.ID? = nil) -> [VideoJobRow.Symbol] {
        let parents = videos.jobs.filter { host == nil || $0.owner.host.id == host }.compactMap { job in
            job.detail.map { VideoJobRow($0, current: true).symbol } ?? (job.cancelledBeforeAdmission ? nil : .waiting)
        }
        let authored = sequence.jobs.filter { host == nil || $0.host.id == host }.map { job in
            job.detail.map {
                VideoJobRow(chain: $0.state, executionPhase: $0.executionPhase, currentStage: $0.currentStage,
                            stageCount: $0.stageCount, cancelling: job.cancelling, current: true).symbol
            } ?? .waiting
        }
        return (parents + authored).filter { NativeVideoQueueItem.liveSymbols.contains($0) }
    }

    /// The operation ids drawn as video rows, so Also Running never repeats one.
    static func operationIDs(videos: VideoJobCoordinator, sequence: NativeSequenceSession,
                             on host: MoldHost.ID) -> Set<String> {
        Set(videos.jobs.filter { $0.owner.host.id == host }.map(\.operationId)
            + sequence.jobs.filter { $0.host.id == host }.map(\.jobId))
    }

    private func videoBackend(_ owner: VideoJobOwner) -> (any MoldBackend)? {
        let target = VideoRestartTarget(host: owner.host, instanceId: owner.instanceId)
        return current(target) ? backend(target) : nil
    }

    private func videoCurrent(_ owner: VideoJobOwner) -> Bool {
        current(.init(host: owner.host, instanceId: owner.instanceId))
    }

    private func target(_ job: NativeSequenceSession.Job) -> VideoRestartTarget? {
        guard let instance = job.installation else { return nil }
        return .init(host: job.host, instanceId: instance)
    }

    private func sequenceBackend(_ job: NativeSequenceSession.Job) -> (any MoldBackend)? {
        guard let target = target(job), current(target) else { return nil }
        return backend(target)
    }

    private func item(_ job: ObservedVideoJob) -> NativeVideoQueueItem {
        let live = videoBackend(job.owner) != nil && videoCurrent(job.owner)
        let row: VideoJobRow = if let detail = job.detail {
            VideoJobRow(detail, current: live)
        } else {
            VideoJobRow(.init(state: job.cancelledBeforeAdmission ? .cancelled : .queued, progressPhase: nil,
                              completedClips: 0, clipCount: 1, totalDurationMs: nil, canCancel: true, current: live))
        }
        var actions: [NativeVideoQueueItem.Action] = []
        var output: String?
        if let filename = job.detail?.outputFilename {
            if job.outputAvailable == false { output = "Saved video is no longer in Library." }
            else if live, !filename.isEmpty { actions.append(.openInLibrary) }
        }
        if live {
            if job.detail?.canResume == true { actions.append(.resume) }
            if capabilities(job.owner.host.id)?.canRestartVideoJob == true, job.detail?.state.canRequestRestart == true {
                actions.append(.startAgain)
            }
        }
        if job.cancelledBeforeAdmission || job.detail?.state == .completed { actions.append(.dismiss) }
        if live {
            if job.detail?.canCancel == true || (job.detail == nil && !job.cancelledBeforeAdmission) { actions.append(.stop) }
            if job.detail?.canDelete == true { actions.append(.delete) }
        }
        let caption = job.detail == nil
            ? (job.cancelledBeforeAdmission ? "Cancelled before queueing" : "Checking saved video attempt…")
            : row.caption
        let error = job.detail?.errorDetail ?? job.detail?.error
        var notes = [error, output, job.outputMessage, job.message].compactMap { $0 }
        if !live { notes.append("Reconnect to this video’s original machine to check or change it.") }
        notes += job.detail?.plan.warnings ?? []
        return NativeVideoQueueItem(
            id: job.id, host: job.owner.host, source: .video(job),
            headline: job.detail.map { Model.headline(forName: $0.plan.model) } ?? "Video",
            prompt: nil,
            row: VideoJobRow(caption: caption, detail: job.detail == nil ? "" : row.detail, symbol: row.symbol, progress: row.progress),
            note: error ?? output ?? job.outputMessage ?? job.message,
            notes: notes, actions: actions, isCurrent: live,
            isBusy: videos.isMutating(job) || videos.isDeleting(job) || videos.isCheckingOutput(job),
            operation: job.operationId)
    }

    private func item(_ job: NativeSequenceSession.Job) -> NativeVideoQueueItem {
        let live = sequenceBackend(job) != nil
        let row: VideoJobRow = if let detail = job.detail {
            VideoJobRow(chain: detail.state, executionPhase: detail.executionPhase, currentStage: detail.currentStage,
                        stageCount: detail.stageCount, cancelling: job.cancelling || detail.cancelling == true, current: live)
        } else {
            VideoJobRow(chain: .queued, executionPhase: nil, currentStage: 0, stageCount: 1,
                        cancelling: job.cancelling, current: live)
        }
        var actions: [NativeVideoQueueItem.Action] = []
        if job.detail?.galleryFilename != nil { actions.append(.openInLibrary) }
        if live, let detail = job.detail {
            if ChainJobRecoveryPolicy.canResume(state: detail.state, ephemeral: detail.ephemeral == true) { actions.append(.resume) }
            if capabilities(job.host.id)?.canRestartChain == true, [.failed, .interrupted, .paused].contains(detail.state) {
                actions.append(.startAgain)
            }
        }
        if job.detail?.state == .completed || job.detail?.state == .cancelled { actions.append(.dismiss) }
        if live {
            if (job.detail.map { ChainJobRecoveryPolicy.canCancel(state: $0.state) } ?? true), !job.cancelling {
                actions.append(.stop)
            }
            if capabilities(job.host.id)?.canRestartChain == true,
               job.detail?.state == .failed || job.detail?.state == .interrupted { actions.append(.delete) }
        }
        let message = job.message.isEmpty ? nil : job.message
        let note = job.detail?.error ?? (job.cancelling ? message : nil)
        var notes = [job.detail?.error, message].compactMap { $0 }
        if !live { notes.append("Reconnect to this video’s original machine to check or change it.") }
        return NativeVideoQueueItem(
            id: job.id, host: job.host, source: .sequence(job),
            headline: job.detail.map { Model.headline(forName: $0.model) } ?? "Video",
            prompt: job.detail?.script?.stage.first?.prompt,
            row: row, note: note, notes: notes, actions: actions, isCurrent: live,
            isBusy: job.cancelling, operation: job.jobId)
    }

    // MARK: Actions

    /// Every surface -- icon button, swipe, contextual menu, details -- calls
    /// this one dispatcher, so one gesture is one action.
    func perform(_ action: NativeVideoQueueItem.Action, on item: NativeVideoQueueItem) {
        guard item.offers(action) else { return }
        switch item.source {
        case let .video(job): perform(action, video: job)
        case let .sequence(job): perform(action, sequence: job)
        }
    }

    private func perform(_ action: NativeVideoQueueItem.Action, video job: ObservedVideoJob) {
        let owner = job.owner
        if action == .dismiss { videos.dismiss(job); return }
        guard let service = videoBackend(owner) else { return }
        let isCurrent = { @MainActor in videoCurrent(owner) }
        switch action {
        case .stop: Task { await videos.cancel(job, backend: service, current: isCurrent) }
        case .resume: Task { await videos.resume(job, backend: service, current: isCurrent) }
        case .startAgain: present(host: owner.host, instance: owner.instanceId, id: job.operationId, kind: .videoJob)
        case .delete: presenter.deleting = job
        case .openInLibrary:
            guard let filename = job.detail?.outputFilename else { return }
            Task {
                guard await videos.checkOutput(job, backend: service, current: isCurrent, force: true) == true,
                      isCurrent(), videos.jobs.contains(where: {
                          $0.id == job.id && $0.owner == owner && $0.detail?.outputFilename == filename && $0.outputAvailable == true
                      }) else { return }
                open(owner.host, filename)
            }
        case .dismiss: break
        }
    }

    private func perform(_ action: NativeVideoQueueItem.Action, sequence job: NativeSequenceSession.Job) {
        switch action {
        case .dismiss: sequence.dismiss(job); return
        case .openInLibrary:
            if let filename = job.detail?.galleryFilename { open(job.host, filename) }
            return
        default: break
        }
        guard let service = sequenceBackend(job) else { return }
        switch action {
        case .stop: sequence.cancel(job, backend: service)
        case .resume: sequence.resume(job, backend: service)
        case .startAgain: present(host: job.host, instance: job.installation, id: job.jobId, kind: .chain)
        case .delete: present(host: job.host, instance: job.installation, id: job.jobId, kind: .chain, discard: job)
        case .dismiss, .openInLibrary: break
        }
    }

    /// Confirmed Delete job… for the exact row and owner the dialog opened on.
    func confirmDelete(_ job: ObservedVideoJob) {
        presenter.deleting = nil
        guard let service = videoBackend(job.owner) else { return }
        Task { await videos.delete(job, backend: service, current: { videoCurrent(job.owner) }) }
    }

    func stopRequested(_ item: NativeVideoQueueItem) -> Bool {
        if case let .video(job) = item.source { return videos.isCancellationRequested(job) }
        return false
    }

    /// The owner re-check every row used to need a "Check queue" press for.
    /// Runs on the Queue's ordinary refresh and poll; `force` is the pane's
    /// own Refresh, which also re-verifies saved Library videos.
    func recheck(force: Bool = false) async {
        await videos.recheck(backend: videoBackend, current: videoCurrent, force: force)
    }

    /// Reads a finished parent's saved video once, as the old card did.
    func checkOutput(_ item: NativeVideoQueueItem) async {
        guard case let .video(job) = item.source, let service = videoBackend(job.owner) else { return }
        await videos.checkOutput(job, backend: service, current: { videoCurrent(job.owner) })
    }

    func present(host: MoldHost, instance: String?, id: String, kind: VideoRestartProtocol,
                             discard: NativeSequenceSession.Job? = nil) {
        guard let instance else { return }
        let source = VideoRestartTarget(host: host, instanceId: instance)
        guard current(source), let service = backend(source) else { return }
        presenter.discarding = discard
        presenter.showingRecovery = true
        Task {
            await restarts.open(
                source: source, jobId: id, protocolKind: kind, backend: service, current: { current(source) })
        }
    }

    func targetOf(_ job: NativeSequenceSession.Job) -> VideoRestartTarget? { target(job) }
    func pageBackend(_ owner: VideoJobOwner) -> (any MoldBackend)? { videoBackend(owner) }
    func pageCurrent(_ owner: VideoJobOwner) -> Bool { videoCurrent(owner) }
}
