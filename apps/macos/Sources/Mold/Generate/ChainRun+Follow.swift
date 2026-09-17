import Foundation
import MoldClient

// Following one chain job to settlement. Split from `ChainRun`'s own shape
// purely for size.
@MainActor
extension ChainRun {
    /// Follows the job until the SERVER says it is over.
    ///
    /// A dropped stream is not a settlement: the job is durable and keeps
    /// rendering, so this reconnects with the same `2^n` backoff capped at 32 s
    /// that `HostStore+Events.watch` uses, re-reading the job each time so the
    /// stage counter resyncs rather than being invented. The record in
    /// `PendingChain` is KEPT throughout -- it used to be forgotten here, which
    /// threw away the app's only handle on a job that was still burning GPU.
    /// Only a terminal state read from the host ends this loop.
    func follow(
        _ jobId: String, on host: MoldHost.ID, backend: any MoldBackend, report: Reporter
    ) async {
        var attempt = 0
        while !Task.isCancelled {
            do {
                for try await event in backend.chainJobEvents(id: jobId) {
                    guard !Task.isCancelled else { return }
                    attempt = 0
                    if apply(event, jobId: jobId, host: host, report: report) { return }
                }
            } catch {
                // A dropped stream. The job is durable and still rendering.
            }
            // Ended or dropped, the question is the same and the SERVER
            // answers it: READ the job once -- never create it again, which
            // would render the whole thing twice. A read that itself fails is
            // the same bad minute as the stream, and is retried.
            guard !Task.isCancelled else { return }
            if let detail = try? await backend.chainJob(id: jobId),
               settleFromDetail(detail, host: host, report: report) { return }
            guard !Task.isCancelled else { return }
            try? await Task.sleep(for: backoff(attempt))
            attempt += 1
        }
    }

    /// Applies one frame. Returns `true` when the follow is over.
    private func apply(
        _ event: ChainJobEvent, jobId: String, host: MoldHost.ID, report: Reporter
    ) -> Bool {
        switch event {
        case let .snapshot(detail):
            // The snapshot is the whole truth, including a job that had
            // already settled before this client attached.
            update({ progress in
                progress.stageCount = max(detail.stageCount, 1)
                progress.currentStage = max(detail.currentStage + 1, 1)
                progress.isPaused = detail.state == .paused
            }, report: report)
            return detail.state.isTerminal
                && settleFromDetail(detail, host: host, report: report)
        case let .stageStart(stage):
            // A new clip resets the step counter: the old one belonged to the
            // clip before it and would read as progress that already happened.
            update({ $0.currentStage = stage + 1; $0.step = nil; $0.total = nil },
                   report: report)
        case let .denoiseStep(stage, step, total):
            update({ progress in
                progress.currentStage = max(stage + 1, progress.currentStage)
                progress.step = step
                progress.total = total
                progress.isPaused = false
            }, report: report)
        case .stageDone, .finalizing, .other:
            break
        case let .finalized(galleryFilename):
            settle()
            report.finished(galleryFilename, host)
            return true
        case let .stateChanged(state, error):
            // A PARKED chain is not over: a host restart parks an ephemeral
            // chain and it can be resumed (CLAUDE.md, "Scripted sequences").
            guard state.isTerminal else {
                update({ $0.isPaused = state == .paused }, report: report)
                return false
            }
            settle()
            switch state {
            // `finalized` normally arrives first and has already returned; a
            // `completed` reaching here published nothing this client fetches.
            case .completed: report.finished(nil, host)
            case .cancelled: report.failed("Cancelled")
            default: report.failed(error ?? "The render didn't finish.")
            }
            return true
        }
        return false
    }

    /// Settles from a READ of the job. `false` means it is not over and the
    /// follow reconnects -- the record stays, because the job stays.
    func settleFromDetail(
        _ detail: ChainJobDetail, host: MoldHost.ID, report: Reporter
    ) -> Bool {
        guard detail.state.isTerminal else {
            update({ progress in
                progress.stageCount = max(detail.stageCount, 1)
                progress.currentStage = max(detail.currentStage + 1, 1)
                progress.isPaused = detail.state == .paused
            }, report: report)
            return false
        }
        settle()
        switch detail.state {
        case .cancelled: report.failed("Cancelled")
        case .failed: report.failed(detail.error ?? "The render didn't finish.")
        default: report.finished(detail.galleryFilename, host)
        }
        return true
    }
}
