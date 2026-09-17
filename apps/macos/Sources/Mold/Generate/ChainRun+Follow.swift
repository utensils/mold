import Foundation
import MoldClient

// Following one chain job to settlement. Split from `ChainRun`'s own shape
// purely for size.
@MainActor
extension ChainRun {
    func follow(
        _ jobId: String, on host: MoldHost.ID, backend: any MoldBackend, report: Reporter
    ) async {
        do {
            for try await event in backend.chainJobEvents(id: jobId) {
                guard !Task.isCancelled else { return }
                if apply(event, jobId: jobId, host: host, report: report) { return }
            }
            // The stream ended without a terminal frame. READ the job once
            // rather than leaving the canvas spinning -- and never by
            // creating it again, which would render the whole thing twice.
            guard !Task.isCancelled else { return }
            settleFromDetail(try await backend.chainJob(id: jobId), host: host, report: report)
        } catch {
            guard !Task.isCancelled else { return }
            // A dropped stream does NOT mean the work stopped: the job is
            // durable and is still going to finish. Not a settlement.
            settle()
            report.failed("Lost contact while rendering. "
                + "The job may still be running — check the Queue.")
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
            }, report: report)
            guard detail.state.isTerminal else { return false }
            settleFromDetail(detail, host: host, report: report)
            return true
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
            }, report: report)
        case .stageDone, .finalizing, .other:
            break
        case let .finalized(galleryFilename):
            settle()
            report.finished(galleryFilename, host)
            return true
        case let .stateChanged(state, error):
            guard state.isTerminal else { return false }
            settle()
            switch state {
            // `finalized` normally arrives first and has already returned;
            // a `completed` reaching here is a job that published nothing
            // this client can fetch.
            case .completed: report.finished(nil, host)
            case .cancelled: report.failed("Cancelled")
            default: report.failed(error ?? "The render didn't finish.")
            }
            return true
        }
        return false
    }

    private func settleFromDetail(
        _ detail: ChainJobDetail, host: MoldHost.ID, report: Reporter
    ) {
        settle()
        switch detail.state {
        case .completed: report.finished(detail.galleryFilename, host)
        case .cancelled: report.failed("Cancelled")
        case .failed: report.failed(detail.error ?? "The render didn't finish.")
        default:
            // Not terminal, and the stream is gone: the job is still running
            // on a durable host and this client simply stopped watching.
            report.failed("Lost contact while rendering. "
                + "The job may still be running — check the Queue.")
        }
    }
}
