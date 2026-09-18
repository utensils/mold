import Foundation
import MoldClient

/// The one ephemeral chain job this pane can have in flight.
///
/// A SEPARATE type rather than more `GenerateController`: the controller is at
/// its size budget, and a chain is its own lifecycle -- a different id, a
/// different stream, a different cancel route. The controller owns one of
/// these and asks it three questions.
///
/// An auto-chained one-shot is not an authored sequence. It exists because the
/// checkpoint cannot render that many frames in one pass, it is hidden from
/// the host's own sequence listings (`ephemeral: true`), and it settles into
/// ONE print.
@MainActor
@Observable
final class ChainRun {
    /// The job on screen, or nil. Written before the follow starts so Stop
    /// always has something to cancel.
    ///
    /// Everything below is the type's own state, deliberately not `private`:
    /// `private` does not cross files for one type, and `ChainRun+Lifecycle`
    /// and `ChainRun+Follow` are this type split for size, not other callers.
    var active: ChainProgress?
    var task: Task<Void, Never>?
    var host: MoldHost.ID?
    /// Where the followed job reports to, held so Resume can push the
    /// cleared paused state onto the same canvas.
    var report: Reporter?
    /// The first reconnect delay. A constructor parameter so a test pins the
    /// BEHAVIOUR rather than waiting out a constant.
    let firstBackoff: Duration

    init(firstBackoff: Duration = .seconds(1)) { self.firstBackoff = firstBackoff }

    /// `2^n`, capped at 32 s -- the same shape `HostStore+Events.watch` uses,
    /// because it is the same question: a machine that is briefly unreachable
    /// is still the machine the work is on.
    func backoff(_ attempt: Int) -> Duration {
        let capped = Swift.min(attempt, 5)
        return firstBackoff * (1 << capped)
    }
    /// Which start owns this type's state. `creating`, `withdrawn`, `active`,
    /// `host` and `task` are all instance state shared by every `start()`, so
    /// without a token an OLDER task's landing clears the flags of the one
    /// after it -- and the Stop aimed at the newer chain then found nothing to
    /// stop and let it render to completion.
    var generation = 0
    /// A create is in the air. Set SYNCHRONOUSLY by `start`, so a Stop
    /// pressed in the same turn is answered even though there is no job id
    /// to cancel yet.
    var creating = false
    /// Stop was pressed while the create was unanswered. The task is
    /// deliberately NOT cancelled there -- the POST has very likely already
    /// reached the host, and killing it would leave a chain rendering with
    /// nobody holding its id (`SubmissionFence`'s lesson, one door along).
    /// The landing reads this and withdraws the job the host just minted.
    var withdrawn = false

    /// What the follow reports back. The controller supplies these rather
    /// than this type reaching into it, so the whole lifecycle is testable
    /// without a pane.
    struct Reporter {
        let progress: (ChainProgress) -> Void
        /// The stitched print's gallery filename, or nil when the job
        /// finished without publishing one this client can fetch.
        let finished: (String?, MoldHost.ID) -> Void
        let failed: (String) -> Void
    }

    /// Creates the job and follows it.
    ///
    /// The operation id is minted and REMEMBERED before the request goes out,
    /// exactly as a batch's client id is: replaying it returns the job the
    /// host already holds rather than starting a second render.
    func start(
        _ request: AutoChainRequest, stageCount: Int,
        on host: MoldHost.ID, backend: any MoldBackend, report: Reporter
    ) {
        task?.cancel()
        generation += 1
        let mine = generation
        self.host = host
        creating = true
        withdrawn = false
        self.report = report
        let operationId = UUID().uuidString
        task = Task { [weak self] in
            let created: CreateChainJobResponse
            do {
                created = try await backend.createChainJob(request, operationId: operationId)
            } catch {
                guard let self, self.generation == mine else { return }
                self.creating = false
                // A withdrawn create that then failed needs no sentence: the
                // user already asked for it to stop.
                guard !self.withdrawn, !Task.isCancelled else { return }
                self.active = nil
                report.failed(error.sentence)
                return
            }
            // Gone, superseded, or withdrawn: the job is REAL on the host and
            // nobody is going to watch it, so cancel the id it just named
            // rather than leaving a GPU rendering for no one. This is the
            // whole reason Stop does not cancel the task above.
            guard let self, self.generation == mine else {
                try? await backend.cancelChainJob(id: created.jobId)
                return
            }
            self.creating = false
            guard !self.withdrawn, !Task.isCancelled else {
                self.withdrawn = false
                try? await backend.cancelChainJob(id: created.jobId)
                return
            }
            let progress = ChainProgress(jobId: created.jobId, stageCount: stageCount)
            self.active = progress
            PendingChain.remember(created.jobId, host: host)
            report.progress(progress)
            await self.follow(created.jobId, on: host, backend: backend, report: report)
        }
    }

    /// Re-attaches to a job this app admitted and then lost -- a relaunch, or
    /// a dropped stream. The job is durable, so following it again is the
    /// whole recovery.
    func reattach(
        jobId: String, stageCount: Int, on host: MoldHost.ID,
        backend: any MoldBackend, report: Reporter,
        currentStage: Int = 1, isPaused: Bool = false
    ) {
        task?.cancel()
        generation += 1
        self.host = host
        creating = false
        withdrawn = false
        self.report = report
        var progress = ChainProgress(jobId: jobId, stageCount: stageCount,
                                     currentStage: Swift.max(currentStage, 1))
        progress.isPaused = isPaused
        active = progress
        report.progress(progress)
        task = Task { [weak self] in
            await self?.follow(jobId, on: host, backend: backend, report: report)
        }
    }
}
