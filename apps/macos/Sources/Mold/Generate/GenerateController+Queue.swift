import Foundation
import MoldClient

/// M8 decision 8: Generate never turns into Stop. The host's queue is
/// durable, so a press while another run is on screen ADMITS the work
/// (`submit(on:backend:routing:)`) instead of being refused, and this is where
/// it waits and advances. A chain waits in the SAME line (`QueuedRun`).
@MainActor
extension GenerateController {
    /// What the capsule's caption names -- "2 more queued".
    var queuedCount: Int { queued.count }

    /// Not `private`: `GenerateController+Run.follow(_:backend:host:)` calls
    /// this at both places a batch's status can land.
    func settle(_ status: BatchStatus, host: MoldHost.ID) {
        if let outcome = BatchOutcome(settling: status) {
            // Settled with any result at all is shown; settled with none is
            // the failure -- the fence is held until every child is in,
            // never on the first one to arrive.
            if outcome.results.isEmpty {
                run = .failed(outcome.failures.first ?? "The render didn't finish.")
            } else {
                run = .finished(outcome, host: host)
            }
            PendingBatch.forget(status.clientBatchId)
            // Settled, so it is no longer the batch Stop would cancel.
            activeBatch = nil
            // M8 decision 8, with the beat made REAL: the next batch takes
            // the canvas once this outcome has actually been drawn on it (or
            // the handoff's grace runs out), never merely in a later turn of
            // the same render pass -- see `ResultHandoff` (finding 02#9).
            handoff.hold { [weak self] in self?.followNext() }
        } else if case let .running(_, progress) = run {
            run = .running(status, progress)
        } else {
            run = .running(status, nil)
        }
    }

    /// Pops the head of `queued`, if there is one, and follows it in place
    /// of whatever the canvas was just showing. Called the instant the
    /// followed batch settles (`settle(_:host:)` above) or is stopped
    /// (`stop()` below) -- never before.
    ///
    /// The connection is resolved from the BATCH's own machine, never
    /// carried over from the one just followed: the Machine control can be
    /// moved between two presses, so two queued batches can belong to two
    /// machines. A batch whose machine has since been removed is dropped --
    /// there is nothing left to follow it on.
    func followNext() {
        while !queued.isEmpty {
            let next = queued.removeFirst()
            guard let backend = hosts.backend(for: next.host) else {
                RunQueueing.forget(next)
                continue
            }
            switch next {
            case let .batch(batch):
                activeBatch = batch
                runTask = Task { [weak self] in
                    await self?.follow(batch.admitted, backend: backend, host: batch.host)
                }
            case let .chain(admitted):
                // A chain job the host already holds: re-attaching to it IS
                // following it, exactly as it is after a relaunch.
                chain.reattach(jobId: admitted.jobId, stageCount: admitted.stageCount,
                               on: admitted.host, backend: backend,
                               report: ChainSubmission.reporter(for: self))
            }
            return
        }
    }

    /// Stops what is on screen, then moves on to whatever is next in `queued`.
    func stop() {
        handoff.cancel()
        // A chain is cancelled through its OWN route, never the queue.
        if chain.stop(backend: { [hosts] in hosts.backend(for: $0) }) {
            run = .idle; followNext(); return
        }
        // Stop pressed while an admission is still in the air. `runTask` is
        // deliberately NOT cancelled: the POST has very likely already reached
        // the host, and killing the task here would leave that batch rendering
        // with nobody holding its id and no recovery record to find it by.
        // The submit task cancels the id the host returns (finding 02#2).
        if submissions.requestStop() {
            run = .idle
            // Anything already waiting starts now rather than sitting behind
            // a POST the user has withdrawn; that POST's own landing checks
            // `run.isBusy` before it advances the queue again.
            followNext()
            return
        }
        guard let active = activeBatch else {
            // A first-ever render that has not been admitted yet: there is
            // nothing to cancel, but Stop must still leave the button alone.
            run = .idle
            return
        }
        runTask?.cancel()
        runTask = nil
        // Cancelled by the user, not lost: nothing to recover on relaunch.
        PendingBatch.forget(active.clientBatchId)
        cancelOnItsMachine(active)
        activeBatch = nil
        run = .idle
        followNext()
    }

    /// Stops everything this pane admitted: every batch still waiting in
    /// `queued`, then the one on screen.
    func stopAll() {
        for waiting in queued { RunQueueing.withdraw(waiting, on: self) }
        queued.removeAll()
        stop()
    }

    /// Not `private`: `GenerateController+Run` cancels the batch a Stop
    /// pressed during `.submitting` was aimed at, once the host has named it.
    func cancelOnItsMachine(_ batch: ActiveBatch) {
        guard let backend = hosts.backend(for: batch.host) else { return }
        Task { [weak self] in
            do { try await backend.cancelBatch(id: batch.id) }
            catch { self?.hosts.report(error, on: batch.host, doing: "cancel that render") }
        }
    }
}
