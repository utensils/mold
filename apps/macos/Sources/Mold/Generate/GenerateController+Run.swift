import Foundation
import MoldClient

// Submitting a render and following it to settlement.
@MainActor
extension GenerateController {


    /// Submits the draft and follows it to settlement.
    ///
    /// The client batch id is minted and PERSISTED BEFORE the request goes
    /// out. If the response is lost, the work is recovered by asking the host
    /// about that id -- submitting again would render twice.
    func submit(on host: MoldHost, backend: any MoldBackend) {
        guard let modelName, !run.isBusy else { return }
        // The Batch control already caps at `maxBatchOutputs`; this is a belt
        // on the one path a stale draft could still exceed it.
        let copies = min(draft.batchSize, hosts.capabilities(of: host)?.maxBatchOutputs ?? draft.batchSize)
        let admission = BatchAdmission(requests: draft.requests(
            model: modelName, copies: copies,
            randomBase: .random(in: 0 ... UInt64(UInt32.max)),
            maxIdentityPhotos: hosts.capabilities(of: host)?.maxIdentityPhotos ?? 0
        ))
        PendingBatch.remember(admission.clientBatchId, host: host.id)

        run = .submitting
        runTask?.cancel()
        runTask = Task { [weak self] in
            do {
                let accepted = try await backend.submit(admission)
                self?.activeBatch = (accepted.id, admission.clientBatchId, host.id)
                await self?.follow(accepted, backend: backend, host: host.id)
            } catch {
                self?.run = .failed(error.sentence)
                PendingBatch.forget(admission.clientBatchId)
            }
        }
    }

    // Not `private`: `GenerateController+Recover` re-enters here for a batch
    // still live after a relaunch.
    func follow(_ initial: BatchStatus, backend: any MoldBackend,
                host: MoldHost.ID) async {
        run = .running(initial, nil)
        let preview = pollPreview(initial, backend: backend)
        defer { preview.cancel() }

        do {
            for try await status in backend.batchEvents(id: initial.id) {
                guard !Task.isCancelled else { return }
                settle(status, host: host)
                if status.isSettled { return }
            }
            // The stream ended without a settled frame; READ the status once
            // rather than leaving the pane spinning. Re-submitting to find out
            // what happened would be asking for a second render.
            //
            // This used to be the ONLY way a render finished, because
            // `AsyncBytes.lines` drops the blank line that terminates an SSE
            // frame and `batchEvents` therefore yielded nothing at all. The
            // picture appeared when the server closed the stream rather than
            // when the batch settled. See `LineAccumulator`.
            settle(try await backend.batchStatus(id: initial.id), host: host)
        } catch {
            // Cancelling is not losing contact -- `cancel()` already set
            // `.idle` and reported anything worth reporting. Without this
            // guard, the task's own cancellation raced that assignment and
            // overwrote it with a failure on every Stop.
            guard !Task.isCancelled else { return }
            // A dropped stream does NOT mean the work stopped: on a durable
            // host the job is still going to run.
            run = .failed("Lost contact while rendering. The job may still be running — check the Queue.")
        }
    }

    private func settle(_ status: BatchStatus, host: MoldHost.ID) {
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
        } else if case let .running(_, progress) = run {
            run = .running(status, progress)
        } else {
            run = .running(status, nil)
        }
    }

    /// Step progress and the denoise preview, which the events stream
    /// deliberately does not carry.
    ///
    /// With several children, one settles while the others keep running --
    /// so this re-reads the current run's status every tick and follows
    /// whichever child is still live, falling back to the last one it had
    /// rather than going quiet the moment the first child finishes.
    private func pollPreview(_ status: BatchStatus, backend: any MoldBackend) -> Task<Void, Never> {
        var target = status.children.first?.jobId
        return Task { [weak self] in
            while !Task.isCancelled {
                if case let .running(current, _) = self?.run {
                    target = current.children.first { $0.state.isLive }?.jobId ?? target
                }
                if let jobId = target,
                   let progress = try? await backend.jobPreview(jobId: jobId),
                   case let .running(current, _) = self?.run {
                    self?.run = .running(current, progress)
                }
                try? await Task.sleep(for: .milliseconds(700))
            }
        }
    }

    func cancel(backend: any MoldBackend) {
        guard let active = activeBatch else { return }
        runTask?.cancel()
        runTask = nil
        // Cancelled by the user, not lost: nothing to recover on relaunch.
        PendingBatch.forget(active.clientBatchId)
        Task { [weak self] in
            do { try await backend.cancelBatch(id: active.id) }
            catch { self?.hosts.report(error, on: active.host, doing: "cancel that render") }
        }
        run = .idle
    }

    func dismissResult() { run = .idle }
}
