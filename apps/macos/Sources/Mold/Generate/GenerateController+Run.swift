import Foundation
import MoldClient

// Submitting a render and following it to settlement.
@MainActor
extension GenerateController {

    /// Submits the draft. When nothing is being followed, follows this one
    /// to settlement; when a batch is already on screen (M8 decision 8), the
    /// new admission is still sent -- the host's queue is durable, so a
    /// second press must not waste the reservation the first one already
    /// made -- and waits in `queued` instead of displacing what is showing.
    ///
    /// The client batch id is minted and PERSISTED BEFORE the request goes
    /// out either way. If the response is lost, the work is recovered by
    /// asking the host about that id -- submitting again would render twice.
    func submit(on host: MoldHost, backend: any MoldBackend) {
        guard let modelName else { return }
        // The Batch control already caps at `maxBatchOutputs`; this is a belt
        // on the one path a stale draft could still exceed it.
        let copies = min(draft.batchSize, hosts.capabilities(of: host)?.maxBatchOutputs ?? draft.batchSize)
        let admission = BatchAdmission(requests: draft.requests(
            model: modelName, copies: copies,
            randomBase: .random(in: 0 ... UInt64(UInt32.max)),
            maxIdentityPhotos: hosts.capabilities(of: host)?.maxIdentityPhotos ?? 0
        ))
        PendingBatch.remember(admission.clientBatchId, host: host.id)

        // Decided HERE, synchronously, before the `Task` below is even
        // scheduled -- so a second `submit()` called right after this one
        // still queues correctly no matter how the two `Task`s interleave.
        let followingNow = !run.isBusy
        if followingNow {
            run = .submitting
            runTask?.cancel()
        }
        let task = Task { [weak self] in
            do {
                let accepted = try await backend.submit(admission)
                guard let self else { return }
                let active = ActiveBatch(
                    id: accepted.id, clientBatchId: admission.clientBatchId,
                    host: host.id, admitted: accepted)
                if followingNow {
                    self.activeBatch = active
                    await self.follow(accepted, backend: backend, host: host.id)
                } else {
                    self.queued.append(active)
                }
            } catch {
                PendingBatch.forget(admission.clientBatchId)
                guard let self else { return }
                if followingNow {
                    self.run = .failed(error.sentence)
                } else {
                    // The render on screen is unaffected by a second one
                    // failing to be admitted -- report it, don't replace `run`.
                    self.hosts.report(error, on: host.id, doing: "queue that render")
                }
            }
        }
        if followingNow { runTask = task }
    }

    // Not `private`: `GenerateController+Recover` re-enters here for a batch
    // still live after a relaunch, and `GenerateController+Queue` re-enters
    // here for the next queued batch.
    func follow(_ initial: BatchStatus, backend: any MoldBackend,
                host: MoldHost.ID) async {
        run = .running(initial, nil)
        let preview = pollPreview(initial, backend: backend)
        defer { preview.cancel() }

        do {
            for try await status in backend.batchEvents(id: initial.id) {
                guard !Task.isCancelled else { return }
                settle(status, host: host)
                // At rest, not merely settled: a hold ends the follow too.
                if status.isAtRest { return }
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
            // Cancelling is not losing contact -- `stop()` already set
            // `.idle` and reported anything worth reporting. Without this
            // guard, the task's own cancellation raced that assignment and
            // overwrote it with a failure on every Stop.
            guard !Task.isCancelled else { return }
            // A dropped stream does NOT mean the work stopped: on a durable
            // host the job is still going to run. Not a settlement -- the
            // queue does not advance on its own here.
            run = .failed("Lost contact while rendering. The job may still be running — check the Queue.")
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

    func dismissResult() { run = .idle }
}
