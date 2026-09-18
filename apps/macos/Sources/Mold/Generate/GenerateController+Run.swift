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
    ///
    /// `routing` is the PANE's answer -- it needs the recipe. A render past
    /// the clip size is not a batch at all (`ChainSubmission`).
    /// `retained` is the print this draft came from, resolved INSIDE the task
    /// below against the requests going out (`RetainedMediaHydration`).
    func submit(on host: MoldHost, backend: any MoldBackend,
                routing: ChainRouting.Decision = .single(),
                retained: RetainedMediaHydration? = nil) {
        guard let modelName else { return }
        // The Batch control already caps at `maxBatchOutputs`; this is a belt
        // on the one path a stale draft could still exceed it.
        let copies = min(draft.batchSize, hosts.capabilities(of: host)?.maxBatchOutputs ?? draft.batchSize)
        let built = draft.requests(
            model: modelName, copies: copies,
            randomBase: .random(in: 0 ... UInt64(UInt32.max)),
            maxIdentityPhotos: hosts.capabilities(of: host)?.maxIdentityPhotos ?? 0)
        if ChainSubmission.take(routing, requests: built,
                                on: host, backend: backend, controller: self) { return }
        let admission = BatchAdmission(requests: built)
        PendingBatch.remember(admission.clientBatchId, host: host.id)

        // Decided HERE, synchronously, before the `Task` below is even
        // scheduled -- so a second `submit()` called right after this one
        // still queues correctly no matter how the two `Task`s interleave.
        let followingNow = !run.isBusy
        if followingNow {
            run = .submitting
            // NEVER while a POST is unanswered. `runTask` is the whole
            // submit-and-follow task, so cancelling it there would abort a
            // request the host has very likely already admitted -- leaving a
            // render nobody holds the id of. Only a FOLLOW is interruptible
            // (finding 02#2, one step further along).
            if !submissions.hasUnansweredPost {
                runTask?.cancel()
            }
            submissions.begin(admission.clientBatchId)
        }
        let task = Task { [weak self] in
            do {
                let accepted = try await backend.submit(
                    RetainedMedia.hydrated(admission, with: retained, on: host, backend: backend))
                guard let self else { return }
                let active = ActiveBatch(
                    id: accepted.id, clientBatchId: admission.clientBatchId,
                    host: host.id, admitted: accepted)
                guard followingNow else { self.queued.append(.batch(active)); return }
                // Stop, or a second press, may have happened while this was in
                // the air. Only now is there an id the host would recognise.
                switch self.submissions.land(admission.clientBatchId) {
                case .cancel:
                    PendingBatch.forget(admission.clientBatchId)
                    self.cancelOnItsMachine(active)
                    // Only if nothing has taken the canvas since: a withdrawn
                    // render must not displace the one that replaced it.
                    if !self.run.isBusy { self.followNext() }
                case .queue:
                    self.queued.append(.batch(active))
                case .follow:
                    self.activeBatch = active
                    await self.follow(accepted, backend: backend, host: host.id)
                }
            } catch {
                guard let self else { return }
                // A CANCELLED post may well have reached the host. Forgetting
                // its recovery record would orphan exactly the render the
                // fence exists to keep findable.
                if !(error is CancellationError) {
                    PendingBatch.forget(admission.clientBatchId)
                }
                guard followingNow else {
                    // The render on screen is unaffected by a second one
                    // failing to be admitted -- report it, don't replace `run`.
                    self.hosts.report(error, on: host.id, doing: "queue that render")
                    return
                }
                switch self.submissions.land(admission.clientBatchId) {
                case .follow: self.run = .failed(error.sentence)
                // Stop already answered for this one; the queue still moves,
                // unless something has taken the canvas since.
                case .cancel: if !self.run.isBusy { self.followNext() }
                // Superseded: a failure here must not replace what took the
                // canvas from it.
                case .queue: break
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
        let preview = PreviewPoll.follow(initial, backend: backend, of: self)
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

}
