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
        let admission = BatchAdmission(requests: [draft.request(model: modelName)])
        PendingBatch.remember(admission.clientBatchId, host: host.id)

        run = .submitting
        runTask?.cancel()
        runTask = Task { [weak self] in
            do {
                let accepted = try await backend.submit(admission)
                self?.activeBatch = (accepted.id, host.id)
                await self?.follow(accepted, backend: backend, host: host.id)
            } catch {
                self?.run = .failed((error as? LocalizedError)?.errorDescription
                    ?? error.localizedDescription)
                PendingBatch.forget(admission.clientBatchId)
            }
        }
    }

    private func follow(_ initial: BatchStatus, backend: any MoldBackend,
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
            settle(try await backend.batchStatus(id: initial.id), host: host)
        } catch {
            // A dropped stream does NOT mean the work stopped: on a durable
            // host the job is still going to run.
            run = .failed("Lost contact while rendering. The job may still be running — check the Queue.")
        }
    }

    private func settle(_ status: BatchStatus, host: MoldHost.ID) {
        guard let child = status.children.first else { return }
        switch child.state {
        case .complete:
            if let result = child.result {
                run = .finished(result, host: host)
            }
            PendingBatch.forget(status.clientBatchId)
        case .failed, .cancelled:
            run = .failed(child.error ?? "The render didn't finish.")
            PendingBatch.forget(status.clientBatchId)
        default:
            if case let .running(_, progress) = run {
                run = .running(status, progress)
            } else {
                run = .running(status, nil)
            }
        }
    }

    /// Step progress and the denoise preview, which the events stream
    /// deliberately does not carry.
    private func pollPreview(_ status: BatchStatus, backend: any MoldBackend) -> Task<Void, Never> {
        Task { [weak self] in
            guard let jobId = status.children.first?.jobId else { return }
            while !Task.isCancelled {
                if let progress = try? await backend.jobPreview(jobId: jobId),
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
        Task { try? await backend.cancelBatch(id: active.id) }
        run = .idle
    }

    func dismissResult() { run = .idle }
}
