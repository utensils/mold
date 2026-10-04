import Foundation
import MoldClient
import UIKit

/// Where a render is. `running` carries the batch and its latest preview.
enum RunState: Equatable {
    case idle
    case submitting
    case running(BatchStatus, JobProgress?)
    case finished(BatchOutcome, host: MoldHost.ID)
    case failed(String)

    var isBusy: Bool {
        switch self {
        case .submitting, .running: true
        case .idle, .finished, .failed: false
        }
    }

    var steps: (done: Int, total: Int)? {
        guard case let .running(_, progress) = self, let step = progress?.step,
              let total = progress?.total, total > 0 else { return nil }
        return (step, total)
    }

    var stage: String? {
        guard case let .running(_, progress) = self else { return nil }
        return progress?.stage
    }

    var previewData: Data? {
        guard case let .running(_, progress) = self else { return nil }
        return progress?.previewData
    }
}

/// A batch the machine admitted: enough to follow it, stop it, or find it
/// again after the app was away.
struct ActiveBatch: Equatable, Codable {
    let id: String
    let clientBatchId: String
    let host: MoldHost.ID
    let prompt: String
    let startedAt: Date
}

// Submitting and following (the Mac's `GenerateController+Run`, for a phone).
extension GenerateController {
    func generate() {
        guard blocker == nil, let host = target, let modelName else { return }
        let backend = hosts.backend(for: host)
        let copies = min(draft.batchSize, referenceBatchLimit)
        let retained = retainedReuse.snapshot()
        let snapshot = draft
        let recipe = recipe
        let maxIdentityPhotos = hosts.capabilities[host.id]?.maxIdentityPhotos ?? 0
        let prompt = draft.prompt
        // Decided now, before the task: a second press right after this one
        // still queues, however the two tasks interleave.
        let followNow = !run.isBusy
        if followNow { run = .submitting }
        saveDraft()

        // A finite background task covers ONLY the upload and admission; once
        // the machine has the batch, it runs whether or not this app does
        // (`.claude/rules/mobile.md`). Ended on every path.
        var background = UIBackgroundTaskIdentifier.invalid
        background = UIApplication.shared.beginBackgroundTask(withName: "Generate") {
            UIApplication.shared.endBackgroundTask(background)
            background = .invalid
        }
        let task = Task { [weak self] in
            defer {
                if background != .invalid { UIApplication.shared.endBackgroundTask(background) }
            }
            do {
                let prepared = try await snapshot.fittingSource(recipe: recipe)
                let requests = RenderRequest.batch(
                    prepared, model: modelName, copies: max(1, copies),
                    randomBase: .random(in: 0 ... UInt64(UInt32.max)), maxIdentityPhotos: maxIdentityPhotos)
                var admission = BatchAdmission(requests: requests)
                if let retained, let origin = self?.hosts.backend(for: retained.origin) {
                    admission = try await RetainedSourceMedia.hydrated(
                        admission, filename: retained.filename, members: retained.members,
                        sameHost: retained.origin == host.id, origin: origin, target: backend)
                } else if retained != nil {
                    throw MoldClientError.unreachable("The machine that kept the source media isn't connected.")
                }
                try Task.checkCancellation()
                let accepted = try await backend.submit(admission)
                guard let self else { return }
                let active = ActiveBatch(id: accepted.id, clientBatchId: admission.clientBatchId,
                                         host: host.id, prompt: prompt, startedAt: .now)
                self.ledger.add(active)
                if background != .invalid {
                    UIApplication.shared.endBackgroundTask(background)
                    background = .invalid
                }
                guard followNow else { self.queued.append(active); return }
                self.activeBatch = active
                await self.follow(accepted, active: active, backend: backend)
            } catch {
                guard !Task.isCancelled, let self else { return }
                if followNow {
                    let sentence: String?
                    if case let .http(_, code, _)? = error as? MoldClientError {
                        sentence = RetainedSourceMedia.refusalSentence(for: code)
                    } else { sentence = nil }
                    self.run = .failed(sentence ?? error.failureSentence)
                } else {
                    self.hosts.report(host, doing: String(localized: "queue that render"), error)
                }
            }
        }
        if followNow { runTask = task }
    }

    func follow(_ initial: BatchStatus, active: ActiveBatch, backend: any MoldBackend) async {
        // Settled while nobody was watching (the app was away): say so now.
        if BatchOutcome(settling: initial) != nil { return settle(initial, active: active) }
        run = .running(initial, nil)
        let preview = previewPoll(backend)
        defer { preview.cancel() }
        do {
            for try await status in backend.batchEvents(id: initial.id) {
                guard !Task.isCancelled else { return }
                settle(status, active: active)
                if status.isAtRest { return }
            }
            // The stream ended without a settled frame: READ it once, never
            // re-submit to find out.
            settle(try await backend.batchStatus(id: initial.id), active: active)
        } catch {
            guard !Task.isCancelled else { return }
            // A dropped stream does not mean the work stopped.
            run = .failed(String(localized: "Lost contact while rendering. The job may still be running — check the Queue."))
            settled?(active, run)
        }
    }

    private func previewPoll(_ backend: any MoldBackend) -> Task<Void, Never> {
        Task { [weak self] in
            while !Task.isCancelled {
                if let self, case let .running(status, _) = self.run,
                   let job = status.children.first(where: { $0.state.isLive })?.jobId ?? status.children.first?.jobId,
                   let progress = try? await backend.jobPreview(jobId: job),
                   case let .running(current, _) = self.run {
                    self.run = .running(current, progress)
                }
                try? await Task.sleep(for: .milliseconds(700))
            }
        }
    }

    func settle(_ status: BatchStatus, active: ActiveBatch) {
        if let outcome = BatchOutcome(settling: status) {
            run = outcome.results.isEmpty
                ? .failed(outcome.failures.first ?? String(localized: "The render didn't finish."))
                : .finished(outcome, host: active.host)
            ledger.remove(active.clientBatchId)
            activeBatch = nil
            settled?(active, run)
            followNext()
        } else if case let .running(_, progress) = run {
            run = .running(status, progress)
        } else {
            run = .running(status, nil)
        }
    }

    /// The next queued batch takes the canvas once this one has settled.
    func followNext() {
        guard !queued.isEmpty, !run.isBusy || activeBatch == nil else { return }
        let next = queued.removeFirst()
        guard let host = hosts.host(next.host) else { return }
        let backend = hosts.backend(for: host)
        activeBatch = next
        runTask = Task { [weak self] in
            guard let status = try? await backend.batchStatus(id: next.id) else { return }
            await self?.follow(status, active: next, backend: backend)
        }
    }

    /// The app is going away: stop listening (iOS would freeze the socket
    /// and it would look alive while saying nothing), but keep the batch --
    /// the machine keeps rendering, and `resumeFollowing` picks it up.
    func suspendFollowing() {
        runTask?.cancel()
        runTask = nil
    }

    /// Back in the foreground: read where the batch on screen got to, then
    /// settle it or follow it again. Never re-submits.
    func resumeFollowing() {
        guard runTask == nil, let active = activeBatch, let host = hosts.host(active.host) else { return }
        let backend = hosts.backend(for: host)
        runTask = Task { [weak self] in
            guard let status = try? await backend.batchStatus(id: active.id) else { return }
            await self?.follow(status, active: active, backend: backend)
        }
    }

    /// Stops the render on screen, on its machine. Queued batches keep their
    /// place; "Stop Everything" cancels them too.
    func stop(everything: Bool = false) {
        let targets = [activeBatch].compactMap(\.self) + (everything ? queued : [])
        runTask?.cancel()
        run = .idle
        activeBatch = nil
        if everything { queued = [] }
        for batch in targets {
            settled?(batch, .idle)
            ledger.remove(batch.clientBatchId)
            guard let host = hosts.host(batch.host) else { continue }
            Task { try? await hosts.backend(for: host).cancelBatch(id: batch.id) }
        }
        if !everything { followNext() }
    }
}
