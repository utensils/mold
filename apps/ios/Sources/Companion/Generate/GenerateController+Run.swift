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
        guard case let .running(status, progress) = self, progress?.stall == nil,
              !status.children.contains(where: { $0.state == .cancelling }),
              let step = progress?.stageCurrent ?? progress?.step,
              let total = progress?.stageTotal ?? progress?.total, total > 0 else { return nil }
        return (step, total)
    }

    var stage: String? {
        guard case let .running(status, progress) = self else { return nil }
        if let stall = progress?.stall { return stall.statusMessage }
        if status.children.contains(where: { $0.state == .cancelling }) {
            return "Cancel requested; awaiting host confirmation."
        }
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

/// Admission has a finite background lifetime, separate from progress sockets.
/// Stop intent survives cancellation if the server has already accepted work.
struct PendingAdmission {
    let task: Task<Void, Never>
    var stopped = false
}

// Submitting and following (the Mac's `GenerateController+Run`, for a phone).
extension GenerateController {
    func generate() {
        guard blocker == nil, let host = target, let modelName else { return }
        if sequence.draft != nil {
            let captured = host
            let capturedInstallation = hosts.instanceID(of: host.id)
            let snapshot = draft
            let capturedRecipe = recipe
            let base = NativeSequenceSession.baseRequest(snapshot, model: modelName,
                maxIdentityPhotos: hosts.capabilities[host.id]?.maxIdentityPhotos ?? 0)
            sequence.submit(base: base, host: host, installation: capturedInstallation,
                backend: hosts.backend(for: host), current: { self.hosts.host(captured.id) == captured && self.hosts.instanceID(of: captured.id) == capturedInstallation }, prepare: {
                    let fitted = try await snapshot.fittingSource(recipe: capturedRecipe)
                    return NativeSequenceSession.baseRequest(fitted, model: modelName,
                        maxIdentityPhotos: self.hosts.capabilities[host.id]?.maxIdentityPhotos ?? 0)
                })
            return
        }
        if usesVideoDuration { generateDurationVideo(); return }
        let capturedInstance = hosts.instanceID(of: host.id)
        let ordinaryBackend = hosts.backend(for: host)
        let backend: any MoldBackend = if let http = ordinaryBackend as? HTTPBackend {
            http.checkingReferenceAuthority { [weak hosts] in
                await MainActor.run {
                    hosts?.host(host.id) == host && hosts?.instanceID(of: host.id) == capturedInstance
                }
            }
        } else { ordinaryBackend }
        let copies = min(draft.batchSize, referenceBatchLimit)
        let retained = retainedReuse.snapshot()
        let snapshot = draft
        let recipe = recipe
        let maxIdentityPhotos = hosts.capabilities[host.id]?.maxIdentityPhotos ?? 0
        let prompt = draft.prompt
        // Decided now, before the task: a second press right after this one
        // still queues, however the two tasks interleave.
        let followNow = activeBatch == nil && !run.isBusy
        if followNow {
            run = .submitting
            referencePreparation = draft.media.generationReferences.contains { $0.freshVideo != nil }
                ? "Preparing video on \(host.name)…" : nil
        }
        saveDraft()

        // A finite background task covers ONLY the upload and admission; once
        // the machine has the batch, it runs whether or not this app does
        // (`.claude/rules/mobile.md`). Ended on every path.
        var background = UIBackgroundTaskIdentifier.invalid
        background = UIApplication.shared.beginBackgroundTask(withName: "Generate") {
            UIApplication.shared.endBackgroundTask(background)
            background = .invalid
        }
        let token = UUID()
        if followNow { currentAdmissionID = token }
        let task = Task { [weak self] in
            defer {
                self?.finishAdmission(token)
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
                if self.currentAdmissionID == token { self.referencePreparation = nil }
                let active = ActiveBatch(id: accepted.id, clientBatchId: admission.clientBatchId,
                                         host: host.id, prompt: prompt, startedAt: .now)
                self.ledger.add(active)
                if background != .invalid {
                    UIApplication.shared.endBackgroundTask(background)
                    background = .invalid
                }
                if self.admissions[token]?.stopped == true {
                    self.cancelStoppedAdmission(active, host: host, backend: backend)
                    return
                }
                if self.currentAdmissionID == token { self.currentAdmissionID = nil }
                if self.activeBatch == nil, self.currentAdmissionID == nil {
                    self.activeBatch = active
                    self.run = .running(accepted, nil)
                    self.startFollowing(initial: accepted, acceptingBackend: backend)
                } else { self.queued.append(active) }
            } catch {
                guard !Task.isCancelled, let self, self.admissions[token]?.stopped == false else { return }
                if self.currentAdmissionID == token {
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
        admissions[token] = PendingAdmission(task: task)
    }

    private func finishAdmission(_ token: UUID) {
        admissions[token] = nil
        guard currentAdmissionID == token else { return }
        currentAdmissionID = nil
        referencePreparation = nil
        followNext()
    }

    private func cancelStoppedAdmission(_ active: ActiveBatch, host: MoldHost, backend: any MoldBackend) {
        // A fresh task can send cancellation even when the admission task was
        // cancelled. Keep the ledger until the machine confirms that request.
        Task { [weak self] in
            do {
                try await backend.cancelBatch(id: active.id)
                self?.ledger.remove(active.clientBatchId)
                self?.settled?(active, .idle)
            } catch { self?.hosts.report(host, doing: String(localized: "stop that render"), error) }
        }
    }

    func follow(_ initial: BatchStatus, active: ActiveBatch, backend: any MoldBackend) async {
        guard !Task.isCancelled, !followingSuspended, activeBatch == active else { return }
        // Settled while nobody was watching (the app was away): say so now.
        if BatchOutcome(settling: initial) != nil { return settle(initial, active: active) }
        run = .running(initial, nil)
        let preview = previewPoll(backend)
        defer { preview.cancel() }
        do {
            for try await status in backend.batchEvents(id: initial.id) {
                guard !Task.isCancelled, activeBatch == active else { return }
                settle(status, active: active)
                if status.isAtRest { return }
            }
            // The stream ended without a settled frame: READ it once, never
            // re-submit to find out.
            let status = try await backend.batchStatus(id: initial.id)
            guard !Task.isCancelled, activeBatch == active else { return }
            settle(status, active: active)
        } catch {
            guard !Task.isCancelled, activeBatch == active else { return }
            // A dropped stream does not mean the work stopped.
            run = .failed(String(localized: "Lost contact while rendering. The job may still be running — check the Queue."))
            settled?(active, run)
        }
    }

    private func previewPoll(_ backend: any MoldBackend) -> Task<Void, Never> {
        Task { [weak self] in
            while !Task.isCancelled {
                if let self, case let .running(status, _) = self.run,
                   let job = (status.children.first(where: { $0.state == .running || $0.state == .cancelling })
                       ?? status.children.first(where: { $0.state.isLive }))?.jobId,
                   let progress = try? await backend.jobPreview(jobId: job),
                   !Task.isCancelled, case let .running(current, _) = self.run, current.id == status.id {
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

    /// The next admitted batch is followed once the current one settles.
    func followNext() {
        guard !queued.isEmpty, activeBatch == nil else { return }
        activeBatch = queued.removeFirst()
        startFollowing()
    }

    private func finishFollowing(_ token: UUID) {
        guard followingID == token else { return }
        followingID = nil
        runTask = nil
    }

    private func startFollowing(initial: BatchStatus? = nil, acceptingBackend: (any MoldBackend)? = nil) {
        // Initial progress belongs to the accepting route, even if the user
        // edited this host's address or key while admission was in flight.
        guard !followingSuspended, let active = activeBatch,
              let backend = acceptingBackend ?? hosts.backend(for: active.host) else { return }
        let token = UUID()
        followingID = token
        runTask = Task { [weak self] in
            defer { self?.finishFollowing(token) }
            do {
                let status: BatchStatus
                if let initial { status = initial }
                else { status = try await backend.batchStatus(id: active.id) }
                guard !Task.isCancelled, self?.followingID == token, self?.activeBatch == active else { return }
                await self?.follow(status, active: active, backend: backend)
            } catch {
                guard !Task.isCancelled, let self, self.followingID == token, self.activeBatch == active else { return }
                self.run = .failed(String(localized: "Lost contact while rendering. The job may still be running — check the Queue."))
            }
        }
    }

    /// Keep the admitted batch while iOS suspends sockets in the background.
    func suspendFollowing() {
        followingSuspended = true
        videoPlanner.clear()
        videoJobs.suspend()
        runTask?.cancel()
        followingID = nil
        runTask = nil
    }

    /// A foreground/reconnect read follows the existing batch, never resubmits.
    func resumeFollowing() {
        followingSuspended = false
        guard runTask == nil else { return }
        startFollowing()
    }

    /// Stops the render on screen, on its machine. Queued batches keep their
    /// place; "Stop Everything" cancels them too.
    func stop(everything: Bool = false) {
        let targets = [activeBatch].compactMap(\.self) + (everything ? queued : [])
        let pending = everything ? Array(admissions.keys) : [currentAdmissionID].compactMap(\.self)
        for token in pending {
            admissions[token]?.stopped = true
            admissions[token]?.task.cancel()
        }
        currentAdmissionID = nil
        runTask?.cancel()
        followingID = nil
        runTask = nil
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

// Only verified, vacant retained roles can satisfy a required-input check.
extension GenerateController {
    var retainedReferenceFields: Set<RetainedSourceMedia.Field> {
        guard !retainedReuse.probing, let authority = retainedReuse.snapshot(), let modelName else { return [] }
        let request = RenderRequest.one(draft, model: modelName,
            maxIdentityPhotos: target.flatMap { hosts.capabilities[$0.id]?.maxIdentityPhotos } ?? 0)
        return Set(RetainedSourceMedia.members(authority.members, forHydrating: request).compactMap {
            RetainedSourceMedia.field(forRole: $0.role)
        })
    }
}
