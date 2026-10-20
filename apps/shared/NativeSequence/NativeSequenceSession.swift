import Foundation
import MoldClient
import Observation

/// Native presentation of durable authored jobs. Ordinary batches and automatic
/// long clips retain their existing controllers and wire contracts.
@MainActor @Observable
final class NativeSequenceSession {
    struct Pending: Codable, Equatable, Identifiable {
        let id: String
        let host: UUID
        let origin: String
        let installation: String?
    }
    struct Job: Identifiable {
        let jobId: String
        var id: String { "\(host.id.uuidString):\(jobId)" }
        let host: MoldHost
        let installation: String?
        var detail: ChainJobDetail?
        var message: String
        var cancelling = false
    }
    static let pendingKey = "nativeAuthoredChainJobsV1"
    struct SavedDraft: Codable {
        struct Clip: Codable { let id: UUID; let prompt: String; let frames: Int; let transition: String }
        struct Source: Codable { let host: UUID; let origin: String; let installation: String; let job: String; let take: Int; let filename: String; var hadReferences: Bool? }
        let clips: [Clip]
        let source: Source?
    }
    var draft: SequenceDraft? { didSet {
        if oldValue != draft { generation += 1; validatedPlan = nil; preparing = false; saveDraft() }
    } }
    private var savedSource: SavedDraft.Source?
    var restoringSource: Bool { savedSource != nil && continuation == nil }

    var continuation: ChainContinuationSource?
    var continuationHost: MoldHost?
    var continuationTitle: String?
    var message: String?
    var historyNotice: String?
    var hasVisibleReferences = false
    @ObservationIgnored var restoreReferences: (([GenerationReference]) -> Void)?
    var needsReferenceDecision: Bool { savedSource?.hadReferences == true && !hasVisibleReferences }
    func restoreOriginalReferences() {
        guard let refs = continuation?.script.stage.last?.generationReferences, !refs.isEmpty else {
            message = "The original references are unavailable. Reattach the inputs or continue without references."; return
        }
        restoreReferences?(refs)
    }
    func continueWithoutReferences() {
        savedSource?.hadReferences = false; saveDraft()
    }
    var preparing = false
    var validatedPlan: ChainValidationResponse?
    var jobs: [Job] = []
    private(set) var pending: [Pending]
    @ObservationIgnored private let defaults: UserDefaults
    @ObservationIgnored private var tasks: [String: Task<Void, Never>] = [:]
    @ObservationIgnored private var followTokens: [String: UUID] = [:]
    @ObservationIgnored private var cancelledBeforeAcceptance: Set<String> = []
    @ObservationIgnored private var generation = 0
    @ObservationIgnored private var stopped: Set<String> = []
    @ObservationIgnored private var authorities: [String: @MainActor () -> Bool] = [:]

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
        pending = defaults.data(forKey: Self.pendingKey).flatMap {
            try? JSONDecoder().decode([Pending].self, from: $0)
        } ?? []
        if let saved = defaults.data(forKey: "nativeSequenceDraftV1").flatMap({ try? JSONDecoder().decode(SavedDraft.self, from: $0) }),
           !saved.clips.isEmpty, saved.clips.count <= 16 {
            var restored = SequenceDraft(prompt: saved.clips[0].prompt, frames: saved.clips[0].frames)
            restored.stages = saved.clips.map { clip in
                var stage = SequenceDraft.Stage(prompt: clip.prompt, frames: clip.frames, transition: clip.transition)
                stage.id = clip.id
                return stage
            }
            draft = restored; savedSource = saved.source
        }
    }
    private func saveDraft() {
        guard let draft else { defaults.removeObject(forKey: "nativeSequenceDraftV1"); return }
        let saved = SavedDraft(clips: draft.stages.map { .init(id: $0.id, prompt: $0.prompt, frames: $0.frames, transition: $0.transition) }, source: savedSource)
        if let data = try? JSONEncoder().encode(saved) { defaults.set(data, forKey: "nativeSequenceDraftV1") }
    }

    func remember(_ record: Pending) throws {
        var next = pending.filter { $0.id != record.id || $0.host != record.host }
        next.append(record)
        defaults.set(try JSONEncoder().encode(next), forKey: Self.pendingKey)
        guard defaults.synchronize() else { throw CocoaError(.fileWriteUnknown) }
        pending = next
    }
    func forget(_ id: String, host: UUID? = nil) {
        let next = pending.filter { $0.id != id || (host != nil && $0.host != host) }
        guard let data = try? JSONEncoder().encode(next) else { return }
        defaults.set(data, forKey: Self.pendingKey)
        pending = next
    }
    func begin(prompt: String, frames: Int) {
        generation += 1
        savedSource = nil
        draft = SequenceDraft(prompt: prompt, frames: frames)
        draft?.append()
        continuation = nil; continuationHost = nil; continuationTitle = nil; message = nil
    }
    func startFresh() {
        generation += 1
        savedSource = nil
        draft = nil; continuation = nil; continuationHost = nil; continuationTitle = nil
        message = nil; preparing = false
    }

    /// Model changes use that recipe's advertised frame grid, as ordinary
    /// duration controls do. Off-grid values must not become uneditable.
    func fitLengths(minimum: Int, maximum: Int, step: Int) {
        guard var next = draft, minimum > 0, maximum >= minimum else { return }
        let stride = max(1, step)
        var changed = false
        for index in next.stages.indices {
            let original = next.stages[index].frames
            let bounded = min(maximum, max(minimum, original))
            let fitted = minimum + ((bounded - minimum) / stride) * stride
            if original != fitted { next.stages[index].frames = fitted; changed = true }
        }
        if changed { draft = next; message = "Clip lengths adjusted to this model’s supported lengths." }
    }

    /// Placement and model-licence probes describe a single new generation,
    /// never the joined length stored in the source video's metadata.
    func probeDraft(_ base: RenderDraft) -> RenderDraft {
        guard let frames = draft?.stages.first?.frames else { return base }
        var probe = base
        probe.frames = frames
        probe.prefersAutomaticDuration = false
        return probe
    }

    static func baseRequest(_ draft: RenderDraft, model: String, maxIdentityPhotos: Int = 0) -> GenerateRequest {
        var request = RenderRequest.one(draft, model: model, maxIdentityPhotos: maxIdentityPhotos)
        // Authored clips have explicit lengths. Preserve the ordinary draft's
        // endpoint indices even when its one-shot duration preference is Auto.
        request.frames = draft.frames
        return request
    }

    func refusal(base: GenerateRequest, limits: ChainLimits?) -> String? {
        guard let draft else { return nil }
        if needsReferenceDecision { return "Restore the original references or confirm continuing without them." }
        if restoringSource { return "Reconnect to the original machine to restore this continued video, or choose Start fresh." }
        guard let limits else { return "Checking this machine’s video support…" }
        do {
            if let continuation { _ = try continuation.request(stages: draft.wireStages(), composer: base) }
            else { _ = try draft.request(base: base, limits: limits, repeatReferences: true) }
            return nil
        } catch { return error.localizedDescription }
    }

    func submit(base: GenerateRequest, host: MoldHost, installation: String?,
                backend: any MoldChainBackend, current: @escaping @MainActor () -> Bool,
                prepare: (@MainActor () async throws -> GenerateRequest)? = nil) {
        guard let draft, !preparing else { return }
        guard !needsReferenceDecision else {
            message = "Restore the original references or confirm continuing without them."
            return
        }
        guard !restoringSource else { message = "Reconnect to the original machine before continuing."; return }
        let mine = generation
        preparing = true
        message = "Checking video on \(host.name)…"
        Task {
            defer { if mine == generation { preparing = false } }
            do {
                guard current() else { throw SequencePresentationError.changedMachine }
                let base = try await prepare?() ?? base
                let limits = try await backend.chainLimits(model: base.model, fps: base.fps)
                let request: AuthoredChainRequest
                if let continuation {
                    guard continuationHost == host else { throw SequencePresentationError.changedMachine }
                    // Latest take and configuration are rechecked on every press.
                    let detail = try await backend.chainJob(id: continuation.jobId)
                    let refreshed = try ChainContinuationSource(hostId: host.id, instanceId: installation,
                                                               detail: detail, limits: limits)
                    guard refreshed.take == continuation.take, refreshed.settings == continuation.settings else { throw SequencePresentationError.changedSource }
                    request = try continuation.request(stages: draft.wireStages(), composer: base)
                } else { request = try draft.request(base: base, limits: limits, repeatReferences: true) }
                let plan = try await backend.validateChain(request)
                guard mine == generation, current() else { throw SequencePresentationError.changedMachine }
                validatedPlan = plan
                let id = UUID().uuidString.lowercased()
                // Server uses the operation UUID as the durable job ID. Persist
                // before POST, so a lost answer is recovered by GET, never replay.
                try remember(Pending(id: id, host: host.id, origin: host.baseURL.absoluteString,
                                     installation: installation))
                authorities["\(host.id.uuidString):\(id)"] = current
                jobs.append(Job(jobId: id, host: host, installation: installation, message: "Submitting \(plan.stageCount) clips…"))
                do {
                    let accepted = try await backend.createChainJob(request, operationId: id)
                    guard accepted.jobId == id else { throw SequencePresentationError.changedSource }
                    if stopped.contains("\(host.id.uuidString):\(id)") { try await backend.cancelChainJob(id: id) }
                    if generation == mine {
                        let warnings = accepted.requestWarnings ?? []
                        message = (["Added to the video queue on \(host.name)."] + warnings).joined(separator: " ")
                    }
                    follow(id: id, host: host, backend: backend, current: current)
                } catch {
                    if cancelledBeforeAcceptance.contains("\(host.id.uuidString):\(id)") { return }
                    if let clientError = error as? MoldClientError, case let .http(status, _, _) = clientError,
                       [400, 401, 403, 404, 413, 415, 422, 451].contains(status) {
                        forget(id, host: host.id)
                        jobs.removeAll { $0.jobId == id && $0.host.id == host.id }
                        if generation == mine { message = error.localizedDescription }
                        return
                    }
                    // An uncertain POST may have succeeded. Keep its known ID.
                    message = "Couldn’t confirm acceptance. Reconnect to check the video queue. \(error.localizedDescription)"
                    follow(id: id, host: host, backend: backend, current: current)
                }
            } catch { if mine == generation { message = error.localizedDescription } }
        }
    }

    func recover(host: MoldHost, installation: String?, backend: any MoldChainBackend,
                 current: @escaping @MainActor () -> Bool) {
        if let source = savedSource, source.host == host.id, source.origin == host.baseURL.absoluteString,
           source.installation == installation, continuation == nil {
            Task {
                do {
                    let detail = try await backend.chainJob(id: source.job, includeSourceMedia: true)
                    let limits = try await backend.chainLimits(model: detail.model, fps: detail.script?.chain.fps)
                    let restored = try ChainContinuationSource(hostId: host.id, instanceId: installation, detail: detail, limits: limits)
                    guard current(), savedSource?.job == source.job,
                          restored.take == source.take, detail.galleryFilename == source.filename else {
                        throw SequencePresentationError.changedSource
                    }
                    continuation = restored; continuationHost = host; continuationTitle = source.filename
                } catch {
                    guard savedSource?.job == source.job else { return }
                    message = "Can’t restore continued video. \(error.localizedDescription)"
                }
            }
        }
        for record in pending where record.host == host.id && record.origin == host.baseURL.absoluteString {
            guard record.installation == installation else { continue }
            if !jobs.contains(where: { $0.jobId == record.id && $0.host.id == host.id }) {
                jobs.append(Job(jobId: record.id, host: host, installation: installation, message: "Reconnecting…"))
            }
            follow(id: record.id, host: host, backend: backend, current: current)
        }
    }

    private func follow(id: String, host: MoldHost, backend: any MoldChainBackend,
                        current: @escaping @MainActor () -> Bool) {
        let identity = "\(host.id.uuidString):\(id)"
        authorities[identity] = current
        guard tasks[identity] == nil else { return }
        let token = UUID()
        followTokens[identity] = token
        tasks[identity] = Task { [weak self] in
            defer {
                if self?.followTokens[identity] == token {
                    self?.tasks[identity] = nil; self?.followTokens[identity] = nil
                }
            }
            while !Task.isCancelled, current() {
                do {
                    let detail = try await backend.chainJob(id: id)
                    guard !Task.isCancelled, current(), let self, self.followTokens[identity] == token else { return }
                    self.update(detail, id: identity)
                    if detail.state == .completed || detail.state == .cancelled { self.forget(id, host: host.id); return }
                } catch {
                    guard let self, current() else { return }
                    self.setMessage("Waiting to reconnect. \(error.localizedDescription)", id: identity)
                }
                try? await Task.sleep(for: .seconds(2))
            }
        }
    }
    private func update(_ detail: ChainJobDetail, id: String) {
        guard let index = jobs.firstIndex(where: { $0.id == id }) else { return }
        jobs[index].detail = detail
        if detail.state == .completed, detail.galleryFilename == nil {
            jobs[index].message = "The video finished without publishing a file to the Library. Check this machine’s output settings."
            return
        }
        if detail.state == .completed || detail.state == .cancelled {
            let settled = jobs.filter { $0.detail?.state == .completed || $0.detail?.state == .cancelled }
            if settled.count > 32 {
                let obsolete = Set(settled.prefix(settled.count - 32).map(\.id))
                jobs.removeAll { obsolete.contains($0.id) }
                for identity in obsolete { authorities[identity] = nil; stopped.remove(identity) }
                guard let refreshed = jobs.firstIndex(where: { $0.id == id }) else { return }
                setMessage(detail.error ?? detail.state.rawValue.capitalized, id: jobs[refreshed].id)
                return
            }
        }
        if detail.state == .cancelled || detail.state == .completed { jobs[index].cancelling = false }
        jobs[index].message = jobs[index].cancelling ? "Cancel requested; awaiting host confirmation." : detail.error ?? "\(detail.executionPhase == "finalizing" ? "Joining clips" : (detail.executionPhase == "queued" ? "Queued" : detail.state.rawValue.capitalized)) · clip \(min(detail.currentStage + 1, detail.stageCount)) of \(detail.stageCount)"
    }
    private func setMessage(_ text: String, id: String) {
        if let index = jobs.firstIndex(where: { $0.id == id }) { jobs[index].message = text }
    }
    func cancel(_ job: Job, backend: any MoldChainBackend) {
        guard let isCurrent = authorities[job.id], isCurrent(), let index = jobs.firstIndex(where: { $0.id == job.id }),
              jobs[index].host == job.host, jobs[index].installation == job.installation else { return }
        stopped.insert(job.id)
        jobs[index].cancelling = true
        setMessage("Cancel requested; awaiting host confirmation.", id: job.id)
        Task {
            guard isCurrent() else {
                historyNotice = "The machine changed before Stop was sent. Reopen its current queue."
                return
            }
            do {
                if job.detail == nil {
                    try await backend.cancelChainOperation(id: job.jobId, operationId: job.jobId)
                    do { _ = try await backend.chainJob(id: job.jobId) }
                    catch {
                        if let client = error as? MoldClientError, case .http(status: 404, code: _, message: _) = client {
                            cancelledBeforeAcceptance.insert(job.id)
                            forget(job.jobId, host: job.host.id)
                            tasks[job.id]?.cancel(); tasks[job.id] = nil; followTokens[job.id] = nil
                            jobs.removeAll { $0.id == job.id }
                        } else { throw error }
                    }
                } else { try await backend.cancelChainJob(id: job.jobId) }
            }
            catch { setMessage(error.localizedDescription, id: job.id) }
        }
    }
    func dismiss(_ job: Job) {
        guard job.detail?.state == .completed || job.detail?.state == .cancelled else { return }
        jobs.removeAll { $0.id == job.id }
        authorities[job.id] = nil; stopped.remove(job.id)
    }
    func resume(_ job: Job, backend: any MoldChainBackend) {
        guard let isCurrent = authorities[job.id], isCurrent() else { return }
        Task {
            guard isCurrent() else { return }
            do { try await backend.resumeChainJob(id: job.jobId) }
            catch { setMessage(error.localizedDescription, id: job.id) }
        }
    }
}

private enum SequencePresentationError: LocalizedError {
    case changedMachine, changedSource
    var errorDescription: String? {
        switch self {
        case .changedMachine: "The video’s machine changed. Reopen it from its original Library."
        case .changedSource: "The source video changed. Reopen its latest result before continuing."
        }
    }
}

extension NativeSequenceSession {
    func openContinuation(entry: LibraryEntry, host: MoldHost, installation: String?,
                          backend: any MoldChainBackend, current: @escaping @MainActor () -> Bool,
                          opened: @escaping @MainActor (ChainContinuationSource) -> Void = { _ in }) {
        guard let id = entry.print.metadata.chainJobId else {
            message = "This video has no saved sequence to continue."; return
        }
        guard let installation, !installation.isEmpty else {
            message = "Reconnect to the video’s original machine before continuing."; return
        }
        generation += 1
        let mine = generation
        preparing = true
        Task {
            defer { if mine == generation { preparing = false } }
            do {
                let detail = try await backend.chainJob(id: id, includeSourceMedia: true)
                guard detail.galleryFilename == entry.print.filename else { throw SequencePresentationError.changedSource }
                let limits = try await backend.chainLimits(model: detail.model, fps: entry.print.metadata.fps.map { Int($0) })
                let source = try ChainContinuationSource(hostId: host.id, instanceId: installation, detail: detail, limits: limits)
                guard current(), generation == mine else { throw SequencePresentationError.changedMachine }
                savedSource = SavedDraft.Source(host: host.id, origin: host.baseURL.absoluteString,
                    installation: installation, job: source.jobId, take: source.take, filename: entry.print.filename,
                    hadReferences: source.script.stage.last?.generationReferences?.isEmpty == false)
                continuation = source; continuationHost = host
                continuationTitle = entry.print.displayName
                preparing = false
                draft = SequenceDraft(prompt: "", frames: limits.framesPerClipRecommended)
                message = nil
                opened(source)
            } catch { message = "Can’t continue this video. \(error.localizedDescription)" }
        }
    }
    func discover(host: MoldHost, installation: String?, backend: any MoldChainBackend,
                  current: @escaping @MainActor () -> Bool) async {
        do {
            let listing = try await backend.listChainJobs()
            guard current() else { return }
            historyNotice = nil
            for detail in listing.jobs where detail.ephemeral != true
                && detail.state != .completed && detail.state != .cancelled {
                if let index = jobs.firstIndex(where: { $0.jobId == detail.id && $0.host.id == host.id }) {
                    if jobs[index].host != host || jobs[index].installation != installation {
                        let fresh = try await backend.chainJob(id: detail.id)
                        guard current() else { return }
                        let identity = "\(host.id.uuidString):\(detail.id)"
                        guard let refreshedIndex = jobs.firstIndex(where: { $0.id == identity }) else { continue }
                        tasks[identity]?.cancel(); tasks[identity] = nil; followTokens[identity] = nil
                        jobs[refreshedIndex] = Job(jobId: detail.id, host: host, installation: installation,
                                          detail: fresh, message: fresh.state.rawValue.capitalized)
                    }
                } else {
                    jobs.append(Job(jobId: detail.id, host: host, installation: installation, detail: detail, message: detail.state.rawValue.capitalized))
                }
                follow(id: detail.id, host: host, backend: backend, current: current)
            }
        } catch {
            guard current(), !Task.isCancelled else { return }
            if let client = error as? MoldClientError, case let .http(status, _, _) = client,
               status == 404 || status == 405 { return }
            historyNotice = "Video queue unavailable on \(host.name). \(error.localizedDescription)"
        }
    }
    func suspend() {
        for task in tasks.values { task.cancel() }
        tasks.removeAll(); followTokens.removeAll()
    }
}
