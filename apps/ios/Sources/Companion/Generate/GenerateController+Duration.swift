import Foundation
import MoldClient
import UIKit

extension GenerateController {
    /// Authored sequence drafts keep their original separate-part semantics.
    var canAuthorVideoDuration: Bool {
        kind == .clip && sequence.draft == nil
            && target.flatMap { hosts.capabilities[$0.id] }?.canCreateVideoJob == true
    }

    var usesVideoDuration: Bool { kind == .clip && sequence.draft == nil && (canAuthorVideoDuration || videoDuration != nil) }

    var durationOwner: VideoJobOwner? {
        guard let host = target, let instance = hosts.instanceID(of: host.id), !instance.isEmpty else { return nil }
        return VideoJobOwner(host: host, instanceId: instance)
    }

    /// Initial length follows the selected recipe. Later model changes do not
    /// overwrite an explicit duration, scene prompts or original source owner.
    func ensureVideoDuration() {
        guard videoDuration == nil, kind == .clip else { return }
        let fps = max(1, draft.fps ?? recipe?.defaults.fps ?? 24)
        let frames = max(1, draft.frames ?? recipe?.defaults.frames ?? fps)
        videoDuration = .init(draft: .init(seconds: (try? VideoDurationDraft.initialSeconds(frames: frames, fps: fps)) ?? ""))
    }

    var durationBlocker: String? {
        guard let authoring = videoDuration else { return nil }
        if let refusal = durationSourceRefusal { return refusal }
        guard canAuthorVideoDuration else { return "Reconnect or update this machine before making this duration-based video." }
        guard let owner = durationOwner, let modelName else { return "Reconnect this machine before making this video." }
        guard draft.batchSize >= 1, draft.batchSize <= referenceBatchLimit else {
            return "Batch exceeds this machine’s current limit. Choose a smaller batch before generating."
        }
        if videoJobs.hasUncertainAdmission(on: owner) { return "Check the saved video attempt in Queue before making another video." }
        do {
            _ = try authoring.request(generation: RenderRequest.one(draft, model: modelName,
                maxIdentityPhotos: hosts.capabilities[owner.host.id]?.maxIdentityPhotos ?? 0), owner: owner)
            return nil
        } catch { return UserFacingError.describe(error) }
    }

    func durationRequests(from prepared: RenderDraft, authoring: VideoDurationAuthoring,
                          owner: VideoJobOwner, model: String, copies: Int, randomBase: UInt64) throws -> [VideoJobRequest] {
        try RenderRequest.batch(prepared, model: model, copies: copies, randomBase: randomBase,
            maxIdentityPhotos: hosts.capabilities[owner.host.id]?.maxIdentityPhotos ?? 0)
            .map { try authoring.request(generation: $0, owner: owner) }
    }

    func generateDurationVideo() {
        guard durationTask == nil, let owner = durationOwner, let modelName else { return }
        ensureVideoDuration()
        guard let authoring = videoDuration, durationBlocker == nil else { return }
        let snapshot = draft, selectedRecipe = recipe
        let copies = snapshot.batchSize
        let retained = retainedReuse.snapshot()
        let current: @MainActor @Sendable () -> Bool = { [weak self] in
            self?.followingSuspended == false && self?.hosts.host(owner.host.id) == owner.host
                && self?.hosts.instanceID(of: owner.host.id) == owner.instanceId
        }
        let ordinaryBackend = hosts.backend(for: owner.host)
        let backend: any MoldBackend = if let http = ordinaryBackend as? HTTPBackend {
            http.checkingReferenceAuthority { await MainActor.run { current() } }.preparingLiteralVideoInputs()
        } else { ordinaryBackend }
        durationError = nil; saveDraft()
        var background = UIBackgroundTaskIdentifier.invalid
        background = UIApplication.shared.beginBackgroundTask(withName: "Generate video") {
            UIApplication.shared.endBackgroundTask(background); background = .invalid
        }
        durationTask = Task { [weak self] in
            guard let self else { return }
            defer {
                self.durationTask = nil
                if background != .invalid { UIApplication.shared.endBackgroundTask(background) }
            }
            do {
                let fitted = try await snapshot.fittingSource(recipe: selectedRecipe)
                guard current(), self.videoDuration == authoring, self.draft == snapshot else { throw CancellationError() }
                let requests = try self.durationRequests(from: fitted, authoring: authoring, owner: owner,
                    model: modelName, copies: max(1, copies), randomBase: .random(in: 0 ... UInt64(UInt32.max)))
                // Hydrate each sibling separately: retained sessions are consumed
                // once by one parent, rather than shared across several POSTs.
                for request in requests {
                    guard !Task.isCancelled, current(), self.videoDuration == authoring, self.draft == snapshot else { return }
                    await self.videoJobs.submit(request, owner: owner, backend: backend, current: current,
                        unchanged: { self.videoDuration == authoring && self.draft == snapshot && self.modelName == modelName }, prepare: {
                            var prepared = request
                            if let retained {
                                guard self.hosts.host(retained.origin) == retained.route,
                                      self.hosts.instanceID(of: retained.origin) == retained.instance,
                                      let origin = self.hosts.backend(for: retained.origin) else {
                                    throw MoldClientError.unreachable("Reconnect the machine that kept the source media.")
                                }
                                let admission = try await RetainedSourceMedia.hydrated(
                                    BatchAdmission(requests: [request.generation]), filename: retained.filename,
                                    members: retained.members, sameHost: retained.origin == owner.host.id,
                                    origin: origin, target: backend)
                                guard admission.requests.count == 1 else { throw MoldClientError.malformedResponse }
                                prepared.generation = admission.requests[0]
                                prepared.retainedMediaSession = admission.retainedMediaSession
                            }
                            return prepared
                        })
                    if self.videoJobs.error != nil { break }
                }
            } catch {
                if current(), !Task.isCancelled { self.durationError = UserFacingError.describe(error) }
            }
        }
    }

    func discoverDurationVideos() async {
        videoJobs.restorePending(hosts: hosts.hosts)
        await restoreDurationSource()
        for host in hosts.upHosts where hosts.capabilities[host.id]?.canCreateVideoJob == true {
            guard let instance = hosts.instanceID(of: host.id) else { continue }
            let owner = VideoJobOwner(host: host, instanceId: instance)
            let backend = hosts.backend(for: host)
            let current: @MainActor @Sendable () -> Bool = { [weak self] in
                self?.followingSuspended == false && self?.hosts.host(host.id) == host && self?.hosts.instanceID(of: host.id) == instance
            }
            await videoJobs.recover(owner: owner, backend: backend, current: current)
            await videoJobs.discover(owner: owner, backend: backend, current: current)
        }
    }

    /// Captures the private scalar draft and installation, including current
    /// credentials. Every asynchronous preview checks both before publishing.
    func refreshDurationPlan() {
        guard canAuthorVideoDuration else { videoPlanner.clear(); return }
        ensureVideoDuration()
        guard let authoring = videoDuration, let owner = durationOwner, let modelName else {
            videoPlanner.clear(); return
        }
        let snapshot = draft
        guard let request = try? authoring.request(generation: RenderRequest.one(snapshot, model: modelName,
            maxIdentityPhotos: hosts.capabilities[owner.host.id]?.maxIdentityPhotos ?? 0), owner: owner) else {
            videoPlanner.clear(); return
        }
        videoPlanner.update(request, backend: hosts.backend(for: owner.host), current: { [weak self] in
            guard let self else { return false }
            return self.hosts.host(owner.host.id) == owner.host && self.durationOwner == owner
                && self.videoDuration == authoring && self.draft == snapshot && self.modelName == modelName
        })
    }
}
