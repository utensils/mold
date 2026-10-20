import MoldClient

extension GenerateController {
    func usesDuration(on host: MoldHost?, recipe: GenerationRecipe?) -> Bool {
        guard recipe?.temporal != nil, sequence.draft == nil else { return false }
        return videoDuration != nil || host.flatMap { hosts.capabilities[$0.id] }?.canCreateVideoJob == true
    }

    func ensureDuration(recipe: GenerationRecipe?, host: MoldHost?) {
        guard videoDuration == nil, usesDuration(on: host, recipe: recipe), let temporal = recipe?.temporal else { return }
        let frames = max(1, draft.frames ?? temporal.frames.default)
        let fps = max(1, temporal.fps.resolvedRate(draft.fps))
        videoDuration = .init(draft: .init(seconds: (try? VideoDurationDraft.initialSeconds(frames: frames, fps: fps)) ?? ""))
    }

    func durationOwner(_ host: MoldHost?) -> VideoJobOwner? {
        guard let host, let installation = hosts.instanceID(of: host.id), !installation.isEmpty else { return nil }
        return .init(host: host, instanceId: installation)
    }

    func durationIsCurrent(_ owner: VideoJobOwner) -> Bool {
        hosts.host(owner.host.id) == owner.host && hosts.instanceID(of: owner.host.id) == owner.instanceId
    }

    func durationRequest(on host: MoldHost?, planning: Bool = false) throws -> VideoJobRequest {
        guard let owner = durationOwner(host), let modelName, let videoDuration else {
            throw SequenceDraftError.invalid("Reconnect to the selected machine before generating this video.")
        }
        let generation = RenderRequest.one(draft, model: modelName,
            maxIdentityPhotos: hosts.capabilities[owner.host.id]?.maxIdentityPhotos ?? 0)
        let request = try videoDuration.request(generation: generation, owner: owner)
        return planning ? request.planningProjection : request
    }

    func planDuration(on host: MoldHost?, recipe: GenerationRecipe?) {
        ensureDuration(recipe: recipe, host: host)
        guard usesDuration(on: host, recipe: recipe), let owner = durationOwner(host),
              let backend = hosts.backend(for: owner.host.id) else { videoPlanner.clear(); return }
        do {
            let request = try durationRequest(on: host, planning: true)
            videoPlanner.update(request, backend: backend, destinationInstanceId: owner.instanceId, current: { [weak self] in
                self?.durationIsCurrent(owner) == true && (try? self?.durationRequest(on: host, planning: true)) == request
            })
        } catch { videoPlanner.clear() }
    }

    func submitDuration(on host: MoldHost, recipe: GenerationRecipe?, retained: RetainedMediaHydration? = nil) {
        guard !durationSubmitting, let owner = durationOwner(host), let modelName,
              let authoring = videoDuration, let originalBackend = hosts.backend(for: host.id) else { return }
        let authority = { @MainActor [weak self] in self?.durationIsCurrent(owner) == true }
        let backend: any MoldBackend = if let http = originalBackend as? HTTPBackend {
            http.checkingReferenceAuthority { await MainActor.run { authority() } }.preparingLiteralVideoInputs()
        } else { originalBackend }
        guard durationSourceRefusal == nil else {
            submissionFeedback.begin(durationSourceRefusal ?? "Check the original video inputs.", phase: .refused); return
        }
        let snapshot = draft
        let caps = hosts.capabilities(of: host)
        let batchLimit = ReferenceUploadPolicy.batchLimit(
            requests: [RenderRequest.one(snapshot, model: modelName)], apiKey: host.apiKey,
            capabilities: caps?.referenceUploads, batchLimit: caps?.maxBatchOutputs ?? snapshot.batchSize)
        guard snapshot.batchSize >= 1, snapshot.batchSize <= batchLimit else {
            submissionFeedback.begin("Batch exceeds this machine’s current limit. Choose a smaller batch before generating.", phase: .refused)
            return
        }
        let maxPhotos = hosts.capabilities[host.id]?.maxIdentityPhotos ?? 0
        do { _ = try authoring.request(generation: RenderRequest.one(snapshot, model: modelName, maxIdentityPhotos: maxPhotos), owner: owner) }
        catch { submissionFeedback.begin(UserFacingError.describe(error), phase: .refused); return }
        durationSubmitting = true
        submissionFeedback.begin("Preparing video…")
        Task {
            defer { durationSubmitting = false }
            let current = { @MainActor [weak self] in self?.durationIsCurrent(owner) == true }
            let unchanged = { @MainActor [weak self] in self?.draft == snapshot && self?.videoDuration == authoring && self?.modelName == modelName }
            do {
                // Fit original inputs once; the sibling splitter retains batch
                // provenance and gives every resulting parent batch_size 1.
                let fitted = try await snapshot.fittingSource(recipe: recipe)
                guard current(), unchanged() else { throw SequenceDraftError.invalid("The draft or machine changed. Press Generate again.") }
                let requests = RenderRequest.batch(fitted, model: modelName, copies: snapshot.batchSize, randomBase: UInt64.random(in: 0 ... UInt64(UInt32.max)),
                    maxIdentityPhotos: maxPhotos)
                for generation in requests {
                    guard current(), unchanged() else { throw SequenceDraftError.invalid("The draft or machine changed. Check the video queue before trying again.") }
                    let request = try authoring.request(generation: generation, owner: owner)
                    await videoJobs.submit(request, owner: owner, backend: backend, current: current, unchanged: unchanged, prepare: {
                        let hydrated = try await RetainedMedia.hydrated(BatchAdmission(requests: [generation]),
                            with: retained, on: host, backend: backend)
                        guard let generation = hydrated.requests.first, hydrated.requests.count == 1 else {
                            throw MoldClientError.malformedResponse
                        }
                        var prepared = try authoring.request(generation: generation, owner: owner)
                        prepared.retainedMediaSession = hydrated.retainedMediaSession
                        return prepared
                    })
                    if let error = videoJobs.error { throw SequenceDraftError.invalid(error) }
                }
                submissionFeedback.begin("Added to the video queue.", phase: .accepted)
            } catch { submissionFeedback.begin(UserFacingError.describe(error), phase: .refused) }
        }
    }
}

extension GenerateController {
    func recoverDurationJobs(activity: [MoldHost.ID: ActivityHostSnapshot] = [:], discover: Bool = false) async {
        videoJobs.restorePending(hosts: hosts.hosts)
        for host in hosts.hosts {
            guard hosts.capabilities[host.id]?.canCreateVideoJob == true,
                  let owner = durationOwner(host), let backend = hosts.backend(for: host.id) else { continue }
            let current = { @MainActor [weak self] in self?.durationIsCurrent(owner) == true }
            await videoJobs.recover(owner: owner, backend: backend, current: current)
            if discover {
                let ids = activity[host.id]?.items.filter { $0.kind == "video_job" }.map(\.id) ?? []
                await videoJobs.discover(owner: owner, backend: backend, activityIDs: ids, current: current)
            }
        }
    }
}
