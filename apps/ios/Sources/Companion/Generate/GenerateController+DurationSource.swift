import Foundation
import MoldClient

extension GenerateController {
    var durationSourceOpeningRefusal: String? {
        durationSourceLoading ? "Checking original video inputs…" : durationSourceError
    }

    func keepDraftAfterSourceFailure() {
        if durationError == durationSourceError { durationError = nil }
        durationSourceToken = UUID()
        durationSourceLoading = false
        durationSourceError = nil
    }

    var durationUsesSourceEnding: Bool { videoDuration?.usesSourceEnding(plan: videoPlanner.plan) == true }

    var durationSourceRefusal: String? {
        guard let authoring = videoDuration, let source = authoring.draft.source else { return nil }
        if let durationSourceError { return durationSourceError }
        guard let host = authoring.sourceOwner.flatMap({ hosts.host($0.hostId) }),
              let instance = hosts.instanceID(of: host.id),
              authoring.sourceOwner?.matches(.init(host: host, instanceId: instance)) == true else {
            return "Reconnect to this video's original machine before extending it."
        }
        guard durationSource?.source == source, durationSource?.owner == VideoJobOwner(host: host, instanceId: instance) else {
            return durationSourceLoading ? "Checking original video inputs…" : "Check the original video inputs before generating."
        }
        if let modelName {
            return authoring.openingConflict(plan: videoPlanner.plan,
                generation: RenderRequest.one(draft.durationGenerationDraft, model: modelName, maxIdentityPhotos: 0))
        }
        return nil
    }

    func restoreDurationSource() async {
        guard let authoring = videoDuration, let selected = authoring.draft.source,
              let host = authoring.sourceOwner.flatMap({ hosts.host($0.hostId) }),
              let instance = hosts.instanceID(of: host.id) else { return }
        let owner = VideoJobOwner(host: host, instanceId: instance)
        guard durationSource?.source != selected || durationSource?.owner != owner || durationSourceError != nil else { return }
        if durationSourceLoading {
            // An explicit newer Extend/Use Settings inspection owns the shared
            // loading indicator. Only another automatic restore may replace us.
            guard let previous = durationSourceRestoration, previous.token == durationSourceToken else { return }
            if previous.owner == owner, previous.source == selected,
               previous.sourceOwner == authoring.sourceOwner, !previous.isCancelled { return }
        }
        let restoration = DurationSourceRestoration(owner: owner, source: selected, sourceOwner: authoring.sourceOwner)
        let token = restoration.token
        durationSourceRestoration = restoration
        durationSourceToken = token; durationSourceLoading = true; durationSourceError = nil
        defer {
            if durationSourceRestoration?.token == token { durationSourceRestoration = nil }
            if durationSourceToken == token { durationSourceLoading = false }
        }
        let current = { @MainActor [weak self] in
            !restoration.isCancelled && self?.durationSourceToken == token && self?.hosts.host(host.id) == host
                && self?.hosts.instanceID(of: host.id) == instance
                && self?.videoDuration?.draft.source == authoring.draft.source && self?.videoDuration?.sourceOwner == authoring.sourceOwner
        }
        await withTaskCancellationHandler {
            do {
                let source = try await VideoLibraryExtensionSource.restore(authoring, owner: owner,
                    backend: hosts.backend(for: host), current: current)
                guard current() else { return }
                durationSource = source; videoDuration?.draft.source = source.source
                saveDraft()
            } catch { if current() { durationSourceError = UserFacingError.describe(error) } }
        } onCancel: { restoration.cancel() }
    }

    /// Library can mount Generate and deliver its action in the same turn.
    /// Settle automatic defaults before capturing a user's source action, so
    /// the view's initial hooks cannot invalidate an otherwise unchanged draft.
    func prepareVideoSourceInspection() {
        settleChoice()
        if canAuthorVideoDuration { ensureVideoDuration() }
    }

    func extendVideo(_ entry: LibraryEntry) {
        guard entry.print.isVideo, entry.print.trashedAt == nil, let host = hosts.host(entry.hostID),
              let instance = hosts.instanceID(of: host.id), hosts.capabilities[host.id]?.canExtendLibraryVideo == true else { return }
        prepareVideoSourceInspection()
        let owner = VideoJobOwner(host: host, instanceId: instance)
        let snapshot = draft, authoring = videoDuration
        let token = UUID(); durationSourceToken = token; durationSourceLoading = true; durationSourceError = nil
        Task {
            defer { if durationSourceToken == token { durationSourceLoading = false } }
            let current = { @MainActor [weak self] in
                self?.durationSourceToken == token && self?.hosts.host(host.id) == host
                    && self?.hosts.instanceID(of: host.id) == instance && self?.draft == snapshot && self?.videoDuration == authoring
            }
            do {
                let source = try await VideoLibraryExtensionSource.inspect(filename: entry.print.filename,
                    title: entry.print.displayName, metadata: entry.print.metadata, owner: owner,
                    backend: hosts.backend(for: host), current: current)
                guard current() else { return }
                retainedReuse.clear(); sequence.startFresh()
                durationSource = source; videoDuration = source.authoring(seconds: "")
                kind = .clip; machine = .pinned(host.id); restoringChoice = false
                draft = source.prefilledDraft; recipeID = nil
                if let name = source.metadata.model { modelName = name }
                if let originalModel = model {
                    if let pipeline = source.metadata.pipeline {
                        recipeID = originalModel.generationProfile?.recipes.first { $0.requestSelector?.pipeline == pipeline }?.id
                    }
                    if let recipe {
                        draft = draft.adopting(recipe, isNewModel: false, for: originalModel)
                        if let temporal = recipe.temporal {
                            videoDuration?.draft.seconds = (try? VideoDurationDraft.initialSeconds(
                                frames: temporal.frames.default, fps: temporal.fps.resolvedRate(draft.fps))) ?? ""
                        }
                    }
                } else { durationError = "Choose or install a compatible video model on the original machine." }
                saveDraft()
            } catch { if current() { durationSourceError = UserFacingError.describe(error); durationError = durationSourceError } }
        }
    }
}
