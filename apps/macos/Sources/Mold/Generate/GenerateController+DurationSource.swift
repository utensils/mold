import Foundation
import MoldClient

extension GenerateController {
    var durationSourceOpeningRefusal: String? {
        durationSourceLoading ? "Checking original video inputs…" : durationSourceError
    }

    func durationRecipeRefusal(_ recipe: GenerationRecipe?) -> String? {
        guard videoDuration?.draft.source != nil, recipe?.temporal == nil else { return nil }
        return "Choose a compatible video model to extend the original video."
    }

    func keepDraftAfterSourceFailure() {
        submissionFeedback.dismiss()
        durationSourceToken = UUID()
        durationSourceLoading = false
        durationSourceError = nil
    }

    var durationUsesSourceEnding: Bool { videoDuration?.usesSourceEnding(plan: videoPlanner.plan) == true }

    var durationSourceRefusal: String? {
        guard let authoring = videoDuration, let source = authoring.draft.source else { return nil }
        if let durationSourceError { return durationSourceError }
        guard let host = authoring.sourceOwner.flatMap({ hosts.host($0.hostId) }),
              let owner = durationOwner(host), authoring.sourceOwner?.matches(owner) == true else {
            return "Reconnect to this video's original machine before extending it."
        }
        guard durationSource?.source == source, durationSource?.owner == owner else {
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
              let owner = durationOwner(host), let backend = hosts.backend(for: host.id) else { return }
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
            !restoration.isCancelled && self?.durationSourceToken == token && self?.durationIsCurrent(owner) == true
                && self?.videoDuration?.draft.source == authoring.draft.source && self?.videoDuration?.sourceOwner == authoring.sourceOwner
        }
        await withTaskCancellationHandler {
            do {
                let source = try await VideoLibraryExtensionSource.restore(authoring, owner: owner,
                    backend: backend, current: current)
                guard current() else { return }
                durationSource = source; videoDuration?.draft.source = source.source
            } catch { if current() { durationSourceError = UserFacingError.describe(error) } }
        } onCancel: { restoration.cancel() }
    }
}
