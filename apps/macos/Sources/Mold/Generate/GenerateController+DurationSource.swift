import Foundation
import MoldClient

extension GenerateController {
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
        guard let authoring = videoDuration, authoring.draft.source != nil,
              let host = authoring.sourceOwner.flatMap({ hosts.host($0.hostId) }),
              let owner = durationOwner(host), let backend = hosts.backend(for: host.id),
              (durationSource?.source != authoring.draft.source || durationSourceError != nil), !durationSourceLoading else { return }
        let token = UUID(); durationSourceToken = token
        durationSourceLoading = true; durationSourceError = nil
        defer { if durationSourceToken == token { durationSourceLoading = false } }
        let current = { @MainActor [weak self] in
            self?.durationSourceToken == token && self?.durationIsCurrent(owner) == true
                && self?.videoDuration?.draft.source == authoring.draft.source && self?.videoDuration?.sourceOwner == authoring.sourceOwner
        }
        do {
            let source = try await VideoLibraryExtensionSource.restore(authoring, owner: owner, backend: backend, current: current)
            guard current() else { return }
            durationSource = source
            videoDuration?.draft.source = source.source
        } catch { if current() { durationSourceError = UserFacingError.describe(error) } }
    }
}
