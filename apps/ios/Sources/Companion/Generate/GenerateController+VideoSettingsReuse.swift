import Foundation
import MoldClient

extension GenerateController {
    func reuseVideoSettings(_ entry: LibraryEntry) {
        guard entry.print.isVideo, entry.print.trashedAt == nil,
              let host = hosts.host(entry.hostID), let instance = hosts.instanceID(of: host.id) else {
            durationSourceError = "Reconnect to this video's original machine before using its settings."
            return
        }
        prepareVideoSourceInspection()
        let owner = VideoJobOwner(host: host, instanceId: instance)
        let snapshot = draft, oldDuration = videoDuration, oldSequence = sequence.draft
        let token = UUID(); durationSourceToken = token; durationSourceLoading = true; durationSourceError = nil
        Task {
            defer { if durationSourceToken == token { durationSourceLoading = false } }
            let current = { @MainActor [weak self] in
                self?.durationSourceToken == token && self?.hosts.host(host.id) == host
                    && self?.hosts.instanceID(of: host.id) == instance && self?.draft == snapshot
                    && self?.videoDuration == oldDuration && self?.sequence.draft == oldSequence
            }
            do {
                let backend = hosts.backend(for: host)
                let archive = try await VideoSettingsArchive.inspect(filename: entry.print.filename,
                    metadata: entry.print.metadata, owner: owner, backend: backend, current: current)
                if archive.inventory.availability == .unavailableLegacy, RetainedSourceMedia.disclosable(archive.metadata) {
                    throw SequenceDraftError.invalid("This video's original input files were not retained. Reattach them explicitly before using its settings.")
                }
                let savedClips = archive.metadata.videoJob == nil
                    ? try await archive.legacySequence(backend: backend, current: current) : nil
                let automatic = archive.metadata.outputMode == "one-shot" ? savedClips : nil
                let restored = try automatic.map { try VideoSettingsReuse(legacyAutomatic: $0) }
                    ?? VideoSettingsReuse(metadata: archive.metadata)
                let legacy = automatic == nil ? savedClips : nil
                guard current(), let name = archive.metadata.model,
                      let model = hosts.models[host.id]?.first(where: { $0.name == name }) else {
                    if current() { throw SequenceDraftError.invalid("Install this video's original model on its machine before using its settings.") }
                    return
                }
                let recipe = try NativeVideoSettingsRecipe.resolve(metadata: archive.metadata, model: model)
                guard current() else { return }
                applyVideoSettings(restored, legacy: legacy, model: model, recipe: recipe, on: host.id)
                guard savedClips == nil else { saveDraft(); return }
                let fence = retainedReuse.begin(draft, metadata: archive.metadata,
                    sourceRevision: sourceMediaRevision, mediaRevisions: mediaRevisions)
                await retainedReuse.probe(entry, fence: fence, controller: self, onlySelectedCopy: true)
                saveDraft()
            } catch { if current() { durationSourceError = UserFacingError.describe(error); durationError = durationSourceError } }
        }
    }

    func applyVideoSettings(_ restored: VideoSettingsReuse, legacy: LegacySequenceSettingsReuse?,
                            model: Model, recipe: GenerationRecipe, on host: MoldHost.ID) {
        retainedReuse.clear(); sequence.startFresh()
        let recipeDraft = legacy?.draft ?? restored.draft
        kind = .clip; machine = .pinned(host); restoringChoice = false
        modelName = model.name; recipeID = recipe.id
        draft = recipeDraft.adopting(recipe, isNewModel: false, for: model)
        draft.frames = recipeDraft.frames
        videoDuration = restored.duration
        durationSource = nil; durationSourceError = nil; durationError = nil
        videoPlanner.clear()
        if let legacy { sequence.restoreSettings(legacy) }
    }
}
