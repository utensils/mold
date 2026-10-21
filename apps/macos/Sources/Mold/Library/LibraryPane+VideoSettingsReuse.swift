import Foundation
import MoldClient

extension LibraryPane {
    /// A chained print opens its authoring shape only after inspecting the
    /// clicked physical copy. A failure leaves the existing draft untouched.
    func reuseVideoSettings(_ entry: LibraryEntry) {
        guard entry.print.isVideo, entry.print.trashedAt == nil,
              let host = hosts.host(entry.hostID), let instance = hosts.instanceID(of: host.id),
              let backend = hosts.backend(for: host.id) else {
            generate.durationSourceError = "Reconnect to this video's original machine before using its settings."
            destination = .generate
            return
        }
        let owner = VideoJobOwner(host: host, instanceId: instance)
        let snapshot = generate.draft, oldDuration = generate.videoDuration, oldSequence = generate.sequence.draft
        let token = UUID(); generate.durationSourceToken = token
        generate.durationSourceLoading = true; generate.durationSourceError = nil
        destination = .generate
        Task {
            defer { if generate.durationSourceToken == token { generate.durationSourceLoading = false } }
            let current = { @MainActor in
                generate.durationSourceToken == token && hosts.host(host.id) == host && hosts.instanceID(of: host.id) == instance
                    && generate.draft == snapshot && generate.videoDuration == oldDuration && generate.sequence.draft == oldSequence
            }
            do {
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
                      let model = models.model(named: name, on: host.id) else {
                    if current() { throw SequenceDraftError.invalid("Install this video's original model on its machine before using its settings.") }
                    return
                }
                let recipe = try NativeVideoSettingsRecipe.resolve(metadata: archive.metadata, model: model)
                guard current() else { return }
                let fence = reuseStore.begin()
                if savedClips == nil, !reuseStore.acceptInspectedArchive(archive) {
                    throw SequenceDraftError.invalid("The original machine changed. Open Use Settings again.")
                }
                generate.applyVideoSettings(restored, legacy: legacy, model: model, recipe: recipe, on: host.id)
                guard savedClips == nil else { return }
                reuseStore.restoring = true
                defer { if reuseStore.isCurrent(fence), !reuseStore.restorationFailed { reuseStore.restoring = false } }
                reuseStore.arm(generate.draft)
                reuseStore.initialMediaRevisions = generate.mediaRevisions
                reuseStore.selectionModel = generate.modelName; reuseStore.selectionRecipe = generate.recipeID
                await reuseStore.remember(archive.metadata, model: generate.modelName, recipe: generate.recipeID,
                    draft: generate.draft, fence: fence)
                guard reuseStore.isCurrent(fence), !Task.isCancelled else { return }
                await reuseStore.loadPreviews(in: generate.draft)
                guard reuseStore.isCurrent(fence), !Task.isCancelled else { return }
                let outgoing = RetainedSourcePicture.outgoing(generate, on: host, hosts: hosts)
                if let placed = await reuseStore.placePicture(in: generate.draft, outgoing: outgoing,
                    live: { generate.draft }, mediaRevisions: { generate.mediaRevisions }) { generate.draft = placed }
            } catch { if current() { generate.durationSourceError = UserFacingError.describe(error) } }
        }
    }
}
