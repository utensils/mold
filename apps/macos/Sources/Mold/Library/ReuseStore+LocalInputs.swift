import MoldClient

extension ReuseStore {
    /// A durable local snapshot no longer needs an archive locator. This is
    /// a completion receipt, never permission to hydrate remote descriptors.
    func restoreLocalInputsIfComplete(_ inputs: DraftInputSnapshot, saved: SavedReuse) -> Bool {
        guard saved.version == 1, inputs.version == 1, saved.localInputsComplete == true,
              inputs.retainedReuseFingerprint == nil, Self.hasOnlyLocalReferences(inputs) else { return false }
        clear()
        return true
    }

    /// Record completion only after the live authoritative inventory has no
    /// hidden roles left. A populated picture alone cannot prove this.
    func rememberLocalInputsComplete(in draft: RenderDraft) {
        guard authority == nil, !restorationFailed, var saved = savedRecipe,
              Self.hasOnlyLocalReferences(DraftInputSnapshot(draft.media)) else { return }
        saved.localInputsComplete = true
        savedRecipe = saved
        savedFile?.save(saved)
    }

    /// Upgrade the old, invalidated source-only locator without attaching a
    /// picture again. A fresh inventory must prove there are no hidden roles;
    /// this never mints authority or downloads/replaces the local picture.
    func recoverLocalSourceSnapshot(_ controller: GenerateController, saved: SavedReuse, fence: Int) async -> Bool {
        let media = controller.draft.media
        guard saved.version == 1, savedInputBaseline != nil,
              saved.model == controller.modelName, saved.recipe == controller.recipeID,
              media.sourceImage != nil || media.sourceImageOriginal != nil || media.parked.sourceImage != nil,
              Self.hasOnlyLocalReferences(DraftInputSnapshot(media)),
              let instance = saved.instance, hosts.instanceID(of: saved.origin) == instance,
              let archive = saved.archive, let output = saved.output,
              let route = hosts.host(saved.origin), let backend = hosts.backend(for: saved.origin) else { return false }
        func isCurrentRecovery() -> Bool {
            isCurrent(fence) && !Task.isCancelled && savedRecipe == saved
                && hosts.host(saved.origin) == route && hosts.instanceID(of: saved.origin) == instance
                && controller.modelName == saved.model && controller.recipeID == saved.recipe
                && controller.draft.media == media
        }
        do {
            let offer = try await backend.retainedMediaTransferOffer(for: saved.filename)
            guard isCurrentRecovery(), offer.archiveIdentitySha256 == archive, offer.outputSha256 == output else { return false }
            let inventory = try await backend.retainedSourceMedia(for: saved.filename)
            guard isCurrentRecovery(), inventory.availability == .available, inventory.members.count == 1,
                  let member = inventory.members.first, ["source_image", "stage_source:0"].contains(member.role) else { return false }
            var completed = saved
            completed.localInputsComplete = true
            savedRecipe = completed
            savedFile?.save(completed)
            restoring = false
            restorationFailed = false
            notice = nil
            return true
        } catch { return false }
    }

    private static func hasOnlyLocalReferences(_ inputs: DraftInputSnapshot) -> Bool {
        (inputs.active.generationReferences + inputs.parked.generationReferences).allSatisfy {
            $0.media.authority == "inline" && $0.media.data != nil
        }
    }
}
