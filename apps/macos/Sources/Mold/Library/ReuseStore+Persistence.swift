import Foundation
import MoldClient

extension ReuseStore {
    func remember(_ metadata: OutputMetadata, model: String?, recipe: String?, draft: RenderDraft, fence: Int) async {
        guard let authority, isCurrent(fence), let restored,
              authority.instance == hosts.instanceID(of: authority.origin) else { return }
        let locatorFence = locatorVersion
        var saved = SavedReuse(origin: authority.origin, instance: authority.instance,
            filename: authority.filename, model: model, recipe: recipe, metadata: metadata,
            invalidated: !RetainedReferenceGuard.canReuseDraft(draft, original: restored))
        savedRecipe = saved
        savedFile?.save(saved)
        if let backend = hosts.backend(for: authority.origin),
           let offer = try? await backend.retainedMediaTransferOffer(for: authority.filename),
           locatorVersion == locatorFence, savedRecipe == saved,
           hosts.instanceID(of: authority.origin) == saved.instance {
            saved.archive = offer.archiveIdentitySha256
            saved.output = offer.outputSha256
            savedRecipe = saved
            savedFile?.save(saved)
        }
    }

    func restoreSaved(into controller: GenerateController) {
        guard let saved = savedFile?.load() else {
            if savedFile?.exists == true {
                restoring = true
                notice = "The saved links to the original input files could not be read. Use the source print’s settings again, or stop restoring its inputs."
            }
            return
        }
        savedRecipe = saved
        selectionModel = saved.model
        selectionRecipe = saved.recipe
        restoring = true
        controller.draft.media = RenderDraft(reusing: saved.metadata).media
        if saved.invalidated || saved.version != 1 {
            notice = "Use the source print’s settings again, or attach replacement input files before generating."
        }
    }

    /// Adoption writes derived capability/layout fields before observers run.
    /// Reconcile the saved media through that same model before comparing it.
    func adoptRestoredBaseline(_ controller: GenerateController, model: Model) {
        guard restoring, var saved = savedRecipe, !saved.invalidated else { return }
        var expected = RenderDraft(reusing: saved.metadata)
        let recipe = saved.recipe.flatMap { model.generationProfile?.recipe(named: $0) } ?? model.defaultRecipe
        if let recipe { expected = expected.adopting(recipe, isNewModel: false, for: model) }
        if expected.media == controller.draft.media {
            arm(controller.draft)
        } else {
            saved.invalidated = true
            savedRecipe = saved
            savedFile?.save(saved)
        }
    }

    /// Re-probe only the recorded origin, never a convenient responding machine.
    func recoverSaved(_ controller: GenerateController) async {
        guard restoring, let saved = savedRecipe,
              controller.modelName != nil else { return }
        let fence = currentFence
        let expectedMedia = controller.draft.media
        guard !saved.invalidated, saved.version == 1,
              saved.model == controller.modelName, saved.recipe == controller.recipeID,
              let instance = saved.instance, hosts.instanceID(of: saved.origin) == instance,
              let archive = saved.archive, let output = saved.output,
              let backend = hosts.backend(for: saved.origin) else {
            notice = "Reconnect the original machine and use the source print’s settings again to restore its input files."
            return
        }
        do {
            let offer = try await backend.retainedMediaTransferOffer(for: saved.filename)
            guard isCurrent(fence), !Task.isCancelled, savedRecipe == saved,
                  hosts.instanceID(of: saved.origin) == instance,
                  controller.draft.media == expectedMedia else { return }
            guard offer.archiveIdentitySha256 == archive, offer.outputSha256 == output,
                  offer.metadata?.references == saved.metadata.references else {
                notice = "The source print changed. Reselect it before generating."
                return
            }
            arm(controller.draft)
            await probe([PrintID(host: saved.origin, filename: saved.filename)],
                fence: fence, disclosing: saved.metadata)
            guard isCurrent(fence), !Task.isCancelled, savedRecipe == saved,
                  authority?.instance == instance, hosts.instanceID(of: saved.origin) == instance else { return }
            let draft = controller.draft
            let outgoing = hosts.host(saved.origin).flatMap {
                RetainedSourcePicture.outgoing(controller, on: $0, hosts: hosts)
            }
            if let placed = await placePicture(in: draft, outgoing: outgoing, live: { controller.draft }) {
                controller.draft = placed
            }
            guard isCurrent(fence), savedRecipe == saved,
                  hosts.instanceID(of: saved.origin) == instance else { return }
            if authority?.members.contains(where: { $0.role == "source_image" || $0.role == "stage_source:0" }) == true,
               controller.draft.media.sourceImage == nil { return }
            restoring = false
            notice = nil
            await loadPreviews(in: controller.draft)
        } catch {
            if isCurrent(fence), !Task.isCancelled {
                notice = "The original input files could not be verified. Reconnect the original machine and use the source print’s settings again."
            }
        }
    }

    func selectionChanged(model: String?, recipe: String?, draft: RenderDraft) {
        // Initial model adoption is part of restoring this same saved recipe.
        if let selectionModel, selectionModel != model || selectionRecipe != recipe {
            clear()
        }
        guard var saved = savedRecipe else { return }
        let original = restored ?? RenderDraft(reusing: saved.metadata)
        let stillMatches = restored == nil && restoring
            ? draft.media == original.media
            : RetainedReferenceGuard.canReuseDraft(draft, original: original)
        guard !stillMatches else { return }
        saved.invalidated = true
        savedRecipe = saved
        savedFile?.save(saved)
    }
}
