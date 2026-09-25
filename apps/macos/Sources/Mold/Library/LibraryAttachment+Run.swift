import MoldClient

extension LibraryPane {
    var attachmentDestinationHost: MoldHost? {
        generate.machineChoice.flatMap(hosts.host)
            ?? generate.hostID.flatMap(hosts.host)
            ?? hosts.preferredHost
    }

    var attachmentModelKey: AttachmentModelKey {
        let host = attachmentDestinationHost
        return AttachmentModelKey(host: host?.id, isUp: host.map(hosts.isUp) == true)
    }

    var attachmentContext: (host: MoldHost, model: Model, recipe: GenerationRecipe)? {
        guard let host = attachmentDestinationHost, hosts.isUp(host) else { return nil }
        let ready = models.ready(on: host.id)
        let model = generate.modelName.flatMap { models.model(named: $0, on: host.id) }
            ?? (generate.modelName == nil
                ? ready.first(where: { $0.name == drafts.restoredModel }) ?? ready.first : nil)
        guard let model else { return nil }
        let recipe = generate.recipeID.flatMap { model.generationProfile?.recipe(named: $0) }
            ?? model.defaultRecipe
        return recipe.map { (host, model, $0) }
    }

    func prepareAttachmentModels() async {
        guard let host = attachmentDestinationHost, hosts.isUp(host),
              !models.hasLoaded(on: host.id)
        else { return }
        await models.refresh(on: host.id)
    }

    func attach(_ entry: LibraryEntry, as kind: LibraryAttachmentKind) {
        attachmentVersion += 1
        let transaction = LibraryAttachmentTransaction(
            version: attachmentVersion, cursor: selection, viewed: viewing,
            scope: navigation.scope, destination: destination)
        let fence = DraftAttachmentFence(controller: generate, reuse: reuseStore)
        Task { await finishAttachment(entry, as: kind, transaction: transaction, fence: fence) }
    }

    private func finishAttachment(
        _ entry: LibraryEntry, as kind: LibraryAttachmentKind,
        transaction: LibraryAttachmentTransaction, fence: DraftAttachmentFence
    ) async {
        guard let data = await actions.data(for: entry) else { return }
        guard attachmentIsCurrent(transaction, entry: entry, fence: fence) else { return }
        let picked: ImportedPicture
        do {
            picked = try await PictureImport.conforming(
                data, name: entry.print.filename, accepting: PictureImport.engineReadable)
        } catch {
            if attachmentIsCurrent(transaction, entry: entry, fence: fence) {
                hosts.report(error, on: entry.hostID, doing: "use that picture")
            }
            return
        }
        guard attachmentIsCurrent(transaction, entry: entry, fence: fence),
              let initialContext = attachmentContext
        else { return }
        if generate.modelName == nil {
            if AttachmentModelAdoption.keepsDraft(
                restoredModel: drafts.restoredModel, adopting: initialContext.model.name) {
                generate.adopt(model: initialContext.model, on: initialContext.host.id,
                               keepingDraft: true)
            } else {
                generate.select(model: initialContext.model, on: initialContext.host.id)
            }
            drafts.adoptedRestoredModel()
        }
        guard let context = attachmentContext else { return }
        var draft = generate.draft
        switch kind {
        case .source:
            guard attachmentOffer.canUseAsSource else { return }
            DraftPictureAttachment.useAsSource(picked, in: &draft, recipe: context.recipe)
        case .reference:
            let layout = ImageConditioningWells.layout(
                recipe: context.recipe, model: context.model, media: draft.media)
            guard let capability = layout.references,
                  capability.hasRoom(for: draft.media.editImages.count) else { return }
            DraftPictureAttachment.addReference(picked, to: &draft, capability: capability)
        }
        generate.draft = draft
        reuseStore.clear()
        destination = .generate
    }

    private func attachmentIsCurrent(
        _ transaction: LibraryAttachmentTransaction, entry: LibraryEntry,
        fence: DraftAttachmentFence
    ) -> Bool {
        // The machine's own row: a tile may be a copy presented under a
        // machine filter, which is not itself a row of `items`.
        let live = library.entry(entry.id)
        let sourceIsUp = hosts.host(entry.hostID).map(hosts.isUp) == true
        return live?.print == entry.print && transaction.permits(
            version: attachmentVersion, cursor: selection, viewed: viewing,
            scope: navigation.scope, destination: destination,
            draftIsCurrent: fence.permits(generate, reuse: reuseStore),
            liveEntry: live, sourceIsUp: sourceIsUp)
    }
}
