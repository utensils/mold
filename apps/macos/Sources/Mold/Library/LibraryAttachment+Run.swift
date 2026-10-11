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
        let routes = attachmentRoutes(for: entry)
        Task { await finishAttachment(entry, as: kind, transaction: transaction, fence: fence, routes: routes) }
    }

    private func finishAttachment(
        _ entry: LibraryEntry, as kind: LibraryAttachmentKind,
        transaction: LibraryAttachmentTransaction, fence: DraftAttachmentFence,
        routes: LibraryAttachmentRoutes
    ) async {
        guard attachmentIsCurrent(transaction, entry: entry, fence: fence, routes: routes),
              let data = await actions.data(for: entry) else { return }
        guard attachmentIsCurrent(transaction, entry: entry, fence: fence, routes: routes) else { return }
        let picked: ImportedPicture
        let typedReference: GenerationReference?
        do {
            picked = try await PictureImport.conforming(
                data, name: entry.print.filename, accepting: PictureImport.engineReadable)
            if kind == .reference, attachmentContext?.recipe.capabilities.generationReferences?.mode.isVisible == true {
                typedReference = try await Task.detached(priority: .userInitiated) {
                    try GenerationReferenceImporter.image(picked)
                }.value
            } else {
                typedReference = nil
            }
        } catch {
            if attachmentIsCurrent(transaction, entry: entry, fence: fence, routes: routes) {
                hosts.report(error, on: entry.hostID, doing: "use that picture")
            }
            return
        }
        guard attachmentIsCurrent(transaction, entry: entry, fence: fence, routes: routes),
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
            guard attachmentOffer.canAddReference else { return }
            if let typedReference {
                do {
                    try DraftPictureAttachment.addTypedReference(typedReference, to: &draft, recipe: context.recipe)
                } catch {
                    hosts.report(error, on: context.host.id, doing: "add that reference")
                    return
                }
            } else {
                let layout = ImageConditioningWells.layout(
                    recipe: context.recipe, model: context.model, media: draft.media)
                guard let capability = layout.references else { return }
                DraftPictureAttachment.addReference(
                    picked, to: &draft, capability: capability, recipe: context.recipe)
            }
        }
        generate.draft = draft
        reuseStore.clear()
        destination = .generate
    }

    private func attachmentIsCurrent(
        _ transaction: LibraryAttachmentTransaction, entry: LibraryEntry,
        fence: DraftAttachmentFence, routes: LibraryAttachmentRoutes
    ) -> Bool {
        // The machine's own row: a tile may be a copy presented under a
        // machine filter, which is not itself a row of `items`.
        let live = library.entry(entry.id)
        let sourceIsUp = hosts.host(entry.hostID).map(hosts.isUp) == true
        guard let source = routes.source,
              PictureSource.isCurrent(entry, origin: source, instance: routes.sourceInstance,
                                      hosts: hosts, library: library) else { return false }
        return routes == attachmentRoutes(for: entry)
            && live?.print == entry.print && transaction.permits(
            version: attachmentVersion, cursor: selection, viewed: viewing,
            scope: navigation.scope, destination: destination,
            draftIsCurrent: fence.permits(generate, reuse: reuseStore),
            liveEntry: live, sourceIsUp: sourceIsUp)
    }

    private func attachmentRoutes(for entry: LibraryEntry) -> LibraryAttachmentRoutes {
        LibraryAttachmentRoutes.capture(source: entry.hostID,
                                        destination: attachmentDestinationHost, hosts: hosts)
    }

}
