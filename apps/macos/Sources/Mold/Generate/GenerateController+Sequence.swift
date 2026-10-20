import MoldClient

extension GenerateController {
    func restoreSequenceRecipe(_ source: ChainContinuationSource, host: MoldHost) {
        modelName = source.model
        recipeID = nil
        hostID = host.id; machineChoice = host.id
        draft.width = source.settings.width; draft.height = source.settings.height
        draft.fps = source.settings.fps; draft.steps = source.settings.steps
        draft.guidance = source.settings.guidance
        draft.preferredAudio = source.settings.enableAudio
    }
    func recoverSequences() {
        for host in hosts.hosts {
            guard let backend = hosts.backend(for: host.id) else { continue }
            let captured = host
            let capturedInstallation = hosts.instanceID(of: host.id)
            sequence.recover(host: host, installation: capturedInstallation,
                backend: backend, current: { self.hosts.host(captured.id) == captured && self.hosts.instanceID(of: captured.id) == capturedInstallation })
        }
    }
    func discoverSequences() async {
        recoverSequences()
        for host in hosts.hosts {
            guard let backend = hosts.backend(for: host.id) else { continue }
            let captured = host
            let capturedInstallation = hosts.instanceID(of: host.id)
            await sequence.discover(host: host, installation: capturedInstallation,
                backend: backend, current: { self.hosts.host(captured.id) == captured && self.hosts.instanceID(of: captured.id) == capturedInstallation })
        }
    }
    func continueVideo(_ entry: LibraryEntry) {
        guard let host = hosts.host(entry.hostID), let backend = hosts.backend(for: host.id) else { return }
        let captured = host
        let capturedInstallation = hosts.instanceID(of: host.id)
        sequence.openContinuation(entry: entry, host: host, installation: capturedInstallation,
            backend: backend, current: { self.hosts.host(captured.id) == captured && self.hosts.instanceID(of: captured.id) == capturedInstallation }, opened: { source in
                self.modelName = source.model
                self.hostID = host.id; self.machineChoice = host.id
                self.draft = RenderDraft(reusing: entry.print.metadata)
                self.draft.media = DraftMedia()
                self.draft.media.generationReferences = (source.script.stage.last?.generationReferences ?? []).map(AuthoringReference.canonical)
                self.restoreSequenceRecipe(source, host: host)
            })
    }
}
