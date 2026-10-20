import MoldClient

extension GenerateController {
    func refreshSequenceLimits() async {
        guard kind == .clip, let host = target, let modelName else { sequenceLimits = nil; return }
        let captured = host
        let capturedInstallation = hosts.instanceID(of: host.id)
        let fps = draft.fps
        sequenceLimits = nil
        do {
            let limits = try await hosts.backend(for: host).chainLimits(model: modelName, fps: fps)
            guard target == captured, hosts.instanceID(of: host.id) == capturedInstallation, self.modelName == modelName, draft.fps == fps else { return }
            sequenceLimits = limits
        } catch { /* Ordinary one-shot controls stay usable on older machines. */ }
    }
    func recoverSequences() {
        for host in hosts.upHosts {
            let captured = host
            let capturedInstallation = hosts.instanceID(of: host.id)
            sequence.recover(host: host, installation: capturedInstallation,
                backend: hosts.backend(for: host), current: { self.hosts.host(captured.id) == captured && self.hosts.instanceID(of: captured.id) == capturedInstallation })
        }
    }
    func discoverSequences() async {
        recoverSequences()
        for host in hosts.upHosts {
            let backend = hosts.backend(for: host)
            let captured = host
            let capturedInstallation = hosts.instanceID(of: host.id)
            await sequence.discover(host: host, installation: capturedInstallation,
                backend: backend, current: { self.hosts.host(captured.id) == captured && self.hosts.instanceID(of: captured.id) == capturedInstallation })
        }
    }
    func continueVideo(_ entry: LibraryEntry) {
        guard let host = hosts.host(entry.hostID) else { return }
        let captured = host
        let capturedInstallation = hosts.instanceID(of: host.id)
        sequence.openContinuation(entry: entry, host: host, installation: capturedInstallation,
            backend: hosts.backend(for: host), current: { self.hosts.host(captured.id) == captured && self.hosts.instanceID(of: captured.id) == capturedInstallation }, opened: { source in
                self.modelName = source.model
                self.retainedReuse.clear()
                self.kind = .clip; self.machine = .pinned(host.id)
                self.draft = RenderDraft(reusing: entry.print.metadata)
                self.draft.media = DraftMedia()
                self.draft.media.generationReferences = (source.script.stage.last?.generationReferences ?? []).map(AuthoringReference.canonical)
                self.draft.width = source.settings.width; self.draft.height = source.settings.height
                self.draft.fps = source.settings.fps; self.draft.steps = source.settings.steps
                self.draft.guidance = source.settings.guidance
            })
    }
}
