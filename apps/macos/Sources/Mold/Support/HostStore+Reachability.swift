import Foundation
import MoldClient

// Asking each machine whether it is there, and what it can do.
@MainActor
extension HostStore {
    func recordConnectionFailure(_ error: any Error, on id: MoldHost.ID) {
        guard host(id) != nil else { return }
        reachability[id] = .down(error.reasonSentence)
        reconcileEventStreams()
    }

    /// The single place a concrete backend is built. `make lint` fails if
    /// one is constructed anywhere else, which keeps "what is this app
    /// talking to" a decision in one file -- and is what makes swapping in
    /// the in-process engine a change here rather than everywhere.
    static let http: @MainActor (MoldHost) -> any MoldBackend = { HTTPBackend(host: $0) }

    func backend(for host: MoldHost) -> any MoldBackend { makeBackend(host) }

    /// The backend for a machine still in the list. `nil` means it was
    /// removed -- the caller's request has nowhere left to go.
    func backend(for id: MoldHost.ID) -> (any MoldBackend)? { host(id).map(backend(for:)) }

    func host(_ id: MoldHost.ID) -> MoldHost? { hosts.first { $0.id == id } }

    func name(of id: MoldHost.ID) -> String? { host(id)?.name }

    func refreshAll() async {
        await withTaskGroup(of: Void.self) { group in
            for host in hosts {
                group.addTask { await self.refresh(host) }
            }
        }
        // Once every answer is in, rather than once per machine: a fleet-wide
        // check reconciles as one decision.
        reconcileEventStreams()
    }

    func refresh(_ original: MoldHost) async {
        await refreshCoordinator.run(original.id.uuidString) { [weak self] in
            guard let self, let current = self.host(original.id), !Task.isCancelled else { return }
            await self.refreshOnce(current)
        }
    }

    private func refreshOnce(_ original: MoldHost) async {
        defer {
            if let current = self.host(original.id), current.baseURL != original.baseURL {
                watchers.removeValue(forKey: original.id)?.cancel()
            }
            reconcileEventStreams()
        }
        var host = original
        if let current = self.host(original.id) { host = current }
        do {
            if let resolved = try await backend(for: host).resolvedConnection() {
                guard self.host(host.id) == host, !Task.isCancelled else { return }
                let previousURL = host.baseURL
                host = resolved
                applyConnection(host, expectedURL: previousURL)
                guard let saved = self.host(host.id) else { return }
                host = saved
            }
        } catch {
            guard self.host(host.id) == host, !Task.isCancelled else { return }
            recordConnectionFailure(error, on: host.id)
            return
        }
        if !isUp(host) { reachability[host.id] = .checking }
        let state = await check(host)
        guard self.host(host.id) == host, !Task.isCancelled else { return }
        reachability[host.id] = state
        // It answered, so whatever "can't be reached" line it was carrying
        // is no longer true -- a real refusal, if this same check also
        // surfaces one below, reports its own line separately.
        if case .up = state {
            succeeded(on: host.id, doing: HostFailure.reachVerb)
        }
        // Capabilities change only when the host is rebuilt, so one fetch per
        // reachability check is plenty. Reconciling comes AFTER them, because
        // whether a machine wants watching is something its capabilities say.
        defer { reconcileEventStreams() }
        guard case let .up(status) = state else { return }
        let client = backend(for: host)
        if let info = try? await client.connectionAddresses(), self.host(host.id) == host, !Task.isCancelled, info.instanceId == status.instanceId {
            host.connectionEndpoints = ConnectionRoutes.sanitized(info.endpoints)
            host.connectionInstanceID = info.instanceId
            applyConnection(host, expectedURL: host.baseURL)
            guard let saved = self.host(host.id) else { return }
            host = saved
        }
        guard self.host(host.id) == host, !Task.isCancelled else { return }
        guard capabilities[host.id] == nil else { return }
        let answer = try? await client.capabilities()
        guard self.host(host.id) == host, !Task.isCancelled else { return }
        capabilities[host.id] = answer
        let options = try? await client.exportOptions()
        guard self.host(host.id) == host, !Task.isCancelled else { return }
        exportOptions[host.id] = options
    }

    /// Asks one machine what it is, and answers rather than recording.
    ///
    /// Separated from `refresh` so the host editor can try an address the
    /// person is still typing without that attempt landing in the machine
    /// list -- a half-typed hostname must not turn a working row red.
    func check(_ host: MoldHost) async -> Reachability {
        do {
            return .up(try await backend(for: host).status())
        } catch MoldClientError.unauthorized {
            return .needsKey
        } catch {
            // The sidebar row reads this alone, so it is a sentence on its own
            // -- "Could not connect to the server." -- not the banner's
            // machine-first clause.
            return .down(error.reasonSentence)
        }
    }

    /// Tries an address nobody has committed to yet.
    func probe(url: URL, apiKey: String?) async -> Reachability {
        await check(MoldHost(name: "", baseURL: url, apiKey: apiKey))
    }

    func reachability(of host: MoldHost) -> Reachability {
        reachability[host.id] ?? .unknown
    }

    func capabilities(of host: MoldHost) -> Capabilities? { capabilities[host.id] }

    func isUp(_ host: MoldHost) -> Bool {
        if case .up = reachability(of: host) { return true }
        return false
    }

    /// The machine to work on by default.
    ///
    /// An explicit choice (`HostStore+Default.swift`) outranks the heuristic
    /// below, even when that machine is down -- the pane says "can't be
    /// reached" rather than looking broken. A default that has been REMOVED
    /// (`remove(_:)`) falls through to it: deliberately not "the first one
    /// configured", since the list starts with this Mac, which on most setups
    /// is not running a server at all, and landing there shows an empty model
    /// picker and reads as the app being broken.
    var preferredHost: MoldHost? {
        defaultMachine.flatMap(host) ?? hosts.first(where: isUp) ?? hosts.first
    }

    /// The machine a remembered `uuidString` names, or somewhere real.
    ///
    /// Machine-scoped controls use a fallback when the remembered id is absent
    /// or removed. Navigation uses `MachineNavigation.path` instead: no explicit
    /// selection means the fleet overview, not the preferred machine's page.
    func machine(selected stored: String?) -> MoldHost? {
        guard let stored, let id = UUID(uuidString: stored) else { return preferredHost }
        return host(id) ?? preferredHost
    }
}
