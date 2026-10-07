import Foundation
import MoldClient

// Asking each machine whether it is there, and what it can do.
extension HostStore {
    func recordConnectionFailure(_ error: any Error, on id: MoldHost.ID) {
        guard host(id) != nil else { return }
        setReachability(.down(error.reasonSentence), for: id)
        reconcileWatchers()
    }

    func refreshAll() async {
        await withTaskGroup(of: Void.self) { group in
            for host in hosts {
                group.addTask { await self.refresh(host) }
            }
        }
        reconcileWatchers()
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
                stopWatching(original.id)
            }
            reconcileWatchers()
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
        // Rechecking an answered machine is not an outage. Hiding its
        // models here destroys source wells and any picker they present.
        // Initial checks still say Checking; the answer records real failure.
        if !isUp(host) { setReachability(.checking, for: host.id) }
        let state = await check(host)
        guard self.host(host.id) == host, !Task.isCancelled else { return }
        // Removed while we asked: nothing to record.
        guard self.host(host.id) != nil else { return }
        setReachability(state, for: host.id)
        defer { reconcileWatchers() }
        guard case .up = state else { return }
        setLastAnswered(.now, for: host.id)
        clearFailures(for: host.id, doing: HostFailure.reachVerb)
        let client = backend(for: host)
        if let info = try? await client.connectionAddresses(), self.host(host.id) == host, !Task.isCancelled, info.instanceId == instanceID(of: host.id) {
            host.connectionEndpoints = ConnectionRoutes.sanitized(info.endpoints)
            host.connectionInstanceID = info.instanceId
            applyConnection(host, expectedURL: host.baseURL)
            guard let saved = self.host(host.id) else { return }
            host = saved
        }
        guard self.host(host.id) == host, !Task.isCancelled else { return }
        if capabilities[host.id] == nil {
            let answer = try? await client.capabilities()
            guard self.host(host.id) == host, !Task.isCancelled else { return }
            setCapabilities(answer, for: host.id)
        }
        if let models = try? await client.models() {
            guard self.host(host.id) == host, !Task.isCancelled else { return }
            setModels(models, for: host.id)
        }
    }

    /// Asks one machine what it is, and answers rather than recording: the
    /// Add sheet checks an address while it is still being typed, and a
    /// half-typed name must not turn a working card red.
    func check(_ host: MoldHost) async -> Reachability {
        do {
            return .up(try await backend(for: host).status())
        } catch MoldClientError.unauthorized {
            return .needsKey
        } catch {
            return .down(error.reasonSentence)
        }
    }

    /// Tries an address nobody has committed to yet.
    func probe(url: URL, apiKey: String?) async -> Reachability {
        await check(MoldHost(name: "", baseURL: url, apiKey: apiKey))
    }
}
