import Foundation
import MoldClient

// Asking each machine whether it is there, and what it can do.
extension HostStore {
    func refreshAll() async {
        await withTaskGroup(of: Void.self) { group in
            for host in hosts {
                group.addTask { await self.refresh(host) }
            }
        }
        reconcileWatchers()
    }

    func refresh(_ host: MoldHost) async {
        setReachability(.checking, for: host.id)
        let state = await check(host)
        // Removed while we asked: nothing to record.
        guard self.host(host.id) != nil else { return }
        setReachability(state, for: host.id)
        defer { reconcileWatchers() }
        guard case .up = state else { return }
        setLastAnswered(.now, for: host.id)
        clearFailures(for: host.id, doing: HostFailure.reachVerb)
        let client = backend(for: host)
        if capabilities[host.id] == nil {
            setCapabilities(try? await client.capabilities(), for: host.id)
        }
        if let models = try? await client.models() { setModels(models, for: host.id) }
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
