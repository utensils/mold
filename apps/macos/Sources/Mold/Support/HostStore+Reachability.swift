import Foundation
import MoldClient

// Asking each machine whether it is there, and what it can do.
@MainActor
extension HostStore {
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
    }

    func refresh(_ host: MoldHost) async {
        reachability[host.id] = .checking
        let state = await check(host)
        reachability[host.id] = state
        // Capabilities change only when the host is rebuilt, so one fetch per
        // reachability check is plenty.
        guard case .up = state, capabilities[host.id] == nil else { return }
        let client = backend(for: host)
        capabilities[host.id] = try? await client.capabilities()
        exportOptions[host.id] = try? await client.exportOptions()
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
            return .down((error as? LocalizedError)?.errorDescription
                ?? error.localizedDescription)
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
    /// Deliberately not "the first one configured": the list starts with this
    /// Mac, which on most setups is not running a server at all. Landing there
    /// shows an empty model picker and reads as the app being broken.
    var preferredHost: MoldHost? {
        hosts.first(where: isUp) ?? hosts.first
    }
}
