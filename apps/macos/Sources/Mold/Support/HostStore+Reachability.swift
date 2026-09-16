import Foundation
import MoldClient

// Asking each machine whether it is there, and what it can do.
@MainActor
extension HostStore {
    /// The single place a concrete backend is built. `make lint` fails if
    /// one is constructed anywhere else, which keeps "what is this app
    /// talking to" a decision in one file -- and is what makes swapping in
    /// the in-process engine a change here rather than everywhere.
    func backend(for host: MoldHost) -> any MoldBackend {
        HTTPBackend(host: host)
    }

    func refreshAll() async {
        await withTaskGroup(of: Void.self) { group in
            for host in hosts {
                group.addTask { await self.refresh(host) }
            }
        }
    }

    func refresh(_ host: MoldHost) async {
        reachability[host.id] = .checking
        do {
            let client = backend(for: host)
            let status = try await client.status()
            reachability[host.id] = .up(status)
            // Capabilities change only when the host is rebuilt, so one fetch
            // per reachability check is plenty.
            if capabilities[host.id] == nil {
                capabilities[host.id] = try? await client.capabilities()
            }
        } catch MoldClientError.unauthorized {
            reachability[host.id] = .needsKey
        } catch {
            let reason = (error as? LocalizedError)?.errorDescription
                ?? error.localizedDescription
            reachability[host.id] = .down(reason)
        }
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
