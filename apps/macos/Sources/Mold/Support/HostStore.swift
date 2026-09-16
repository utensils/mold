import Foundation
import MoldClient

/// The machines the app knows about, and what each last said about itself.
///
/// Reachability is held per host rather than as one global "connected" flag:
/// mold is multi-host by design, one machine being down says nothing about
/// another, and the Library merges prints from all of them.
@MainActor
@Observable
final class HostStore {
    private(set) var hosts: [MoldHost]
    private(set) var reachability: [MoldHost.ID: Reachability] = [:]

    enum Reachability {
        case unknown
        case checking
        case up(ServerStatus)
        case down(String)
    }

    init(hosts: [MoldHost]) {
        self.hosts = hosts
    }

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
            let status = try await backend(for: host).status()
            reachability[host.id] = .up(status)
        } catch {
            let reason = (error as? LocalizedError)?.errorDescription
                ?? error.localizedDescription
            reachability[host.id] = .down(reason)
        }
    }

    func reachability(of host: MoldHost) -> Reachability {
        reachability[host.id] ?? .unknown
    }
}

extension HostStore {
    /// First-run hosts.
    ///
    /// `MOLD_NATIVE_HOSTS` seeds extra machines as `name=url` pairs so a dev
    /// run can point at real hardware without those addresses living in the
    /// source. The devshell's `macos-dev` sets it.
    static func seededHosts() -> [MoldHost] {
        var hosts = [
            MoldHost(name: "This Mac", baseURL: URL(string: "http://localhost:7680")!)
        ]
        let seed = ProcessInfo.processInfo.environment["MOLD_NATIVE_HOSTS"] ?? ""
        for entry in seed.split(separator: ",") {
            let parts = entry.split(separator: "=", maxSplits: 1)
            guard parts.count == 2, let url = URL(string: String(parts[1])) else { continue }
            hosts.append(MoldHost(name: String(parts[0]), baseURL: url))
        }
        return hosts
    }
}
