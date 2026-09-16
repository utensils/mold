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
        /// Answering, but refusing us. That is a different problem from being
        /// off: the machine is there and the fix is a key, not a reboot.
        case needsKey
        case down(String)
    }

    init(hosts: [MoldHost]) {
        self.hosts = hosts
    }

    // MARK: - Editing

    func add(name: String, url: URL, apiKey: String?) {
        hosts.append(MoldHost(name: name, baseURL: url, apiKey: apiKey))
        persist()
    }

    func update(_ host: MoldHost) {
        guard let index = hosts.firstIndex(where: { $0.id == host.id }) else { return }
        hosts[index] = host
        persist()
        // The key or address may have changed, so what we knew is stale.
        reachability[host.id] = .unknown
    }

    func remove(_ host: MoldHost) {
        hosts.removeAll { $0.id == host.id }
        reachability[host.id] = nil
        HostPersistence.forget(host)
        persist()
    }

    private func persist() {
        HostPersistence.save(hosts)
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

extension HostStore {
    /// First-run hosts.
    ///
    /// `MOLD_NATIVE_HOSTS` seeds extra machines as `name=url` pairs so a dev
    /// run can point at real hardware without those addresses living in the
    /// source. The devshell's `macos-dev` sets it.
    static func seededHosts() -> [MoldHost] {
        // A saved list wins. Seeding over it would resurrect machines the
        // person removed on every launch.
        if let saved = HostPersistence.load() { return saved }

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
