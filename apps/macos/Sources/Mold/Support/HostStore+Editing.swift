import Foundation
import MoldClient

// Adding, changing and removing machines. Split from the reachability
// half purely for size.
@MainActor
extension HostStore {


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
        capabilities[host.id] = nil
    }

    func remove(_ host: MoldHost) {
        hosts.removeAll { $0.id == host.id }
        reachability[host.id] = nil
        capabilities[host.id] = nil
        HostPersistence.forget(host)
        persist()
    }

    /// Adds the in-process engine to the machine list.
    ///
    /// NOT persisted: it exists only while this launch has it running, and
    /// writing it to preferences would leave a dead loopback host behind.
    func adoptLocalEngine(_ host: MoldHost) {
        if let index = hosts.firstIndex(where: { $0.id == host.id }) {
            hosts[index] = host
        } else {
            hosts.insert(host, at: 0)
        }
        reachability[host.id] = .unknown
        capabilities[host.id] = nil
        Task { await refresh(host) }
    }

    func dropLocalEngine() {
        hosts.removeAll { $0.id == MoldEngine.localHostID }
        reachability[MoldEngine.localHostID] = nil
        capabilities[MoldEngine.localHostID] = nil
    }

    private func persist() {
        // The local engine is a property of this launch, not of the machine
        // list, so it never goes to disk.
        HostPersistence.save(hosts.filter { $0.id != MoldEngine.localHostID })
    }
}
