import Foundation
import MoldClient

// One tick: ask every machine, reconcile each answer against what it last
// said.
@MainActor
extension ActivityStore {
    func refresh() async {
        await withTaskGroup(of: Void.self) { group in
            for host in hosts.hosts {
                group.addTask { await self.refresh(on: host.id) }
            }
        }
        // A machine that is no longer listed must stop contributing rows
        // nobody can attribute to a machine -- `LibraryStore.prune`'s rule.
        let listed = Set(hosts.hosts.map(\.id))
        byHost = byHost.filter { listed.contains($0.key) }
    }

    /// One machine, throttled to one read at a time.
    func refresh(on host: MoldHost.ID) async {
        await reads.once(per: host) { [weak self] host in
            await self?.readOnce(on: host)
        }
    }

    /// Not `private`: the throttle calls it from a closure this extension
    /// hands over.
    func readOnce(on host: MoldHost.ID) async {
        guard let route = hosts.host(host) else { return }
        let epoch = (epochs[host] ?? 0) + 1
        epochs[host] = epoch
        let result: Result<ActiveWorkSnapshot, Error>
        do {
            // A machine this app knows is down is not asked. Its rows are
            // kept and marked stale, which is what `reconcile` does with a
            // failure -- an unreachable machine is not evidence that its work
            // has gone.
            guard hosts.isUp(route), let backend = hosts.backend(for: host) else {
                throw MoldClientError.unreachable("it is not answering")
            }
            result = .success(try await backend.activity())
        } catch {
            result = .failure(error)
        }
        // An answer under an older number is about a read this app has
        // already replaced.
        guard epochs[host] == epoch else { return }
        byHost[host] = ActivityReconcile.host(
            routeURL: route.baseURL.absoluteString,
            previous: byHost[host], result: result)
    }
}
