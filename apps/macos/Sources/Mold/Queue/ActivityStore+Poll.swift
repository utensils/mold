import Foundation
import MoldClient

// One tick: ask every machine, reconcile each answer against what it last
// said.
@MainActor
extension ActivityStore {
    func canAct(_ action: AlsoRunningActions.Kind, on row: FleetActiveWork) -> Bool {
        guard !acting.contains(row.id), !row.stale, !row.unavailableKind,
              let host = hosts.host(row.host), hosts.isUp(host),
              let snapshot = byHost[row.host], !snapshot.stale,
              snapshot.routeURL == host.baseURL.absoluteString,
              let instance = snapshot.instanceId, hosts.instanceID(of: row.host) == instance,
              let current = snapshot.items.first(where: { $0.id == row.item.id && $0.kind == row.item.kind && $0.execution == row.item.execution }),
              current.phase == row.item.phase else { return false }
        let currentRow = AlsoRunningRow(host: row.host, work: .reported(FleetActiveWork(host: row.host, item: current, stale: false, unavailableKind: snapshot.unavailableKinds.contains(current.authorityKind))))
        switch action {
        case .cancel: return currentRow.canCancel
        case .resume: return currentRow.canResume
        default: return false
        }
    }

    func act(_ action: AlsoRunningActions.Kind, on row: FleetActiveWork) async {
        guard canAct(action, on: row), let host = hosts.host(row.host),
              let instance = byHost[row.host]?.instanceId, let backend = hosts.backend(for: row.host) else { return }
        acting.insert(row.id)
        defer { acting.remove(row.id) }
        let verb = action == .resume ? "resume that long clip" : "cancel that long clip"
        do {
            let status = try await backend.status()
            let fresh = try await backend.activity()
            guard status.instanceId == instance, fresh.instanceId == instance,
                  hosts.host(row.host) == host, hosts.instanceID(of: row.host) == instance, hosts.isUp(host),
                  let current = fresh.items.first(where: { $0.id == row.item.id && $0.kind == "generation" && $0.execution == "chain" }),
                  !fresh.unavailableKinds.contains(current.authorityKind), current.phase == row.item.phase else {
                await refresh(on: row.host)
                return
            }
            let freshRow = AlsoRunningRow(host: row.host, work: .reported(FleetActiveWork(host: row.host, item: current, stale: false, unavailableKind: false)))
            switch action {
            case .cancel:
                guard freshRow.canCancel else { await refresh(on: row.host); return }
                try await backend.cancelChainJob(id: current.id)
            case .resume:
                guard freshRow.canResume else { await refresh(on: row.host); return }
                try await backend.resumeChainJob(id: current.id)
            default: return
            }
            hosts.succeeded(on: row.host)
        } catch { hosts.report(error, on: row.host, doing: verb) }
        await refresh(on: row.host)
    }

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
