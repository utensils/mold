import Foundation
import MoldClient

// How often this app is allowed to ask a machine for its queue. Split from
// `QueueStore.swift` past the file-size advisory; the read itself (`poll`)
// stays there, because every mutation calls it directly.
@MainActor
extension QueueStore {
    func refresh() async {
        isLoading = true
        defer { isLoading = false }
        await withTaskGroup(of: Void.self) { group in
            for host in hosts.hosts {
                group.addTask { await self.refresh(on: host.id) }
            }
        }
    }

    /// One machine's queue, then its batches -- what every fallback calls.
    ///
    /// Throttled by `SingleFlight`: one read at a time per machine, at most
    /// one queued behind it. See that type for what happens without it.
    func refresh(on host: MoldHost.ID) async {
        await reads.once(per: host) { [weak self] host in
            await self?.readOnce(on: host)
        }
    }

    /// The actual read. Not `private`: the throttle calls it from a closure
    /// this extension hands over, and it is the one place the two halves are
    /// sequenced.
    func readOnce(on host: MoldHost.ID) async {
        await poll(host)
        // A superseded read stops here rather than spending a batch-status
        // call on a listing nobody is waiting for.
        guard !Task.isCancelled else { return }
        await hydrate(on: host)
    }
}
