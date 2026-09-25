import Foundation
import MoldClient

/// Reordering the machine's queue, and clearing it in one call. Neither
/// route exists anywhere else in this store: `QueuePane` computes WHAT to
/// send (`QueueOrder`, `QueuePane.reorderCalls`) and this only sends it.
@MainActor
extension QueueStore {
    /// One PATCH per call, in the order given -- ascending target order is
    /// what lands N children of a dragged batch contiguous (design M6 fact
    /// 3), and this issues whatever order it is handed rather than choosing
    /// one itself. Ends in the one re-read the row's new position is the
    /// server's to state.
    func reorder(_ calls: [(id: String, position: Int)], on host: MoldHost.ID) async {
        guard !refuseIfFixture(host, doing: "reorder its queue") else { return }
        guard let client = hosts.backend(for: host) else { return }
        for call in calls {
            do {
                try await client.reorderJob(id: call.id, position: call.position)
                hosts.succeeded(on: host)
            } catch {
                hosts.report(error, on: host, doing: "reorder its queue")
            }
        }
        await poll(host)
    }

    /// Everything on ONE machine that is not rendering: the waiting and
    /// restart-paused rows, and the HELD rows the confirm counted. The bulk
    /// route (`DELETE /api/queue`, `routes.rs:7869-7898`) deliberately leaves
    /// holds alone, so "Empty Queue" used to leave a pane full of them; each
    /// hold is cleared with `DELETE /api/queue/:id?only_held=true`, which the
    /// machine refuses once a Retry has moved the row -- so this never stops
    /// a render. Running work is untouched either way.
    ///
    /// `held` is the set the person was shown. A job that became held while
    /// the confirm was open was not counted, so it is not cleared.
    func empty(on host: MoldHost.ID, held: Set<String>) async {
        let verb = QueueEmptyConfirm.verb
        guard !refuseIfFixture(host, doing: verb) else { return }
        guard let client = hosts.backend(for: host) else { return }
        var failed = false
        if hosts.capabilities[host]?.canCancelAllQueued == true {
            do {
                try await client.cancelAllQueued()
            } catch {
                hosts.report(error, on: host, doing: verb)
                failed = true
            }
        }
        // Deliberately NO per-row fallback for WAITING rows on a machine
        // without the bulk route: a row can start between the listing and
        // its DELETE, and the per-row route cancels running work -- which
        // the confirm promises never happens.
        for id in held.sorted() {
            do {
                _ = try await client.cancelHeldJob(id: id)
            } catch {
                hosts.report(error, on: host, doing: verb)
                failed = true
            }
        }
        if !failed { hosts.succeeded(on: host, doing: verb) }
        await poll(host)
    }

    /// The held rows this store is showing for a machine -- what a confirm
    /// counts and hands to `empty(on:held:)`.
    func heldIDs(on host: MoldHost.ID) -> Set<String> {
        Set(entries(on: host).filter { $0.state == .held }.map(\.id))
    }

    /// `empty(on:held:)` for every machine at once, concurrently -- each one
    /// reports its own failure line, so one unreachable machine never hides
    /// what the others did.
    func emptyAll(_ targets: [MoldHost.ID: Set<String>]) async {
        await withTaskGroup(of: Void.self) { group in
            for (host, held) in targets {
                group.addTask { await self.empty(on: host, held: held) }
            }
        }
    }
}
