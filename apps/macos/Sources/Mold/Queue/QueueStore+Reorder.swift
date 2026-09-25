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
    /// restart-paused rows, and every HELD row. The bulk route
    /// (`DELETE /api/queue`, `routes.rs:7869-7898`) deliberately leaves holds
    /// alone, so "Empty Queue" used to leave a pane full of them; each hold is
    /// cleared the way its own × clears it, `DELETE /api/queue/:id`, which
    /// settles a held child as cancelled (`generation_batches.rs:710`).
    /// Running work is untouched either way.
    ///
    /// Reads the listing FIRST: the holds to clear are the machine's, not
    /// whatever this store last saw.
    func empty(on host: MoldHost.ID) async {
        let verb = QueueEmptyConfirm.verb
        guard !refuseIfFixture(host, doing: verb) else { return }
        guard let client = hosts.backend(for: host) else { return }
        await poll(host)
        let rows = entries(on: host)
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
        // without the bulk route: a row can start between this listing and
        // its DELETE, and the per-row route cancels running work -- which
        // the confirm promises never happens. A HELD row cannot start.
        for row in rows where row.state == .held {
            failed = await cancelRow(row.id, via: client, on: host, doing: verb) || failed
        }
        if !failed { hosts.succeeded(on: host, doing: verb) }
        await poll(host)
    }

    /// `empty(on:)` for every machine at once, concurrently -- each one
    /// reports its own failure line, so one unreachable machine never hides
    /// what the others did.
    func emptyAll(_ targets: [MoldHost.ID]) async {
        await withTaskGroup(of: Void.self) { group in
            for host in targets {
                group.addTask { await self.empty(on: host) }
            }
        }
    }

    /// `true` when the machine refused.
    private func cancelRow(_ id: String, via client: any MoldBackend,
                           on host: MoldHost.ID, doing verb: String) async -> Bool {
        do {
            try await client.cancelJob(id: id)
            return false
        } catch {
            hosts.report(error, on: host, doing: verb)
            return true
        }
    }
}
