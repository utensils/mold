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

    /// Cancels every queued or restart-paused row on ONE machine. Running
    /// work is untouched (`routes.rs:7854-7859`) -- the confirm that leads
    /// here says so, and the re-read afterward is what actually shows it.
    func cancelAll(on host: MoldHost.ID) async {
        guard let client = hosts.backend(for: host) else { return }
        do {
            try await client.cancelAllQueued()
            hosts.succeeded(on: host)
        } catch {
            hosts.report(error, on: host, doing: "cancel everything waiting")
        }
        await poll(host)
    }
}
