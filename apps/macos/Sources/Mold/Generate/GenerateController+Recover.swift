import Foundation
import MoldClient

/// Recovering a render whose admission response never made it back --
/// a crash, a force-quit, a dropped connection while a batch was in flight.
///
/// mold's queue is idempotent on the client batch id, so recovery asks the
/// host what happened rather than submitting again, which would render the
/// same thing twice and bill the GPU for both.
@MainActor
extension GenerateController {
    /// Runs once, on the pane's first appearance. A pane already mid-run
    /// (relaunch is instant; this only matters after a restart) has nothing
    /// to recover into.
    func recoverPending() async {
        guard case .idle = run else { return }
        for (clientBatchId, hostID) in PendingBatch.all() {
            guard let uuid = UUID(uuidString: hostID), let host = hosts.host(uuid) else {
                // The machine that would own this batch is gone.
                PendingBatch.forget(clientBatchId)
                continue
            }
            let backend = hosts.backend(for: host)
            do {
                let status = try await backend.batchStatus(clientBatchId: clientBatchId)
                guard !status.isSettled else {
                    PendingBatch.forget(clientBatchId)
                    continue
                }
                activeBatch = (status.id, clientBatchId, host.id)
                await follow(status, backend: backend, host: host.id)
            } catch let error as MoldClientError where !error.isTransient {
                // Unknown to the host (e.g. a 404) or refused outright --
                // there is nothing left to recover.
                PendingBatch.forget(clientBatchId)
            } catch {
                // A bad minute on the network or the machine. Leave it for
                // the next launch rather than guessing it is gone.
            }
        }
    }
}
