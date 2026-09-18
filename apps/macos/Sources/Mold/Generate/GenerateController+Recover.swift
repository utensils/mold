import Foundation
import MoldClient

// Following a render `PendingRecovery` found still live after a relaunch.
@MainActor
extension GenerateController {
    /// Runs once, on the pane's first appearance. A pane already mid-run
    /// (relaunch is instant; this only matters after a restart) has nothing
    /// to recover into.
    func recoverPending() async {
        guard case .idle = run else { return }
        await PendingChainRecovery.reattach(on: self)
        guard case .idle = run else { return }
        for found in await PendingRecovery.resolve(hosts: hosts) {
            guard let backend = hosts.backend(for: found.host) else { continue }
            activeBatch = ActiveBatch(
                id: found.status.id, clientBatchId: found.clientBatchId,
                host: found.host, admitted: found.status)
            await follow(found.status, backend: backend, host: found.host)
        }
    }
}
