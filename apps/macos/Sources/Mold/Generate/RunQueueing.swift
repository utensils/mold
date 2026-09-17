import Foundation
import MoldClient

/// Dropping and withdrawing work that is waiting for the canvas.
///
/// A free type rather than more `GenerateController`: the two kinds of queued
/// run are forgotten and cancelled through DIFFERENT routes, and putting that
/// fork in the controller would be behaviour it has no room for.
@MainActor
enum RunQueueing {
    /// Drops the local recovery record for work nothing can follow any more --
    /// the machine it belongs to is gone.
    static func forget(_ run: QueuedRun) {
        switch run {
        case let .batch(batch): PendingBatch.forget(batch.clientBatchId)
        case let .chain(chain): PendingChain.forget(chain.jobId)
        }
    }

    /// Withdraws it on its own machine. A queued chain is a REAL job on the
    /// host, not a local intention -- Stop All must reach it there or it keeps
    /// the GPU with nobody watching.
    static func withdraw(_ run: QueuedRun, on controller: GenerateController) {
        forget(run)
        switch run {
        case let .batch(batch):
            controller.cancelOnItsMachine(batch)
        case let .chain(chain):
            guard let backend = controller.hosts.backend(for: chain.host) else { return }
            Task { [hosts = controller.hosts] in
                do { try await backend.cancelChainJob(id: chain.jobId) }
                catch { hosts.report(error, on: chain.host, doing: "cancel that render") }
            }
        }
    }
}
