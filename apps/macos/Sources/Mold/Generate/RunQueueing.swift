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

    /// Puts it on the canvas. A batch is followed on its batch stream; a
    /// chain job the host already holds is RE-ATTACHED, which is the same
    /// thing a relaunch does with a job it finds in `PendingChain`.
    static func follow(
        _ run: QueuedRun, on controller: GenerateController, backend: any MoldBackend
    ) {
        switch run {
        case let .batch(batch):
            controller.activeBatch = batch
            controller.runTask = Task { [weak controller] in
                await controller?.follow(batch.admitted, backend: backend, host: batch.host)
            }
        case let .chain(chain):
            controller.chain.reattach(
                jobId: chain.jobId, stageCount: chain.stageCount, on: chain.host,
                backend: backend, report: ChainSubmission.reporter(for: controller))
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
