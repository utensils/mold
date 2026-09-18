import Foundation
import MoldClient

// Putting recovered chain jobs back on the canvas and in the queue. Split from
// the asking, and kept OUT of `GenerateController`, which has no room for it.
@MainActor
extension PendingChainRecovery {
    /// Re-attaches what this Mac admitted and never followed to settlement.
    ///
    /// A chain job outlives a quit -- and a host restart PARKS it as `paused`
    /// rather than losing it (CLAUDE.md, "Scripted sequences"). Re-attaching
    /// is following it again; nothing is ever created twice.
    ///
    /// The FIRST live job takes the canvas and the rest wait in the run queue,
    /// which is where a second press would have put them anyway. A job that
    /// finished while the app was closed is SHOWN rather than dropped
    /// silently: the render happened, and a record vanishing with no picture
    /// is how somebody comes to wonder whether it ever ran.
    static func reattach(on controller: GenerateController) async {
        let found = await resolve(hosts: controller.hosts)
        for resumable in found.resumable {
            guard let backend = controller.hosts.backend(for: resumable.host) else { continue }
            let admitted = AdmittedChain(jobId: resumable.jobId,
                                         stageCount: resumable.stageCount,
                                         host: resumable.host)
            guard case .idle = controller.run else {
                controller.queued.append(.chain(admitted))
                continue
            }
            controller.chain.reattach(
                jobId: admitted.jobId, stageCount: admitted.stageCount, on: admitted.host,
                backend: backend, report: ChainSubmission.reporter(for: controller),
                currentStage: resumable.currentStage, isPaused: resumable.isPaused)
        }
        guard case .idle = controller.run, let last = found.finished.last else { return }
        controller.run = .finished(
            BatchOutcome(chainResults: [BatchResult(filename: last.filename)], failures: []),
            host: last.host)
    }
}
