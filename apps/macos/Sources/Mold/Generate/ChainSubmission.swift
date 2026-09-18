import Foundation
import MoldClient

/// The branch a routed render takes instead of an ordinary batch.
///
/// A free type rather than more `GenerateController`: the controller asks ONE
/// question here and everything about the chain lifecycle lives in `ChainRun`
/// beside it.
@MainActor
enum ChainSubmission {
    /// Takes the render when the routing says it is not one denoise.
    ///
    /// `true` means this has been dealt with -- admitted as an ephemeral chain
    /// job, or refused BY NAME with the server's own sentence -- and the batch
    /// path must not also run. `false` is an ordinary single render.
    ///
    /// A chain NEVER preempts what is on the canvas and never cancels a POST
    /// in flight: `runTask` is the whole submit-and-follow task for a batch,
    /// and cancelling it here left that batch rendering with nobody holding
    /// its id, its result never drawn, and a failure banner over a chain that
    /// was running perfectly well. It is admitted and waits its turn, which is
    /// what M8 decision 8 says for every press.
    static func take(
        _ routing: ChainRouting.Decision, requests: [GenerateRequest],
        on host: MoldHost, backend: any MoldBackend, controller: GenerateController
    ) -> Bool {
        guard !requests.isEmpty else { return false }
        switch routing {
        case .single:
            return false
        case let .reject(reason):
            // `mold_core::chain::text_only_auto_chain_refusal`'s own words,
            // so this app, the CLI and the server's 422 read the same.
            controller.run = .failed(reason)
            return true
        case let .chain(clipFrames, motionTail, stageCount):
            // Four copies of a long clip are FOUR chains, one per seed. One
            // press used to build a single request and ignore `batchSize`
            // entirely, so asking for four silently rendered one.
            let bodies = requests.map {
                AutoChainRequest($0, clipFrames: clipFrames, motionTail: motionTail)
            }
            // Decided HERE, synchronously, before any `Task` is scheduled --
            // the same rule `submit` follows, so two presses in one turn can
            // never both think they are the one being followed.
            var following = !controller.run.isBusy
            for body in bodies {
                if following {
                    controller.run = .submitting
                    controller.chain.start(body, stageCount: stageCount, on: host.id,
                                           backend: backend, report: reporter(for: controller))
                    following = false
                } else {
                    admitAndQueue(body, stageCount: stageCount, on: host,
                                  backend: backend, controller: controller)
                }
            }
            return true
        }
    }

    /// Creates the job and parks it in the run queue. The host's queue is
    /// durable, so the work is real from the moment it is admitted; nothing
    /// watches it until the canvas is free.
    private static func admitAndQueue(
        _ body: AutoChainRequest, stageCount: Int, on host: MoldHost,
        backend: any MoldBackend, controller: GenerateController
    ) {
        Task { [weak controller] in
            do {
                let created = try await backend.createChainJob(
                    body, operationId: UUID().uuidString)
                guard let controller else {
                    // Nobody left to follow it: withdraw rather than leave a
                    // GPU rendering for no one.
                    try? await backend.cancelChainJob(id: created.jobId)
                    return
                }
                PendingChain.remember(created.jobId, host: host.id)
                controller.queued.append(.chain(AdmittedChain(
                    jobId: created.jobId, stageCount: stageCount, host: host.id)))
            } catch {
                // The render on screen is unaffected by a second one failing
                // to be admitted -- report it, never replace `run`.
                controller?.hosts.report(error, on: host.id, doing: "queue that render")
            }
        }
    }

    /// What the follow reports back onto the pane. Not `private`:
    /// `followNext` re-attaches a queued chain with the same reporter.
    static func reporter(for controller: GenerateController) -> ChainRun.Reporter {
        ChainRun.Reporter(
            progress: { [weak controller] progress in
                controller?.run = .runningChain(progress)
            },
            finished: { [weak controller] filename, host in
                guard let controller else { return }
                // ONE print, one canvas result -- a stitched long video is a
                // single render that happened to be made in pieces. The seed
                // and timing belong to its stages, not to the print, so they
                // are absent rather than invented.
                let outcome = BatchOutcome(
                    chainResults: filename.map { [BatchResult(filename: $0)] } ?? [],
                    failures: filename == nil
                        ? ["The clip finished, but the machine published no file for it."]
                        : [])
                controller.run = outcome.results.isEmpty
                    ? .failed(outcome.failures.first ?? "The render didn't finish.")
                    : .finished(outcome, host: host)
                // The queue moves exactly as it does for a batch: the next one
                // takes the canvas once this outcome has been drawn on it.
                controller.handoff.hold { [weak controller] in controller?.followNext() }
            },
            failed: { [weak controller] message in
                controller?.run = .failed(message)
                controller?.handoff.hold { [weak controller] in controller?.followNext() }
            })
    }
}
