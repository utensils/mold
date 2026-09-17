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
    static func take(
        _ routing: ChainRouting.Decision, request: GenerateRequest,
        on host: MoldHost, backend: any MoldBackend, controller: GenerateController
    ) -> Bool {
        switch routing {
        case .single:
            return false
        case let .reject(reason):
            // `mold_core::chain::text_only_auto_chain_refusal`'s own words,
            // so this app, the CLI and the server's 422 read the same.
            controller.run = .failed(reason)
            return true
        case let .chain(clipFrames, motionTail, stageCount):
            controller.run = .submitting
            controller.runTask?.cancel()
            controller.chain.start(
                AutoChainRequest(request, clipFrames: clipFrames, motionTail: motionTail),
                stageCount: stageCount, on: host.id, backend: backend,
                report: reporter(for: controller))
            return true
        }
    }

    /// What the follow reports back onto the pane.
    private static func reporter(for controller: GenerateController) -> ChainRun.Reporter {
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
            })
    }
}
