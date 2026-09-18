import Foundation
import MoldClient

/// Where a submitted render stands, as the pane needs to show it.
enum RunState {
    case idle
    case submitting
    case running(BatchStatus, JobProgress?)
    /// A clip too long for one denoise, running as an EPHEMERAL chain job.
    /// Its own arm rather than a synthesized `BatchStatus`: there is no batch
    /// -- `POST /api/chain-jobs` mints a different kind of id, and pretending
    /// otherwise would let Stop cancel it through the wrong route.
    case runningChain(ChainProgress)
    case finished(BatchOutcome, host: MoldHost.ID)
    case failed(String)

    /// The machine a finished batch actually ran on -- which is where its
    /// bytes are. The canvas and the result bar used to fetch from the pane's
    /// CURRENT machine, so switching machines (or a default changing under
    /// you) after pressing Generate turned a finished render into "That
    /// didn't arrive · Image not found" with Save and Copy still offered
    /// (2026-09-17).
    var finishedHost: MoldHost.ID? {
        guard case let .finished(_, host) = self else { return nil }
        return host
    }

    var isBusy: Bool {
        switch self {
        case .submitting, .running, .runningChain: true
        case .idle, .finished, .failed: false
        }
    }

    /// The denoise step counter, when there is one.
    var steps: (done: Int, total: Int)? {
        if case let .runningChain(chain) = self {
            guard let step = chain.step, let total = chain.total, total > 0 else { return nil }
            return (step, total)
        }
        guard case let .running(_, progress) = self,
              let step = progress?.step, let total = progress?.total, total > 0
        else { return nil }
        return (step, total)
    }

    var stage: String? {
        // A chain's stage IS its clip counter -- the one thing on screen that
        // says a long render was made in pieces.
        if case let .runningChain(chain) = self { return chain.label }
        guard case let .running(_, progress) = self else { return nil }
        return progress?.stage
    }

    var previewData: Data? {
        guard case let .running(_, progress) = self else { return nil }
        return progress?.previewData
    }
}
