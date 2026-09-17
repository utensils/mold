import Foundation
import MoldClient

/// Where a submitted render stands, as the pane needs to show it.
enum RunState {
    case idle
    case submitting
    case running(BatchStatus, JobProgress?)
    case finished(BatchOutcome, host: MoldHost.ID)
    case failed(String)

    var isBusy: Bool {
        switch self {
        case .submitting, .running: true
        case .idle, .finished, .failed: false
        }
    }

    /// The denoise step counter, when there is one.
    var steps: (done: Int, total: Int)? {
        guard case let .running(_, progress) = self,
              let step = progress?.step, let total = progress?.total, total > 0
        else { return nil }
        return (step, total)
    }

    var stage: String? {
        guard case let .running(_, progress) = self else { return nil }
        return progress?.stage
    }

    var previewData: Data? {
        guard case let .running(_, progress) = self else { return nil }
        return progress?.previewData
    }
}
