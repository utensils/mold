import Foundation

/// Stop, pressed while a submission is still in the air.
///
/// The Stop button renders for the whole of `.submitting`, but at that moment
/// `activeBatch` still names the PREVIOUS batch -- or, on a first-ever render,
/// nothing at all. So Stop cancelled an already-settled batch on the host,
/// forgot the wrong `PendingBatch` record, and did nothing whatsoever the
/// first time; meanwhile the POST that was already on its way was admitted,
/// rendered to completion, was never cancelled, and was no longer in
/// `PendingBatch` for launch recovery to find. The user pressed Stop, saw an
/// error banner, and the GPU kept going (finding 02#2).
///
/// This is the fence. One admission at a time, and a flag saying whether Stop
/// was pressed before the host answered -- the submit task asks on the way out
/// and cancels the id the HOST returned, which is the only id that can be
/// cancelled.
@MainActor
final class SubmissionFence {
    /// What the in-flight admission should do once the host answers.
    enum Landing: Equatable {
        /// Follow it, as usual.
        case follow
        /// Stop was pressed while it was in the air: cancel the id the host
        /// just minted and drop its recovery record.
        case cancel
        /// Another submission took the canvas while this one was in the air.
        /// It is a perfectly good batch -- it queues, exactly as a second
        /// press would have (M8 decision 8).
        case queue
    }

    private var inFlight: String?
    private var stopRequested = false

    /// Whether a submission the canvas is following is still unanswered.
    var isPending: Bool { inFlight != nil }

    func begin(_ clientBatchId: String) {
        inFlight = clientBatchId
        stopRequested = false
    }

    /// What to do with the admission that just landed.
    func land(_ clientBatchId: String) -> Landing {
        guard inFlight == clientBatchId else { return .queue }
        inFlight = nil
        defer { stopRequested = false }
        return stopRequested ? .cancel : .follow
    }

    /// Records that Stop was pressed. `false` means nothing was in the air,
    /// which is the caller's cue to stop the batch on screen instead.
    func requestStop() -> Bool {
        guard inFlight != nil else { return false }
        stopRequested = true
        return true
    }
}
