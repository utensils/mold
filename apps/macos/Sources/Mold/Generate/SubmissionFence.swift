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
/// This is the fence, and it is KEYED BY CLIENT BATCH ID rather than a single
/// slot. A single slot lost the stop the moment Generate was pressed again --
/// `run` goes `.idle` on Stop, so the button is live immediately -- and the
/// withdrawn batch then landed as an ordinary queued one and took the canvas
/// later, never cancelled. A stop belongs to the ID it was aimed at and
/// survives any number of later submissions.
@MainActor
final class SubmissionFence {
    /// What the in-flight admission should do once the host answers.
    enum Landing: Equatable {
        /// Follow it, as usual.
        case follow
        /// Stop was pressed for THIS id while it was in the air: cancel the
        /// batch the host just minted and drop its recovery record.
        case cancel
        /// Another submission took the canvas while this one was in the air.
        /// It is a perfectly good batch -- it queues, exactly as a second
        /// press would have (M8 decision 8).
        case queue
    }

    /// The id the canvas is following, while its POST is unanswered.
    private var following: String?
    /// Every POST still in the air, stopped or not. A submit task must never
    /// be cancelled while one of these is outstanding.
    private var unanswered: Set<String> = []
    /// Ids whose Stop was pressed before the host answered.
    private var stopped: Set<String> = []

    /// Whether any POST at all is still unanswered -- including one the user
    /// has already stopped, which still has to reach its `land` so the batch
    /// the host minted can be cancelled.
    var hasUnansweredPost: Bool { !unanswered.isEmpty }

    func begin(_ clientBatchId: String) {
        following = clientBatchId
        unanswered.insert(clientBatchId)
        // A fresh id cannot carry an older id's stop, and must not clear one.
        stopped.remove(clientBatchId)
    }

    /// What to do with the admission that just landed. A recorded stop wins
    /// over everything else, whatever has taken the canvas since.
    func land(_ clientBatchId: String) -> Landing {
        unanswered.remove(clientBatchId)
        guard stopped.remove(clientBatchId) == nil else {
            if following == clientBatchId { following = nil }
            return .cancel
        }
        guard following == clientBatchId else { return .queue }
        following = nil
        return .follow
    }

    /// Records that Stop was pressed. `false` means nothing was in the air,
    /// which is the caller's cue to stop the batch on screen instead.
    ///
    /// The id stops being the FOLLOWED one at once: a render the user
    /// withdrew must never take the canvas when its answer arrives.
    func requestStop() -> Bool {
        guard let following else { return false }
        stopped.insert(following)
        self.following = nil
        return true
    }
}
