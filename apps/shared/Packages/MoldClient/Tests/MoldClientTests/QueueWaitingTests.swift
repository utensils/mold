import Testing
@testable import MoldClient

struct QueueWaitingTests {
    @Test func waitingBadgeCountsWorkStillWaitingToRun() {
        for state in [QueueState.queued, .paused, .held] {
            #expect(state.isWaiting)
        }
        for state in [QueueState.running, .cancelling, .complete, .failed, .cancelled, .unknown] {
            #expect(!state.isWaiting)
        }
    }
}
