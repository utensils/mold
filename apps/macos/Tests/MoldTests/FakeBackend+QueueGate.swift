import Foundation
import MoldClient
@testable import Mold

// The whole-queue gate on the fake. Its state lives in `FakeExtras`, for the
// reason that type's own comment gives.
extension FakeBackend {
    /// Answers the gate's NEW state, the way the server does -- so a test can
    /// plant a machine that refuses to move (`gateAnswers`) and prove the
    /// store writes what the machine said rather than what it asked for.
    func pauseQueue() async throws -> QueuePauseState {
        try record("pauseQueue")
        await pause("pauseQueue")
        extras.gateCalls.append(true)
        return QueuePauseState(paused: extras.gateAnswer ?? true)
    }

    func resumeQueue() async throws -> QueuePauseState {
        try record("resumeQueue")
        await pause("resumeQueue")
        extras.gateCalls.append(false)
        return QueuePauseState(paused: extras.gateAnswer ?? false)
    }
}
