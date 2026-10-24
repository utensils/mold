import Foundation
import Testing
import MoldClientTesting
@testable import MoldClient

extension VideoJobCoordinatorTests {
    @Test func finishedParentsLeaveTheQueueLikeEveryOtherRow() async throws {
        let (_, owner, defaults) = try setup()
        let coordinator = VideoJobCoordinator(defaults: defaults)
        coordinator.publish(try detail("done", state: "completed", cancel: false), owner: owner)
        coordinator.publish(try detail("stopped", state: "cancelled", cancel: false), owner: owner)
        #expect(coordinator.jobs.allSatisfy { !$0.belongsInQueue })
        #expect(!coordinator.hasQueueContent, "a finished video must not keep an otherwise-empty Queue open")
        for state in ["queued", "running", "paused", "failed", "interrupted"] {
            coordinator.publish(try detail(state, state: state), owner: owner)
        }
        #expect(coordinator.jobs.filter(\.belongsInQueue).map(\.operationId)
            == ["queued", "running", "paused", "failed", "interrupted"])
        #expect(coordinator.hasQueueContent)
        var unconfirmed = ObservedVideoJob(operationId: "checking", owner: owner)
        #expect(unconfirmed.belongsInQueue, "an admission still being confirmed is live work")
        unconfirmed.cancelledBeforeAdmission = true
        #expect(!unconfirmed.belongsInQueue)
    }
}
