import Foundation
import Testing
import MoldClientTesting
@testable import MoldClient

@MainActor struct VideoPlanControllerTests {
    func draft(_ seconds: UInt64 = 30_000) -> VideoJobRequest {
        .init(generation: .init(prompt: "one prompt", model: "model", width: 768, height: 768, steps: 2, guidance: 1), durationMs: seconds)
    }
    func plan() throws -> VideoJobPlan {
        try MoldJSON.decoder.decode(VideoJobPlan.self,
            from: Data(VideoJobContractTests.plan().replacingOccurrences(of: "\"validation_revision\":\"revision\"", with: "\"validation_revision\":\"\"").utf8))
    }
    @Test func debouncedPreviewIsReadOnlyScalarsAndPublishesOnlyCurrentOwner() async throws {
        let backend = FakeBackend(), plan = try plan()
        backend.stub("planVideoJob(_:)", returning: plan)
        let controller = VideoPlanController(debounceNanoseconds: 1_000_000)
        var request = draft(); request.generation.sourceImage = "PRIVATE_PIXELS"
        controller.update(request, backend: backend, current: { true })
        controller.update(request, backend: backend, current: { true })
        await controller.waitForPendingPlan()
        #expect(backend.count("planVideoJob(_:)") == 1)
        let argument = try #require(backend.calls.first?.arguments.first as? VideoJobRequest)
        #expect(argument.generation.sourceImage == nil)
        #expect(controller.plan?.clipCount == 4)
        #expect(!controller.isCalculating)
        #expect(backend.count("createVideoJob(_:operationId:destinationInstanceId:)") == 0)
    }
    @Test func lateNoncooperativePlanCannotOverwriteEditedDurationOrChangedOwner() async throws {
        let backend = FakeBackend(), plan = try plan()
        let gate = AsyncStream<Void>.makeStream()
        backend.stub("planVideoJob(_:)") { _ in
            for await _ in gate.stream { break }
            return plan
        }
        let controller = VideoPlanController(debounceNanoseconds: 0)
        var current = true
        let old = controller.update(draft(), backend: backend, current: { current })
        for _ in 0..<100 where backend.count("planVideoJob(_:)") == 0 { await Task.yield() }
        current = false
        controller.clear()
        gate.continuation.finish()
        await old.value
        #expect(controller.plan == nil)
        #expect(controller.error == nil)
        #expect(!controller.isCalculating)
    }
    @Test func recalculatingKeepsTheLastPlanOnScreenUntilTheNextAnswer() async throws {
        let backend = FakeBackend(), plan = try plan()
        backend.stub("planVideoJob(_:)", returning: plan)
        let controller = VideoPlanController(debounceNanoseconds: 1_000_000)
        await controller.update(draft(), backend: backend, current: { true }).value
        #expect(controller.shownPlan?.clipCount == 4)
        let next = controller.update(draft(31_000), backend: backend, current: { true })
        #expect(controller.plan == nil, "submission checks never read a plan for another length")
        #expect(controller.shownPlan?.clipCount == 4, "a slider drag must not blank the plan between answers")
        await next.value
        #expect(controller.shownPlan?.clipCount == 4)
        controller.clear()
        #expect(controller.shownPlan == nil, "a changed owner or model drops the shown plan too")
    }
    @Test func anotherModelOrMachineNeverBorrowsTheShownPlan() async throws {
        let backend = FakeBackend(), plan = try plan()
        backend.stub("planVideoJob(_:)", returning: plan)
        let controller = VideoPlanController(debounceNanoseconds: 1_000_000)
        await controller.update(draft(), backend: backend, destinationInstanceId: "a", current: { true }).value
        var other = draft(); other.generation.model = "other-model"
        let next = controller.update(other, backend: backend, destinationInstanceId: "a", current: { true })
        #expect(controller.shownPlan == nil, "another model's window must not bound this model's slider")
        await next.value
        controller.update(other, backend: backend, destinationInstanceId: "b", current: { true })
        #expect(controller.shownPlan == nil, "another installation's plan is not this one's")
    }
    @Test func aFailedRecalculationDropsTheShownPlan() async throws {
        let backend = FakeBackend(), plan = try plan()
        backend.stub("planVideoJob(_:)", returning: plan)
        let controller = VideoPlanController(debounceNanoseconds: 1_000_000)
        await controller.update(draft(), backend: backend, current: { true }).value
        backend.stub("planVideoJob(_:)", throwing: MoldClientError.unreachable("offline"))
        await controller.update(draft(45_000), backend: backend, current: { true }).value
        #expect(controller.error != nil)
        #expect(controller.shownPlan == nil, "an old total must not sit beside the error")
    }
    @Test func changedOwnerDuringDebounceDoesNotLeaveTheEditorCalculating() async throws {
        let backend = FakeBackend()
        let controller = VideoPlanController(debounceNanoseconds: 1_000_000)
        var current = true
        let old = controller.update(draft(), backend: backend, current: { current })
        current = false
        await old.value
        #expect(backend.calls.isEmpty)
        #expect(!controller.isCalculating)
        #expect(controller.plan == nil)
    }

}
