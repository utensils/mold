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
