import Foundation
import Testing
import MoldClientTesting
@testable import MoldClient

extension VideoJobCoordinatorTests {
    @Test func lostStopReplySurvivesRelaunchAndUsesOnlyTheKnownParentCancellationFence() async throws {
        let (backend, owner, saved) = try setup()
        backend.stub("createVideoJob(_:operationId:destinationInstanceId:)", throwing: MoldClientError.unreachable("Lost reply"))
        let original = VideoJobCoordinator(defaults: saved)
        await original.submit(request(), owner: owner, backend: backend, current: { true }, unchanged: { true })
        let row = try #require(original.jobs.first)
        backend.stub("cancelVideoJob(id:destinationInstanceId:)", throwing: MoldClientError.unreachable("Lost Stop reply"))
        await original.cancel(row, backend: backend, current: { true })
        #expect(original.pending.first?.cancelRequested == true)
        #expect(original.pending.first?.cancelAcknowledged == false)
        original.suspend()
        backend.stub("cancelVideoJob(id:destinationInstanceId:)", returning: ())
        backend.stub("videoJob(id:)", throwing: MoldClientError.http(status: 410, code: "VIDEO_OPERATION_CANCELLED", message: "Cancelled"))
        let recovered = VideoJobCoordinator(defaults: saved)
        await recovered.recover(owner: owner, backend: backend, current: { true })
        #expect(recovered.pending.isEmpty)
        #expect(recovered.jobs.first?.cancelledBeforeAdmission == true)
        let stops = backend.calls.filter { $0.route == "cancelVideoJob(id:destinationInstanceId:)" }
        #expect(stops.count == 2)
        #expect(stops.allSatisfy { $0.arguments[0] as? String == row.operationId && $0.arguments[1] as? String == owner.instanceId })
        #expect(backend.count("createVideoJob(_:operationId:destinationInstanceId:)") == 1)
        recovered.suspend()
    }

    @Test func lateStatusCannotResurrectAnExplicitlyDeletedRecoveryJob() async throws {
        let (backend, owner, saved) = try setup()
        let coordinator = VideoJobCoordinator(defaults: saved)
        let failed = try detail("parent", state: "failed", cancel: false, resume: true, delete: true)
        coordinator.publish(failed, owner: owner)
        let row = try #require(coordinator.jobs.first)
        let gate = AsyncStream<Void>.makeStream()
        backend.stub("videoJob(id:)") { _ in
            for await _ in gate.stream { break }
            return failed
        }
        backend.stub("deleteVideoJob(id:destinationInstanceId:)", returning: ())
        let old = Task { await coordinator.refresh("parent", owner: owner, backend: backend, current: { true }) }
        for _ in 0..<100 where backend.count("videoJob(id:)") == 0 { await Task.yield() }
        await coordinator.delete(row, backend: backend, current: { true })
        gate.continuation.finish(); await old.value
        #expect(coordinator.jobs.isEmpty)
    }

    @Test func resumeClearsOldCancellationIntentDurablyBeforeMutation() async throws {
        let (backend, owner, saved) = try setup()
        let coordinator = VideoJobCoordinator(defaults: saved)
        var receipt = VideoJobAdmission(operationId: "parent", owner: owner)
        receipt.cancelRequested = true; receipt.cancelAcknowledged = true
        try coordinator.save(receipt)
        coordinator.publish(try detail("parent", state: "failed", cancel: false, resume: true, delete: true), owner: owner)
        let row = try #require(coordinator.jobs.first)
        let read: @MainActor @Sendable () -> Data? = { saved.data(forKey: VideoJobCoordinator.journalKey) }
        backend.stub("resumeVideoJob(id:destinationInstanceId:)") { _ in
            let data = try #require(await read())
            let record = try #require(MoldJSON.localDecoder.decode([VideoJobAdmission].self, from: data).first)
            #expect(!record.cancelRequested && !record.cancelAcknowledged)
            throw MoldClientError.unreachable("Lost resume reply")
        }
        await coordinator.resume(row, backend: backend, current: { true })
        #expect(coordinator.pending.first?.cancelRequested == false)
    }
    @Test func lateRefreshCannotUndoDismissingACompletedVideo() async throws {
        let (backend, owner, saved) = try setup()
        let coordinator = VideoJobCoordinator(defaults: saved)
        let completed = try detail("parent", state: "completed", cancel: false, delete: true)
        coordinator.publish(completed, owner: owner)
        let row = try #require(coordinator.jobs.first)
        let gate = AsyncStream<Void>.makeStream()
        backend.stub("videoJob(id:)") { _ in
            for await _ in gate.stream { break }; return completed
        }
        let old = Task { await coordinator.refresh("parent", owner: owner, backend: backend, current: { true }) }
        for _ in 0..<100 where backend.count("videoJob(id:)") == 0 { await Task.yield() }
        coordinator.dismiss(row)
        gate.continuation.finish(); await old.value
        #expect(coordinator.jobs.isEmpty)
    }

}
