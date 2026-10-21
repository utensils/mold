import Foundation
import Testing
import MoldClientTesting
@testable import MoldClient

extension VideoJobCoordinatorTests {
    @Test func dismissalSurvivesCapturedRecoveryRediscoveryAndRelaunch() async throws {
        let (backend, owner, defaults) = try setup()
        let coordinator = VideoJobCoordinator(defaults: defaults)
        let first = VideoJobAdmission(operationId: "waiting", owner: owner)
        let second = VideoJobAdmission(operationId: "done", owner: owner)
        try coordinator.save(first); try coordinator.save(second)
        coordinator.publish(try detail("done", state: "completed", cancel: false, delete: true), owner: owner)
        let completed = try #require(coordinator.jobs.first)
        let waiting = try detail("waiting", state: "paused", cancel: false, resume: true)
        let done = try detail("done", state: "completed", cancel: false, delete: true)
        let gate = AsyncStream<Void>.makeStream()
        backend.stub("videoJob(id:)") { args in
            if args[0] as? String == "waiting" { for await _ in gate.stream { break }; return waiting }
            return done
        }
        let recovering = Task { await coordinator.recover(owner: owner, backend: backend, current: { true }) }
        for _ in 0..<100 where backend.count("videoJob(id:)") == 0 { await Task.yield() }
        coordinator.dismiss(completed)
        #expect(!coordinator.jobs.contains { $0.operationId == "done" }, "local Dismiss is immediate")
        gate.continuation.finish(); await recovering.value
        #expect(!coordinator.jobs.contains { $0.operationId == "done" }, "a captured pending loop cannot resurrect a dismissed row")
        #expect(backend.calls.filter { $0.route == "videoJob(id:)" && $0.arguments.first as? String == "done" }.isEmpty)
        await coordinator.refresh("done", owner: owner, backend: backend, current: { true })
        #expect(!coordinator.jobs.contains { $0.operationId == "done" })
        let restored = VideoJobCoordinator(defaults: defaults)
        await restored.refresh("done", owner: owner, backend: backend, current: { true })
        #expect(restored.jobs.isEmpty, "dismissal is durable across relaunch")
        let replacement = VideoJobOwner(host: owner.host, instanceId: "replacement-installation")
        restored.publish(done, owner: replacement)
        #expect(restored.jobs.first?.owner == replacement, "a different installation is not dismissed")
        coordinator.suspend(); restored.suspend()
    }

    @Test func capturedActivityFallbackCannotRecreateDismissedCompletedRow() async throws {
        let (backend, owner, defaults) = try setup()
        let coordinator = VideoJobCoordinator(defaults: defaults)
        let done = try detail("done", state: "completed", cancel: false, delete: true)
        coordinator.publish(done, owner: owner)
        let row = try #require(coordinator.jobs.first)
        let gate = AsyncStream<Void>.makeStream()
        let empty = try listing([], cursor: nil)
        backend.stub("listVideoJobs(limit:cursor:)") { _ in for await _ in gate.stream { break }; return empty }
        backend.stub("videoJob(id:)", returning: done)
        let discovering = Task { await coordinator.discover(owner: owner, backend: backend, activityIDs: ["done"], current: { true }) }
        for _ in 0..<100 where backend.count("listVideoJobs(limit:cursor:)") == 0 { await Task.yield() }
        coordinator.dismiss(row); gate.continuation.finish(); await discovering.value
        #expect(coordinator.jobs.isEmpty)
        #expect(backend.count("videoJob(id:)") == 0)
        coordinator.suspend()
    }
}
