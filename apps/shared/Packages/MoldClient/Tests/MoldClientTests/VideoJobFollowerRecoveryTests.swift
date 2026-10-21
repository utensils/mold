import Foundation
import Testing
import MoldClientTesting
@testable import MoldClient

extension VideoJobCoordinatorTests {
    @Test func changedOwnerReplacesPendingOldFollowerWithoutWaitingForItsSocket() async throws {
        let (oldBackend, owner, defaults) = try setup()
        let id = UUID().uuidString
        let value = try detail(id)
        let oldRead = AsyncStream<Void>.makeStream()
        oldBackend.stub("videoJob(id:)") { _ in
            for await _ in oldRead.stream { break }
            return value
        }
        var oldCurrent = true
        let coordinator = VideoJobCoordinator(defaults: defaults)
        coordinator.publish(value, owner: owner)
        coordinator.follow(.init(operationId: id, owner: owner), owner: owner, backend: oldBackend, current: { oldCurrent })
        for _ in 0..<100 where oldBackend.count("videoJob(id:)") == 0 { await Task.yield() }
        #expect(oldBackend.count("videoJob(id:)") == 1)
        let key = coordinator.identity(id, owner: owner)
        let staleFollower = try #require(coordinator.followers[key])
        var host = owner.host; host.name = "Refreshed machine metadata"
        let replacement = VideoJobOwner(host: host, instanceId: owner.instanceId)
        let newBackend = FakeBackend(host: host)
        newBackend.stub("status()", returning: try await oldBackend.status())
        newBackend.stub("capabilities()", returning: try await oldBackend.capabilities())
        newBackend.stub("videoJob(id:)", returning: value)
        oldCurrent = false
        coordinator.publish(value, owner: replacement)
        coordinator.follow(.init(operationId: id, owner: replacement), owner: replacement, backend: newBackend, current: { true })
        for _ in 0..<200 where newBackend.count("videoJob(id:)") == 0 { await Task.yield() }
        #expect(newBackend.count("videoJob(id:)") == 1, "new authority must start reading without waiting for the stale suspended request")
        oldRead.continuation.finish()
        await staleFollower.value
        #expect(coordinator.followOwners[key] == replacement, "a stale follower’s defer must not erase its successor")
        let currentToken = coordinator.followTokens[key]
        coordinator.follow(.init(operationId: id, owner: replacement), owner: replacement, backend: newBackend, current: { true })
        #expect(coordinator.followTokens[key] == currentToken, "the same live authority coalesces duplicate discovery")
        coordinator.follow(.init(operationId: id, owner: owner), owner: owner, backend: oldBackend, current: { false })
        #expect(coordinator.followTokens[key] == currentToken, "stale authority must not cancel the live reader")
        coordinator.suspend()
    }
}
